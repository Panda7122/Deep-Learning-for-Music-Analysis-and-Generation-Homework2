#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.spatial.distance import cdist
from scipy.signal import medfilt
from sklearn.metrics.pairwise import cosine_similarity
import torch
from laion_clap import CLAP_Module

BASE = "./home/fundwotsai/Deep_MIR_hw2"
TARGET_DIR = os.path.join(BASE, "target_music_list_60s")
CAPTION_DIR = os.path.join(BASE, "captions")
GENERATED_DIR = os.path.join(BASE, "generated_music")
OUTPUT_CSV = "./evaluation_results.csv"
OUTPUT_JSON = "./evaluation_verbose.json"
device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or False) else "cpu"
SR = 44100
FMIN = 80
FMAX = 2000
F0_HOP_SECONDS = 0.032  # pyin hop length ~ 32ms
PITCH_CENTS_THRESHOLD = 50  # cents for framewise accuracy


def list_audio_files(folder):
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = [os.path.join(folder, f) for f in os.listdir(folder) if Path(f).suffix.lower() in exts]
    files.sort()
    return files

def read_audio(path, sr=SR):
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def extract_f0_pyin(path, sr=SR, fmin=FMIN, fmax=FMAX, hop_length=None):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if hop_length is None:
        hop_length = int(sr * F0_HOP_SECONDS)
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    f0_clean = f0.copy()
    nan_mask = np.isnan(f0_clean)
    if nan_mask.all():
        return f0_clean, voiced_flag.astype(bool), times
    idx = np.arange(len(f0_clean))
    good = ~nan_mask
    f0_clean[nan_mask] = np.interp(idx[nan_mask], idx[good], f0_clean[good])
    try:
        f0_clean = medfilt(f0_clean, kernel_size=5)
    except Exception:
        pass
    f0_clean_masked = f0_clean.copy()
    f0_clean_masked[~voiced_flag.astype(bool)] = np.nan
    return f0_clean_masked, voiced_flag.astype(bool), times

def hz_to_cents_ratio(hz_ref, hz_est):
    if np.isnan(hz_ref) or np.isnan(hz_est) or hz_ref <= 0 or hz_est <= 0:
        return np.nan
    return 1200.0 * np.log2(hz_est / hz_ref)

def framewise_pitch_accuracy(f0_target, mask_target, f0_gen, mask_gen, cents_threshold=PITCH_CENTS_THRESHOLD):
    L = min(len(f0_target), len(f0_gen))
    f0_t = f0_target[:L]
    f0_g = f0_gen[:L]
    m_t = mask_target[:L]
    m_g = mask_gen[:L]

    voiced_both = m_t & m_g
    if voiced_both.sum() == 0:
        return 0.0
    cents = []
    for a, b in zip(f0_t[voiced_both], f0_g[voiced_both]):
        if np.isnan(a) or np.isnan(b) or a <= 0 or b <= 0:
            cents.append(np.nan)
        else:
            cents.append(abs(1200.0 * math.log2(b / a)))
    cents = np.array([c for c in cents if not np.isnan(c)])
    if len(cents) == 0:
        return 0.0
    acc = np.mean(cents <= cents_threshold)
    return float(acc)

def dtw_distance_normalized(f0_target, f0_gen):
    import numpy as np
    def prepare(arr):
        arr = np.array(arr, dtype=float)
        nan_mask = np.isnan(arr)
        if nan_mask.all():
            return np.zeros_like(arr) + 1e6
        idx = np.arange(len(arr))
        good = ~nan_mask
        arr[nan_mask] = np.interp(idx[nan_mask], idx[good], arr[good])
        return arr

    a = prepare(f0_target)
    b = prepare(f0_gen)
    def to_log(x):
        x = np.array(x)
        x[x <= 1e-6] = 1e-6
        return 1200.0 * np.log2(x)
    A = to_log(a).reshape(-1, 1)
    B = to_log(b).reshape(-1, 1)
    D = cdist(A, B, metric="euclidean") 
    n, m = D.shape
    acc = np.zeros((n + 1, m + 1)) + 1e12
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = D[i - 1, j - 1]
            acc[i, j] = cost + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    dtw_dist = acc[n, m]
    norm = dtw_dist / (n + m)
    sim = max(0.0, 1.0 - (norm / 1000.0))
    sim = float(min(1.0, max(0.0, sim)))
    return sim, float(norm)

class CLAPWrapper:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = CLAP_Module(enable_fusion=False)
        print("Loading CLAP weights (this may download weights)...")
        self.model.load_ckpt()
        self.model = self.model.to(device)

    def audio_emb_from_file(self, path):
        emb = self.model.get_audio_embedding_from_filelist([path])
        if hasattr(emb, "squeeze"):
            try:
                arr = emb.squeeze().cpu().numpy()
            except Exception:
                arr = np.array(emb)
        else:
            arr = np.array(emb)
        return arr

    def text_emb_from_list(self, texts):
        if hasattr(self.model, "get_text_embedding_from_list"):
            emb = self.model.get_text_embedding_from_list(texts)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            return np.asarray(emb)
        if hasattr(self.model, "get_text_embedding"):
            emb = self.model.get_text_embedding(texts)
            if hasattr(emb, "cpu"):
                emb = emb.cpu().numpy()
            return np.asarray(emb)
        raise RuntimeError("No available text embedding method (CLAP model lacks text API and sentence-transformers not installed).")


def main():
    target_files = list_audio_files(TARGET_DIR)
    gen_files = list_audio_files(GENERATED_DIR)
    caption_files = [os.path.join(CAPTION_DIR, os.path.basename(f).replace(".wav", ".txt").replace(".mp3", ".txt")) for f in target_files]

    entries = []
    for t in target_files:
        base = Path(t).stem
        gen_candidates = [g for g in gen_files if Path(g).stem.startswith(base)]
        gen = gen_candidates[0] if gen_candidates else None
        cap_path = os.path.join(CAPTION_DIR, f"{base}.txt")
        if not os.path.exists(cap_path):
            alt_caps = [c for c in list_audio_files(CAPTION_DIR) if False]
        entries.append({"target": t, "generated": gen, "caption": cap_path if os.path.exists(cap_path) else None})


    
    clap = CLAPWrapper(device=device)
    
    results = []
    verbose = {}

    for ent in tqdm(entries, desc="Evaluating items"):
        tpath = ent["target"]
        gpath = ent["generated"]
        cpath = ent["caption"]
        base = Path(tpath).stem

        rec = {
            "target_file": os.path.basename(tpath),
            "generated_file": os.path.basename(gpath) if gpath else None,
            "caption_file": os.path.basename(cpath) if cpath else None,
            "clap_target_text": None,
            "clap_text_generated": None,
            "clap_generated_target": None,
            "CE": None, "CU": None, "PC": None, "PQ": None,
            "melody_dtw_similarity": None,
            "melody_dtw_norm": None,
            "melody_frame_accuracy": None
        }
        verbose[base] = {}

        if clap:
            audio_emb_target = clap.audio_emb_from_file(tpath)
            rec_audio_target = audio_emb_target
        else:
            audio_emb_target = None

        if cpath and os.path.exists(cpath):
            with open(cpath, "r") as f:
                caption_text = f.read().strip()
        else:
            caption_text = ""
        text_emb = clap.text_emb_from_list([caption_text])[0]
        audio_emb_gen = None
        if gpath and clap:
            audio_emb_gen = clap.audio_emb_from_file(gpath)

        def cos(a, b):
            if a is None or b is None:
                return None
            a = np.asarray(a).reshape(1, -1)
            b = np.asarray(b).reshape(1, -1)
            return float(cosine_similarity(a, b)[0, 0])

        rec["clap_target_text"] = cos(audio_emb_target, text_emb) if (audio_emb_target is not None and text_emb is not None) else None
        rec["clap_text_generated"] = cos(text_emb, audio_emb_gen) if (text_emb is not None and audio_emb_gen is not None) else None
        rec["clap_generated_target"] = cos(audio_emb_gen, audio_emb_target) if (audio_emb_gen is not None and audio_emb_target is not None) else None

        verbose[base]["audio_emb_target_shape"] = None if audio_emb_target is None else np.asarray(audio_emb_target).shape
        verbose[base]["text_emb_shape"] = None if text_emb is None else np.asarray(text_emb).shape
        verbose[base]["audio_emb_gen_shape"] = None if audio_emb_gen is None else np.asarray(audio_emb_gen).shape

        f0_t, mask_t, times_t = extract_f0_pyin(tpath)
        if gpath:
            f0_g, mask_g, times_g = extract_f0_pyin(gpath)
        else:
            f0_g, mask_g = np.array([]), np.array([])
        if len(f0_g) > 0:
            acc = framewise_pitch_accuracy(f0_t, mask_t, f0_g, mask_g, cents_threshold=PITCH_CENTS_THRESHOLD)
            rec["melody_frame_accuracy"] = acc
            dtw_sim, dtw_norm = dtw_distance_normalized(f0_t, f0_g)
            rec["melody_dtw_similarity"] = dtw_sim
            rec["melody_dtw_norm"] = dtw_norm
            verbose[base]["f0_target_preview"] = f0_t[:10].tolist()
            verbose[base]["f0_gen_preview"] = f0_g[:10].tolist()
        else:
            rec["melody_frame_accuracy"] = None
            rec["melody_dtw_similarity"] = None
            rec["melody_dtw_norm"] = None


        results.append(rec)
        verbose[base]["record"] = rec

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(verbose, f, indent=2)
    print(f"\nevaluate result is saved at： {OUTPUT_CSV} and {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
