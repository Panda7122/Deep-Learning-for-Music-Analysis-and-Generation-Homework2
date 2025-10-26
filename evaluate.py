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
import torchaudio
import numpy as np
from scipy.signal import savgol_filter
import librosa
import scipy.signal as signal
from torchaudio import transforms as T
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

def extract_melody_one_hot(audio_path,
                           sr=44100,
                           cutoff=261.2, 
                           win_length=2048,
                           hop_length=256):
    """
    Extract a one-hot chromagram-based melody from an audio file (mono).
    
    Parameters:
    -----------
    audio_path : str
        Path to the input audio file.
    sr : int
        Target sample rate to resample the audio (default: 44100).
    cutoff : float
        The high-pass filter cutoff frequency in Hz (default: Middle C ~ 261.2 Hz).
    win_length : int
        STFT window length for the chromagram (default: 2048).
    hop_length : int
        STFT hop length for the chromagram (default: 256).
    
    Returns:
    --------
    one_hot_chroma : np.ndarray, shape=(12, n_frames)
        One-hot chromagram of the most prominent pitch class per frame.
    """
    # ---------------------------------------------------------
    # 1. Load audio (Torchaudio => shape: (channels, samples))
    # ---------------------------------------------------------
    audio, in_sr = torchaudio.load(audio_path)

    # Convert to mono by averaging channels: shape => (samples,)
    audio_mono = audio.mean(dim=0)

    # Resample if necessary
    if in_sr != sr:
        resample_tf = T.Resample(orig_freq=in_sr, new_freq=sr)
        audio_mono = resample_tf(audio_mono)

    # Convert torch.Tensor => NumPy array: shape (samples,)
    y = audio_mono.numpy()

    # ---------------------------------------------------------
    # 2. Design & apply a high-pass filter (Butterworth, order=2)
    # ---------------------------------------------------------
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = signal.butter(N=2, Wn=norm_cutoff, btype='high', analog=False)
    
    # filtfilt expects shape (n_samples,) for 1D
    y_hp = signal.filtfilt(b, a, y)

    # ---------------------------------------------------------
    # 3. Compute the chromagram (librosa => shape: (12, n_frames))
    # ---------------------------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y_hp,
        sr=sr,
        n_fft=win_length,      # Usually >= win_length
        win_length=win_length,
        hop_length=hop_length
    )

    # ---------------------------------------------------------
    # 4. Convert chromagram to one-hot via argmax along pitch classes
    # ---------------------------------------------------------
    # pitch_class_idx => shape=(n_frames,)
    pitch_class_idx = np.argmax(chroma, axis=0)

    # Make a zero array of the same shape => (12, n_frames)
    one_hot_chroma = np.zeros_like(chroma)

    # For each frame (column in chroma), set the argmax row to 1
    one_hot_chroma[pitch_class_idx, np.arange(chroma.shape[1])] = 1.0
    
    return one_hot_chroma
def melody_score(target_audio_path, generated_audio_path):
    gt_melody = extract_melody_one_hot(target_audio_path)      
    gen_melody = extract_melody_one_hot(generated_audio_path)
    min_len_melody = min(gen_melody.shape[1], gt_melody.shape[1])
    matches = ((gen_melody[:, :min_len_melody] == gt_melody[:, :min_len_melody]) & (gen_melody[:, :min_len_melody] == 1)).sum()
    accuracy = matches / min_len_melody
    return accuracy
def list_audio_files(folder):
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files = [os.path.join(folder, f) for f in os.listdir(folder) if Path(f).suffix.lower() in exts]
    files.sort()
    return files
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
            "melody_accuracy":None
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

        ms = melody_score(tpath, gpath)
        rec["melody_accuracy"] = ms
        


        results.append(rec)
        verbose[base]["record"] = rec

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(verbose, f, indent=2)
    print(f"\nevaluate result is saved at： {OUTPUT_CSV} and {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
