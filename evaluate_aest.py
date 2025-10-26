#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import torch
from audiobox_aesthetics.infer  import AesPredictor

AUDIO_DIR = "./home/fundwotsai/Deep_MIR_hw2/generated_music"
OUTPUT_CSV = "./evaluation_results.csv"
device = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") or False) else "cpu"

def list_audio_files(folder):
    exts = [".wav", ".mp3", ".flac"]
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
def main():
    files = list_audio_files(AUDIO_DIR)
    print(f"found {len(files)} audios for evaluation.")

    aest = AesPredictor(checkpoint_pth=None, precision="bf16", batch_size=2, data_col="path")

    df_main = pd.read_csv(OUTPUT_CSV)
    metadata = []
    for path in tqdm(files, desc="Predicting aesthetics"):
        metadata.append({"path":path})
        
    preds = aest.forward(metadata)
    for m, out in zip(metadata, preds):
        CE = out.get("goodness") or out.get("CE")
        CU = out.get("usefulness") or out.get("CU")
        PC = out.get("production_complexity") or out.get("PC")
        PQ = out.get("production_quality") or out.get("PQ")
        # assign predictions back into df_main for the matching generated_file row
        filename = os.path.basename(m['path'])
        df_main.loc[df_main["generated_file"] == filename, "CE"] = CE
        df_main.loc[df_main["generated_file"] == filename, "CU"] = CU
        df_main.loc[df_main["generated_file"] == filename, "PC"] = PC
        df_main.loc[df_main["generated_file"] == filename, "PQ"] = PQ

    df_main.to_csv(OUTPUT_CSV, index=False)
    print(f"\nresult saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()