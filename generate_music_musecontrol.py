import os
import torch
from tqdm import tqdm
import torch
import numpy as np
from MuseControlLite.config_inference import get_config
import argparse
import json
import soundfile as sf
from diffusers import StableAudioPipeline

from MuseControlLite.MuseControlLite_setup import (
    setup_MuseControlLite,
    initialize_condition_extractors,
    evaluate_and_plot_results,
    load_audio_file,
    process_musical_conditions
)


caption_dir = "./home/fundwotsai/Deep_MIR_hw2/captions"
output_dir = "./home/fundwotsai/Deep_MIR_hw2/generated_music"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading MuseControlLite ({device}) ...")

weight_dtype = torch.float32
stable_audio = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype)
stable_audio = stable_audio.to(device)

def get_caption_files(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]

caption_files = sorted(get_caption_files(caption_dir))
print(f"Found {len(caption_files)} captions。")
config = get_config()
negative_text_prompt = config["negative_text_prompt"]

for path in tqdm(caption_files, desc="Generating music"):
    file_name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(output_dir, f"{file_name}_generated.wav")

    with open(path, "r") as f:
        prompt = f.read().strip()

    tqdm.write(f"\ngenerating {file_name}.wav")
    tqdm.write(f"Prompt: {prompt}")

    audio = stable_audio(
        prompt=prompt,
        negative_prompt=negative_text_prompt,
        num_inference_steps=config["denoise_step"],
        guidance_scale=config["guidance_scale_text"],
        num_waveforms_per_prompt=1,
        audio_end_in_s=2097152/44100,
        generator = torch.Generator().manual_seed(42)
    ).audios
    output = audio[0].T.float().cpu().numpy()
    sf.write(out_path, output, stable_audio.vae.sampling_rate) 

print(f"\nall generated musics are saved to {output_dir}/")
