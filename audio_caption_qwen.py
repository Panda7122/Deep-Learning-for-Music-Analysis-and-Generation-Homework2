import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
target_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s"
caption_dir = "./home/fundwotsai/Deep_MIR_hw2/captions"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading Qwen2-Audio ({device}) ...")

model_name = "Qwen/Qwen2-Audio-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()

def get_audio_files(folder):
    exts = [".wav", ".mp3", ".flac"]
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]

audio_files = sorted(get_audio_files(target_dir))
print(f"Found {len(audio_files)} target audio files.")

for path in tqdm(audio_files, desc="Generating captions"):
    file_name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(caption_dir, f"{file_name}.txt")

    prompt = """
This audio contains no vocals. Treat the track as fully instrumental—do not infer or invent any vocal content. Where vocal information would normally be reported, explicitly state "no vocals".

Provide a detailed description of this audio track. Include:
1) Overall genre(s) and specific subgenres or stylistic influences.
2) Estimated tempo (approximate BPM) and overall rhythmic feel (e.g., laid-back, driving, syncopated).
3) Instrumentation and arrangement: list prominent instruments, their roles (melody, harmony, rhythm), electronic vs acoustic, notable timbres.
4) Vocal characteristics if present: if there are no vocals, explicitly state "no vocals".
5) Song structure and dynamics: intro, verse, chorus, bridge, build-ups, quiet/loud contrasts.
6) Rhythm details: time signature, groove, percussion patterns, presence of swing or syncopation.
7) Mood, emotional atmosphere, and lyrical themes or imagery (if no vocals, describe mood and any implied themes).
8) Production and sonic qualities: effects (reverb, delay, distortion), stereo placement, cleanliness vs lo-fi, notable mixing/mastering traits.
9) Possible cultural, regional, or historical influences and comparable artists.
10) Five concise tags/keywords that summarize the track.
11) Instruments used: list specific instruments heard and their roles (if different from #3, be explicit).
12) Estimated key/tonality: likely musical key (e.g., C major, A minor) or modal/atonal character if applicable.

Finally, produce exactly five lines of plain text with no additional labels or explanation:
Line 1: One short caption (one sentence).
Line 2: A 2-3 sentence detailed descriptive paragraph using concrete, specific descriptors.
Line 3: Instruments used: <list instruments>.
Line 4: Estimated key/tonality: <predicted key or tonal description>.

Do not output "A)", "B)", numbers, bullets, labels, or any extra explanatory text. Do not include code blocks or any additional content—only the five plain text lines described above."""
    query = tokenizer.from_list_format([
        {'audio': path}, # Either a local path or an url
        {'text': prompt},
    ])
    with torch.no_grad():
        response, history = model.chat(tokenizer, query=query, history=None)

    if not os.path.isdir(caption_dir):
        os.makedirs(caption_dir, exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(response.strip())

    tqdm.write(f"done {file_name}:\n {response.strip()}")

print(f"\nall caption saves to {caption_dir}/ ")
