# Deep-Learning-for-Music-Analysis-and-Generation-Homework2

## 41147009S 陳炫佑

## Minimum Hardware Requirements

- **CPU:** AMD Ryzen 5 7500 or equivalent and above (6 cores, 12 threads)
- **Memory:** 16GB RAM or more  
- **Storage:** At least 20GB (for datasets and models)  
- **GPU (Recommended):** NVIDIA RTX 3090 or above, at least 24GB VRAM, CUDA support  
- **Operating System:** Arch Linux x86_64(Linux 6.16.10-zen1-1-zen)
- **Python Version:** 3.11.9 and 3.8.8

## envirement set up

1. use `git clone https://github.com/QwenLM/Qwen-Audio.git` for download Qwen-Audio as caption model.
2. use `git clone https://github.com/fundwotsai2001/MuseControlLite.git` for download MuseControlLite as music generation model.
3. download dataset to `./home`, the directory structure is looklike:

```text
./home
└── fundwotsai
    └── Deep_MIR_hw2
        ├── referecne_music_list_60s/
        ├── target_music_list_60s/
        └── Melody_acc.py
```

3. download python 3.8.8 and open a new envirment(I'll name it "env3.8.8") for retrival and captioning.
4. download python 3.11.9 and open a new envirment(I'll name it "env3.11.9") for generate_music.
5. download python 3.9.19 and open a new envirment(I'll name it "env3.9.19") for audiobox_aesthetics.
6. in envirment env3.8.8, use `pip3 install -r ./requirements.txt` for install needed module.
7. in envirment env3.11.9, use `pip3 install -r ./MuseControlLite/requirements.txt` for install needed module.
8. in envirment env3.9.19, use `pip3 install -r ./requirements_aest.txt` for install needed module.

## task 1 Retrieval

It will run CLAP as Retrieval model, it will show me Top-1 similar music for all target song.

please run following step in env3.8.8

run `python3 retrieval_clap.py`

and result will save as a csv in `./retrieval_results.csv`

## task 2 Generation

It will use Qwen-Audio as captioning model and use MuseControlLite as music generation model

### captioning

It will run Qwen-Audio for all target music for generate caption

here is the prompt for generation

```text
Provide a detailed description of this audio track. Include:
1) Overall genre(s) and specific subgenres or stylistic influences.
2) Estimated tempo (approximate BPM) and overall rhythmic feel (e.g., laid-back, driving, syncopated).
3) Instrumentation and arrangement: list prominent instruments, their roles (melody, harmony, rhythm), electronic vs acoustic, notable timbres.
4) Vocal characteristics if present: gender/age impression, delivery style, backing vocals/harmony, lyrical articulation.
5) Song structure and dynamics: intro, verse, chorus, bridge, build-ups, quiet/loud contrasts.
6) Rhythm details: time signature, groove, percussion patterns, presence of swing or syncopation.
7) Mood, emotional atmosphere, and lyrical themes or imagery.
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

Do not output "A)", "B)", numbers, bullets, labels, or any extra explanatory text. Do not include code blocks or any additional content—only the five plain text lines described above.
```

please run following step in env3.8.8
run `python3 audio_caption_qwen.py`

and result will saved to `./home/fundwotsai/Deep_MIR_hw2/captions/`

### generation

It will run MuseControlLite for all target music caption for generate music

please run following step in env3.11.9

run `PYTHONPATH='./MuseControlLite' python3 generate_music_musecontrol.py`

and result will saved to `home/fundwotsai/Deep_MIR_hw2/generated_music`

### evaluation

It will run CLAP, DWT and audiobox_aesthetics for evaluation the result of model

#### CLAP and DWT

please run following step in env3.8.8

run `python3 evaluate.py`

it will save evaluation result at `evaluation_results.csv`

#### audiobox_aesthetics

please run following step in env3.9.19

run `python3 evaluate_aest.py`

it will save CE,CU,PC,PQ result at `evaluation_results.csv`
