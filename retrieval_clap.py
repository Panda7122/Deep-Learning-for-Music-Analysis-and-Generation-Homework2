import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from laion_clap import CLAP_Module

target_dir = "./home/fundwotsai/Deep_MIR_hw2/target_music_list_60s"
ref_dir = "./home/fundwotsai/Deep_MIR_hw2/referecne_music_list_60s"
output_csv = "retrieval_results.csv"

print("loading CLAP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLAP_Module(enable_fusion=False)
model.load_ckpt()
model = model.to(device)
model.eval()
def get_audio_files(folder):
    exts = [".wav", ".mp3", ".flac"]
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]

target_files = sorted(get_audio_files(target_dir))
ref_files = sorted(get_audio_files(ref_dir))

print(f"Found {len(target_files)} target audio files and {len(ref_files)} reference audio files.")

def get_embeddings(file_list):
    embeddings = []
    for path in tqdm(file_list, desc="Extracting embeddings"):
        emb = model.get_audio_embedding_from_filelist([path])
        # emb may be a numpy.ndarray or a torch.Tensor; handle both safely
        if isinstance(emb, torch.Tensor):
            arr = emb.squeeze().cpu().numpy()
        else:
            arr = np.squeeze(emb)
        embeddings.append(arr)
    # return a numpy array for sklearn.cosine_similarity compatibility
    return np.stack(embeddings, axis=0)

target_emb = get_embeddings(target_files)
ref_emb = get_embeddings(ref_files)

print("calculate cosine similarity ...")
similarity_matrix = cosine_similarity(target_emb, ref_emb)

top_k = 1
results = []
for i, target_name in enumerate(target_files):
    top_idx = similarity_matrix[i].argsort()[::-1][:top_k]
    for rank, j in enumerate(top_idx):
        results.append({
            "target_file": os.path.basename(target_name),
            "rank": rank + 1,
            "ref_file": os.path.basename(ref_files[j]),
            "similarity": round(float(similarity_matrix[i][j]), 4)
        })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"retrieval result saved to {output_csv}")
