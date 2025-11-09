from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from datetime import datetime
import yaml

# ---------------------------
# Load config
# ---------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SYN_PROB = config['augmentation']['synonym']['aug_p']
RS_PROB = config['augmentation']['random_swap']['aug_p'] if 'random_swap' in config['augmentation'] else 0.2
EMBED_MODEL = config['embedding']['model_name']
FAISS_INDEX_FILE = config['faiss']['index_file']
METADATA_FILE = config['faiss']['metadata_file']
TOP_K_DEFAULT = config['experiment']['top_k']

INPUT_CSV = config['data']['input_csv']
AUG_CSV = config['data']['augmented_csv']
EMBED_CSV = config['data']['embedded_csv']

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI(title="Semantic Search API + Augmentation")

# Load embedding model
model = SentenceTransformer(EMBED_MODEL)

# ---------------------------
# Helper functions
# ---------------------------
def normalize_vector(vec):
    vec = np.array(vec).astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def augment_synonym(text):
    aug = naw.SynonymAug(aug_p=SYN_PROB)
    aug_text = aug.augment(text)
    return aug_text, "synonym_replacement", {"aug_p": SYN_PROB}, datetime.utcnow().isoformat()

def augment_random_swap(text):
    aug = naw.RandomWordAug(action="swap", aug_p=RS_PROB)
    aug_text = aug.augment(text)
    return aug_text, "random_swap", {"aug_p": RS_PROB}, datetime.utcnow().isoformat()

def compute_embedding(text):
    vec = model.encode(text)
    return normalize_vector(vec)  # Normalize here for cosine similarity

# ---------------------------
# API Models
# ---------------------------
class Query(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome! Use /augment, /index, /search endpoints."}

@app.post("/augment")
def run_augmentation():
    df = pd.read_csv(INPUT_CSV)
    augmented_texts, transform_types, params_list, timestamps, source_ids = [], [], [], [], []

    for _, row in df.iterrows():
        original_text = row['text']
        source_id = row['id']

        # Apply synonym replacement
        syn_text, t_type, params, ts = augment_synonym(original_text)
        augmented_texts.append(syn_text)
        transform_types.append(t_type)
        params_list.append(params)
        timestamps.append(ts)
        source_ids.append(source_id)

        # Apply random swap
        rs_text, t_type, params, ts = augment_random_swap(original_text)
        augmented_texts.append(rs_text)
        transform_types.append(t_type)
        params_list.append(params)
        timestamps.append(ts)
        source_ids.append(source_id)

    df_aug = pd.DataFrame({
        "source_id": source_ids,
        "original_text": [df.loc[df['id'] == sid, 'text'].values[0] for sid in source_ids],
        "augmented_text": augmented_texts,
        "transform_type": transform_types,
        "params": params_list,
        "timestamp": timestamps
    })

    df_aug.to_csv(AUG_CSV, index=False)
    return {"message": f"Augmentation completed successfully.", "saved_to": AUG_CSV, "num_rows": len(df_aug)}

@app.post("/index")
def build_index():
    df = pd.read_csv(AUG_CSV)

    # Compute normalized embeddings for cosine similarity
    df['embedding'] = df['augmented_text'].apply(lambda x: compute_embedding(x).tolist())
    embeddings = np.vstack(df['embedding'].to_numpy()).astype('float32')

    # Build FAISS index for cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = Cosine for normalized vectors
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Save metadata
    metadata = df.to_dict(orient='records')
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    return {"message": f"FAISS index built with {len(embeddings)} vectors."}

@app.post("/search")
def semantic_search(query: Query):
    # Load FAISS index & metadata
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    # Embed and normalize query
    query_vector = compute_embedding(query.query).reshape(1, -1)
    distances, indices = index.search(query_vector, query.top_k)
    # distances are cosine similarities (inner product of normalized vectors)

    results = []
    for idx, sim in zip(indices[0], distances[0]):
        entry = metadata[idx]
        results.append({
            "id": entry["source_id"],
            "text": entry["augmented_text"],
            "transform_type": entry["transform_type"],
            "params": entry["params"],
            "timestamp": entry["timestamp"],
            "similarity": float(sim)
        })

    return {"query": query.query, "results": results}
