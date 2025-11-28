# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from datetime import datetime, timezone
import yaml
import nltk
import logging

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_api")

# ---------------------------
# Load config
# ---------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SYN_PROB = config['augmentation']['synonym']['aug_p']
RS_PROB = config['augmentation'].get('random_swap', {}).get('aug_p', 0.2)
EMBED_MODEL = config['embedding']['model_name']
FAISS_INDEX_FILE = config['faiss']['index_file']
METADATA_FILE = config['faiss']['metadata_file']
TOP_K_DEFAULT = config['experiment']['top_k']

INPUT_CSV = config['data']['input_csv']
AUG_CSV = config['data']['augmented_csv']
EMBED_CSV = config['data'].get('embedded_csv', AUG_CSV)

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI(title="Semantic Search API + Augmentation")

# ---------------------------
# Download required NLTK resources
# ---------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# ---------------------------
# Load embedding model once
# ---------------------------
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
    if isinstance(aug_text, list):
        aug_text = aug_text[0]
    ts = datetime.now(timezone.utc).isoformat()
    return aug_text, "synonym_replacement", {"aug_p": SYN_PROB}, ts

def augment_random_swap(text):
    aug = naw.RandomWordAug(action="swap", aug_p=RS_PROB)
    aug_text = aug.augment(text)
    if isinstance(aug_text, list):
        aug_text = aug_text[0]
    ts = datetime.now(timezone.utc).isoformat()
    return aug_text, "random_swap", {"aug_p": RS_PROB}, ts

def compute_embedding(text):
    vec = model.encode(text)
    return normalize_vector(vec)

def batch_compute_embeddings(texts, batch_size=32):
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    embeddings = np.array([normalize_vector(e) for e in embeddings])
    return embeddings

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
    return {"message": "Welcome! Use /augment, /index, /search, /demo endpoints."}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.post("/augment")
def run_augmentation():
    try:
        df = pd.read_csv(INPUT_CSV)
        if df.empty:
            return {"message": f"No data found in {INPUT_CSV}"}
    except FileNotFoundError:
        return {"message": f"File {INPUT_CSV} not found."}

    augmented_texts, transform_types, params_list, timestamps, source_ids = [], [], [], [], []

    for _, row in df.iterrows():
        original_text = row['text']
        source_id = row['id']

        # Synonym replacement
        syn_text, t_type, params, ts = augment_synonym(original_text)
        augmented_texts.append(syn_text)
        transform_types.append(t_type)
        params_list.append(params)
        timestamps.append(ts)
        source_ids.append(source_id)

        # Random swap
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
    logger.info(f"Augmentation completed and saved to {AUG_CSV}")
    return {"message": "Augmentation completed successfully.", "saved_to": AUG_CSV, "num_rows": len(df_aug)}

@app.post("/index")
def build_index():
    try:
        df = pd.read_csv(AUG_CSV)
        if df.empty:
            return {"message": f"No data found in {AUG_CSV}"}
    except FileNotFoundError:
        return {"message": f"File {AUG_CSV} not found."}

    embeddings = batch_compute_embeddings(df['augmented_text'].tolist())
    df['embedding'] = embeddings.tolist()

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, FAISS_INDEX_FILE)
    metadata = df.to_dict(orient='records')
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"FAISS index built with {len(embeddings)} vectors")
    return {"message": f"FAISS index built with {len(embeddings)} vectors."}

@app.post("/search")
def semantic_search(query: Query):
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        return {"message": "FAISS index or metadata not found. Run /index first."}

    query_vector = compute_embedding(query.query).reshape(1, -1)
    distances, indices = index.search(query_vector, query.top_k)

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

# ---------------------------
# Demo endpoint for validation/testing
# ---------------------------
@app.get("/demo")
def run_demo():
    sample_queries = [
        "Why is data preprocessing important?",
        "What improves machine learning performance?",
        "Why is data crucial for training models?",
        "Is machine learning only about algorithms?",
        "What do ML models need for good results?"
    ]

    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    except FileNotFoundError:
        return {"message": "FAISS index or metadata not found. Run /index first."}

    demo_results = {}
    for query in sample_queries:
        query_vector = compute_embedding(query).reshape(1, -1)
        distances, indices = index.search(query_vector, TOP_K_DEFAULT)
        results = []
        for idx, sim in zip(indices[0], distances[0]):
            entry = metadata[idx]
            results.append({
                "id": entry["source_id"],
                "text": entry["augmented_text"],
                "transform_type": entry["transform_type"],
                "similarity": float(sim)
            })
        demo_results[query] = results

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo_queries": demo_results
    }

