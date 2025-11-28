# evaluate.py
from datetime import datetime, timezone
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml

# ---------------------------
# Load config
# ---------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EMBED_MODEL = config['embedding']['model_name']
FAISS_INDEX_FILE = config['faiss']['index_file']
METADATA_FILE = config['faiss']['metadata_file']
TOP_K = config['experiment']['top_k']

# ---------------------------
# Load model once
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

def compute_embedding(text):
    vec = model.encode(text)
    return normalize_vector(vec)

# ---------------------------
# Evaluate query
# ---------------------------
def evaluate(query: str, top_k: int = TOP_K):
    print(f"\n=== EVALUATION RUN @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Query: {query}")

    # Load FAISS index & metadata
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    # Compute query embedding
    query_vec = compute_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx, sim in zip(indices[0], distances[0]):
        entry = metadata[idx]
        results.append({
            "id": entry["source_id"],           # matches app.py / indexing
            "text": entry["augmented_text"],    # matches app.py / indexing
            "transform_type": entry["transform_type"],
            "params": entry.get("params", {}),
            "timestamp": entry.get("timestamp", None),
            "similarity": float(sim),
        })

    # Print top results
    print("\nTop Retrieved:")
    for r in results:
        print(f"- [{r['transform_type']}] (sim={r['similarity']:.4f}) : {r['text']}")
    return results

# ---------------------------
# Example queries
# ---------------------------
if __name__ == "__main__":
    TEST_QUERIES = [
        "Why is data preprocessing important?",
        "What improves machine learning performance?",
        "Why is data crucial for training models?",
        "Is machine learning only about algorithms?",
        "What do ML models need for good results?"
    ]
    for query in TEST_QUERIES:
        evaluate(query)
