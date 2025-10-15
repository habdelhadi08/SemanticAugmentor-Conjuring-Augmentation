from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import dist

# ---------------------------
# Load FAISS index and metadata
# ---------------------------
index = faiss.read_index("data/faiss_index.index")
with open("data/faiss_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)  # should be a list of dicts: [{"id":..., "text":..., "augmented_text":...}, ...]

# Load embedding model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI(title="Semantic Search API")

class Query(BaseModel):
    query: str
    top_k: int = 5

# Health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the Semantic Search API! Use /search to query."}

# Normalize vectors (important for cosine similarity)
def normalize_vector(vec):
    vec = np.array(vec).astype("float32")
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

@app.post("/search")
def semantic_search(query: Query):
    # Embed and normalize the user query
    query_vector = normalize_vector(model.encode([query.query])[0]).reshape(1, -1)
    
    # Search FAISS index
    distances, indices = index.search(query_vector, query.top_k)
    
    # Convert distances to similarity (cosine similarity)
    similarities = 1 - distances  # if index uses L2 on normalized vectors
    
    # Retrieve metadata
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        entry = metadata[idx]
        results.append({
            "id": entry["id"],
            "text": entry["text"],
            "augmented": entry["augmented"],
            "similarity": float(sim)
        })
    
    return {"query": query.query, "results": results}



