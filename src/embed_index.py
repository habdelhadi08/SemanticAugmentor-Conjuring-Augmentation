# full_pipeline.py
import os
import pandas as pd
import numpy as np
import nltk
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# -----------------------------
# 1️⃣ NLTK Downloads (once)
# -----------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# -----------------------------
# 2️⃣ Load your CSV
# -----------------------------
INPUT_CSV = 'data/text.csv'  # replace if needed
AUG_CSV = 'data/augmented_text.csv'
EMBED_CSV = 'data/augmented_text_with_embeddings.csv'
FAISS_INDEX_FILE = 'data/faiss_index.index'
METADATA_FILE = 'data/faiss_metadata.pkl'

df = pd.read_csv(INPUT_CSV)
print("Original data:")
print(df)

# -----------------------------
# 3️⃣ Augmentation
# -----------------------------
# Using WordNet synonym replacement
aug = naw.SynonymAug(aug_p=0.3)  # 30% of words replaced

df['augmented_text'] = df['text'].apply(lambda x: aug.augment(x))
print("Augmented data:")
print(df)

# Save augmented CSV
df.to_csv(AUG_CSV, index=False)
print(f"Augmented CSV saved: {AUG_CSV}")

# -----------------------------
# 4️⃣ Compute Embeddings
# -----------------------------
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_embedding(text):
    return model.encode(text)

df['original_embedding'] = df['text'].apply(compute_embedding)
df['augmented_embedding'] = df['augmented_text'].apply(compute_embedding)

# Save embeddings CSV
df.to_csv(EMBED_CSV, index=False)
print(f"Embeddings CSV saved: {EMBED_CSV}")

# -----------------------------
# 5️⃣ Build FAISS index
# -----------------------------
embedding_dim = df['original_embedding'][0].shape[0]

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_dim)

# Combine original + augmented embeddings
all_embeddings = np.vstack([
    np.vstack(df['original_embedding'].to_numpy()),
    np.vstack(df['augmented_embedding'].to_numpy())
]).astype('float32')

index.add(all_embeddings)
print(f"FAISS index with {index.ntotal} vectors created.")

# Metadata for retrieval
metadata = []
for idx, row in df.iterrows():
    metadata.append({'id': row['id'], 'text': row['text'], 'augmented': False})
for idx, row in df.iterrows():
    metadata.append({'id': row['id'], 'text': row['augmented_text'], 'augmented': True})

# Save FAISS index & metadata
faiss.write_index(index, FAISS_INDEX_FILE)
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(metadata, f)

print(f"FAISS index saved: {FAISS_INDEX_FILE}")
print(f"Metadata saved: {METADATA_FILE}")

print("Full pipeline completed: augmentation → embeddings → FAISS index ready!")

