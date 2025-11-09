# embed_index.py
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datetime import datetime
import yaml
import random
import nltk

# -----------------------------
# 1️⃣ Load config.yaml
# -----------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Paths
INPUT_CSV = config['data']['input_csv']
AUG_CSV = config['data']['augmented_csv']
EMBED_CSV = config['data']['embedded_csv']
FAISS_INDEX_FILE = config['faiss']['index_file']
METADATA_FILE = config['faiss']['metadata_file']

# Embedding model
EMBED_MODEL = config['embedding']['model_name']

# Augmentation probabilities
SYN_PROB = config['augmentation']['synonym']['aug_p']
RI_PROB = config['augmentation']['random_insertion']['aug_p']

# Random seed
SEED = config['experiment'].get('random_seed', 42)
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 2️⃣ NLTK downloads
# -----------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# -----------------------------
# 3️⃣ Load dataset
# -----------------------------
df = pd.read_csv(INPUT_CSV)
if df.empty:
    raise ValueError(f"{INPUT_CSV} is empty or missing.")
print(f"Original data loaded: {len(df)} rows")

# -----------------------------
# 4️⃣ Define augmentation functions
# -----------------------------
def augment_synonym(text):
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=SYN_PROB)
    return aug.augment(text), 'synonym', {'aug_p': SYN_PROB}

def augment_random_insertion(text):
    aug = naw.RandomWordAug(action="insert", aug_p=RI_PROB)
    return aug.augment(text), 'random_insertion', {'aug_p': RI_PROB}

# -----------------------------
# 5️⃣ Apply augmentations
# -----------------------------
augmented_data = []

for idx, row in df.iterrows():
    original_text = row['text']
    source_id = row['id']

    # Synonym replacement
    syn_text, t_type, params = augment_synonym(original_text)
    augmented_data.append({
        'id': source_id,
        'text': syn_text,
        'augmented': True,
        'transform_type': t_type,
        'params': params,
        'timestamp': datetime.utcnow().isoformat()
    })

    # Random insertion
    ri_text, t_type, params = augment_random_insertion(original_text)
    augmented_data.append({
        'id': source_id,
        'text': ri_text,
        'augmented': True,
        'transform_type': t_type,
        'params': params,
        'timestamp': datetime.utcnow().isoformat()
    })

# Save augmented CSV
df_aug = pd.DataFrame(augmented_data)
df_aug.to_csv(AUG_CSV, index=False)
print(f"Augmented data saved: {AUG_CSV}")

# -----------------------------
# 6️⃣ Compute embeddings
# -----------------------------
model = SentenceTransformer(EMBED_MODEL)

def compute_embeddings_batch(texts, batch_size=32):
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings

texts = df_aug['text'].tolist()
embeddings = compute_embeddings_batch(texts)
df_aug['embedding'] = embeddings.tolist()
df_aug.to_csv(EMBED_CSV, index=False)
print(f"Embeddings saved: {EMBED_CSV}")

# -----------------------------
# 7️⃣ Build FAISS index
# -----------------------------
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)

# Optional: normalize embeddings if you want cosine similarity
# faiss.normalize_L2(embeddings)

all_embeddings = np.array(embeddings).astype('float32')
index.add(all_embeddings)
print(f"FAISS index with {index.ntotal} vectors created.")

# Save FAISS index & metadata
faiss.write_index(index, FAISS_INDEX_FILE)
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(augmented_data, f)

print(f"FAISS index saved: {FAISS_INDEX_FILE}")
print(f"Metadata saved: {METADATA_FILE}")
print("Pipeline completed: augmentation → embeddings → FAISS index ready!")


