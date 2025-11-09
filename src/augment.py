# augment.py
import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
from datetime import datetime
import yaml

# -----------------------------
# Load configuration
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

syn_aug_p = config['augmentation']['synonym']['aug_p']
ins_aug_p = config['augmentation'].get('insertion', {}).get('aug_p', 0.1)
del_aug_p = config['augmentation'].get('deletion', {}).get('aug_p', 0.1)

INPUT_CSV = config['data']['input_csv']
AUG_CSV = config['data']['augmented_csv']

# -----------------------------
# Download required NLTK resources
# -----------------------------
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# -----------------------------
# Load CSV file
# -----------------------------
df = pd.read_csv(INPUT_CSV)

# -----------------------------
# Initialize augmenters
# -----------------------------
syn_aug = naw.SynonymAug(aug_src='wordnet', aug_p=syn_aug_p)
ins_aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert', aug_p=ins_aug_p)
del_aug = naw.RandomWordAug(action='delete', aug_p=del_aug_p)

# -----------------------------
# Apply augmentations
# -----------------------------
augmented_rows = []

for idx, row in df.iterrows():
    original_text = row['text']

    # Synonym replacement
    syn_text = syn_aug.augment(original_text)
    augmented_rows.append({
        "source_id": row['id'],
        "original_text": original_text,
        "augmented_text": syn_text,
        "transform_type": "synonym_replacement",
        "params": {"aug_p": syn_aug_p},
        "timestamp": datetime.utcnow().isoformat()
    })

    # Random insertion
    ins_text = ins_aug.augment(original_text)
    augmented_rows.append({
        "source_id": row['id'],
        "original_text": original_text,
        "augmented_text": ins_text,
        "transform_type": "random_insertion",
        "params": {"aug_p": ins_aug_p},
        "timestamp": datetime.utcnow().isoformat()
    })

    # Random deletion
    del_text = del_aug.augment(original_text)
    augmented_rows.append({
        "source_id": row['id'],
        "original_text": original_text,
        "augmented_text": del_text,
        "transform_type": "random_deletion",
        "params": {"aug_p": del_aug_p},
        "timestamp": datetime.utcnow().isoformat()
    })

# -----------------------------
# Save augmented data
# -----------------------------
aug_df = pd.DataFrame(augmented_rows)
aug_df.to_csv(AUG_CSV, index=False)
print(f"✅ Augmented data saved with metadata to '{AUG_CSV}'")

