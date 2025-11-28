import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
from datetime import datetime, timezone
import yaml

# -----------------------------
# Load configuration
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

SYN_PROB = config['augmentation']['synonym']['aug_p']
RS_PROB = config['augmentation'].get('random_swap', {}).get('aug_p', 0.2)

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
if df.empty:
    raise ValueError(f"{INPUT_CSV} is empty or missing.")
print(f"Original data loaded: {len(df)} rows")

# -----------------------------
# Define augmentation functions
# -----------------------------
def augment_synonym(text):
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=SYN_PROB)
    aug_text = aug.augment(text)
    if isinstance(aug_text, list):  # nlpaug sometimes returns a list
        aug_text = aug_text[0]
    return aug_text, 'synonym_replacement', {'aug_p': SYN_PROB}, datetime.now(timezone.utc).isoformat()


def augment_random_swap(text):
    aug = naw.RandomWordAug(action="swap", aug_p=RS_PROB)
    aug_text = aug.augment(text)
    if isinstance(aug_text, list):
        aug_text = aug_text[0]
    return aug_text, 'random_swap', {'aug_p': RS_PROB}, datetime.now(timezone.utc).isoformat()

# -----------------------------
# Apply augmentations
# -----------------------------
augmented_rows = []

for idx, row in df.iterrows():
    original_text = row['text']
    source_id = row['id']

    # Synonym replacement
    syn_text, t_type, params, ts = augment_synonym(original_text)
    augmented_rows.append({
        "source_id": source_id,
        "original_text": original_text,
        "augmented_text": syn_text,
        "transform_type": t_type,
        "params": params,
        "timestamp": ts,
        "augmented": True
    })

    # Random swap
    rs_text, t_type, params, ts = augment_random_swap(original_text)
    augmented_rows.append({
        "source_id": source_id,
        "original_text": original_text,
        "augmented_text": rs_text,
        "transform_type": t_type,
        "params": params,
        "timestamp": ts,
        "augmented": True
    })

# -----------------------------
# Save augmented data
# -----------------------------
aug_df = pd.DataFrame(augmented_rows)
aug_df.to_csv(AUG_CSV, index=False)
print(f"✅ Augmented data saved with metadata to '{AUG_CSV}'")



