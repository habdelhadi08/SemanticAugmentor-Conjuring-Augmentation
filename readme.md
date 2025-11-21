# Conjuring Augmentation

This project implements a text augmentation and semantic search pipeline using FastAPI, FAISS, and SentenceTransformers.
It takes a small text dataset, augments it using synonym replacement and random insertion, embeds both original and augmented data, indexes them in a FAISS vector store, and exposes a simple API for semantic search.

---

## Project Structure

```
SemanticAugmentor-ConjuringAaugmentation/
│
├── data/
│ ├── text.csv # Original dataset
│ ├── augmented_text.csv # Augmented text data
│ ├── faiss_index.index # FAISS vector index
│ ├── faiss_metadata.pkl # Metadata for indexed entries
│ ├──augmented_text_with_embeddings.csv
│
├── src/
│ ├── app.py # FastAPI semantic search API
│ ├── augment.py # Text augmentation script
│ ├── embed_index.py # Full pipeline: embeddings + FAISS index
│── config.yaml
├── requirements.txt
└── README.md
```

---

## Installation

### 1️⃣ Create and activate a virtual environment

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

## 2️⃣ Install dependencies

```bash 
pip install -r requirements.txt
```

## Step 1 — Run Text Augmentation

```bash
python src/augment.py
```

#### What it does:

- Loads data/text.csv

- Applies synonym-based text augmentation using nlpaug

- Saves the augmented results to data/augmented_text.csv

## Step 2 — Build Embeddings & FAISS Index

```bash
python src/embed_index.py
```
#### What it does:

- Loads the original and augmented text

- Computes embeddings using SentenceTransformer('paraphrase-MiniLM-L6-v2')

- Builds a FAISS index for efficient semantic search

#### Saves:

- FAISS index → data/faiss_index.index

- Metadata → data/faiss_metadata.pkl

## Step 3 — Run the FastAPI App

```bash
uvicorn src.app:app --reload
```

#### Open in your browser:
http://127.0.0.1:8000

### API Endpoints

| Endpoint   | Method | Description             |
| ---------- | ------ | ----------------------- |
| `/`        | GET    | Root message            |
| `/augment` | POST   | Run text augmentation   |
| `/index`   | POST   | Build FAISS index       |
| `/search`  | POST   | Perform semantic search |


#### Example Request

POST /search

curl -X POST "http://127.0.0.1:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "top_k": 5}'

#### Example Response:

 body
Download
{
  "query": "What is machine learning?",
  "results": [
    {
      "id": 1,
      "text": "Machine learning is not just about algorithms",
      "transform_type": "random_swap",
      "params": "{'aug_p': 0.2}",
      "timestamp": "2025-11-21T15:48:14.738508",
      "similarity": 0.7713919281959534
    },
    {
      "id": 2,
      "text": "Data crucial is for machine training learning models",
      "transform_type": "random_swap",
      "params": "{'aug_p': 0.2}",
      "timestamp": "2025-11-21T15:48:14.782004",
      "similarity": 0.6956778168678284
    },
    {
      "id": 2,
      "text": "Data be crucial for take machine learning models",
      "transform_type": "synonym_replacement",
      "params": "{'aug_p': 0.3}",
      "timestamp": "2025-11-21T15:48:14.782004",
      "similarity": 0.6451414823532104
    },
    {
      "id": 1,
      "text": "Machine acquisition is not just astir algorithm",
      "transform_type": "synonym_replacement",
      "params": "{'aug_p': 0.3}",
      "timestamp": "2025-11-21T15:48:14.738508",
      "similarity": 0.5486204624176025
    },
    {
      "id": 3,
      "text": "Data preprocessing is essential for model operation",
      "transform_type": "synonym_replacement",
      "params": "{'aug_p': 0.3}",
      "timestamp": "2025-11-21T15:48:14.782004",
      "similarity": 0.3149666488170624
    }
  ]
}
   


## Evaluation Notes
- Compared retrieval performance between original and augmented data.

- Observation: Synonym and random swap augmentation improve search recall for paraphrased queries.

## Evaluation Results

Below are the semantic search evaluation results from running evaluate.py using augmented data (synonym replacement + random swap):

| Query                                           | Top 1 Result (Method)                                                     | Sim    | Top 2 Result (Method)                                                                         | Sim    | Top 3 Result (Method)                                                                         | Sim    |
| ----------------------------------------------- | ------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------- | ------ |
| **Why is data preprocessing important?**        | Preprocessing data is essential model for performance *(random_swap)*     | 0.7634 | Information preprocessing make up all important for model performance *(synonym_replacement)* | 0.6565 | Data is crucial training for learning machine models *(random_swap)*                          | 0.4448 |
| **What improves machine learning performance?** | Machine learning follow not only astir algorithms *(synonym_replacement)* | 0.6218 | Data is crucial training for learning machine models *(random_swap)*                          | 0.6202 | Information preprocessing make up all important for model performance *(synonym_replacement)* | 0.4843 |
| **Why is data crucial for training models?**    | Data is crucial training for learning machine models *(random_swap)*      | 0.8329 | Data is essential for train automobile learning models *(synonym_replacement)*                | 0.6031 | Information preprocessing make up all important for model performance *(synonym_replacement)* | 0.5584 |
| **Is machine learning only about algorithms?**  | Learning machine is just not about algorithms *(random_swap)*             | 0.8052 | Machine learning follow not only astir algorithms *(synonym_replacement)*                     | 0.7060 | Data is crucial training for learning machine models *(random_swap)*                          | 0.5188 |
| **What do ML models need for good results?**    | Data is crucial training for learning machine models *(random_swap)*      | 0.5546 | Information preprocessing make up all important for model performance *(synonym_replacement)* | 0.4844 | Preprocessing data is essential model for performance *(random_swap)*                         | 0.4403 |

## Interpretation

- Augmented data significantly improves the similarity scores.

- Synonym replacement + random swap produce strong agreement with paraphrased queries.

- Even when queries are phrased differently, top-1 and top-3 results remain relevant.

- Random swap often produces the highest similarity on data-centric queries.

- Augmentation increases robustness and semantic recall compared to baseline text.

## Key Features

✅ Text augmentation using WordNet synonyms

✅ Embedding generation with SentenceTransformers

✅ Vector indexing with FAISS

✅ FastAPI interface for real-time semantic search

### Author
Heba Abdelhadi

AI Application Developer & Data Scientist

habdelhadi08@gmail.com

Shelby Township, MI





