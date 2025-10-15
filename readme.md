# Conjuring Augmentation

This project implements a **text augmentation and semantic search pipeline** using **FastAPI**, **FAISS**, and **SentenceTransformers**.  
It takes a small text dataset, augments it using synonym replacement, embeds both original and augmented data, indexes them in a FAISS vector store, and exposes a simple API for semantic search.

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
│
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

| Endpoint  | Method | Description             |
| --------- | ------ | ----------------------- |
| `/`       | GET    | Root message            |
| `/search` | POST   | Perform semantic search |

#### Example Request

POST /search

curl -X POST "http://127.0.0.1:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?", "top_k": 5}'

#### Example Response:

{"query":"What is machine learning?","results":[
    {"id":1,"text":"Machine learning is not just about algorithms","augmented":false,"similarity":-29.568674087524414},{"id":3,"text":"Data preprocessing live all important for exemplar performance","augmented":true,"similarity":-32.94023132324219},{"id":2,"text":"Data is crucial for training machine learning models","augmented":false,"similarity":-34.68223571777344},{"id":3,"text":"Data preprocessing is essential for model performance","augmented":false,"similarity":-36.97655487060547},{"id":1,"text":"Machine encyclopedism is non just about algorithm","augmented":true,"similarity":-38.00421905517578}]}


## Evaluation Notes
- Compared retrieval performance between original and augmented data.

- Observation: Synonym-based augmentation improves search recall for paraphrased queries.

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





