# Questions Pairs Duplicate on Quora

02807 Computational Tools for Data Science, final project, Group 22.

Team: Gabriele Armani, Giovanni Carpenedo, Marcelo Sànchez, Alberto Iurlo

In this project we measure how many duplicate questions on Quora are missed using the MinHash-LSH algorithm because they are paraphrases, and test whether sentence embeddings and graph clustering recover them. Data: Quora Question Pairs (Kaggle).

## Project structure

```data/raw/ the file downloaded from Kaggle (never modified, not in git)
data/processed/   files produced by our code, e.g. cleaned questions (not in git)
src/              reusable code (functions shared by everyone)
notebooks/        jupyter notebooks
docs/             project plan and workflow PDFs
requirements.txt  Python packages
```

Rules of thumb

- Raw data is never edited. If something goes wrong, delete `data/processed/` and regenerate it.
- A function that more than one person needs goes in `src/`. Notebooks import it.
- Each notebook has a number prefix (`01_`, `02_`, ...) and each person edits only their.

## Who works where

| Part | Code | Notebook |

|---|---|---|
| Data and evaluation | `src/data.py`, `src/evaluate.py` | `notebooks/01_data.ipynb` |
| MinHash-LSH | `src/minhash.py` | `notebooks/02_lsh.ipynb` |
| Embeddings and HNSW | `src/embeddings.py` | `notebooks/03_embeddings.ipynb` |
| Graph and clustering | `src/graph.py` | `notebooks/04_graph.ipynb` |

## Setup

```python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `train.csv` from the Kaggle competition "Quora Question Pairs" (log in and accept the rules) and put it in `data/raw/`.
