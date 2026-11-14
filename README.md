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

| Part                 | Code                             | Notebook                        |
| -------------------- | -------------------------------- | ------------------------------- |
| Data and evaluation  | `src/data.py`, `src/evaluate.py` | `notebooks/01_data.ipynb`       |
| MinHash-LSH          | `src/minhash.py`                 | `notebooks/02_lsh.ipynb`        |
| Embeddings and HNSW  | `src/embeddings.py`              | `notebooks/03_embeddings.ipynb` |
| Graph and clustering | `src/graph.py`                   | `notebooks/04_graph.ipynb`      |

## Setup

```python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `questions.csv` from the Kaggle dataset [Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset) and put it in `data/raw/` (not shared in the repository).

## Data

`notebooks/01_data.ipynb` builds every file in `data/processed/` from the raw file.

### Files

| File                         | Rows    | Content                              |
| ---------------------------- | ------- | ------------------------------------ |
| `questions.parquet`          | 537,174 | one row per distinct question        |
| `pairs.parquet`              | 403,772 | one row per clean labelled pair      |
| `questions_sample10.parquet` | 53,721  | 10% sample of the questions          |
| `pairs_sample10.parquet`     | 43,417  | the labelled pairs inside the sample |

### Columns

`questions`

- `qid`: our question id (the Kaggle ids are unreliable and not used)
- `text_norm`: question text after light normalisation (NFKC, lower case, whitespace collapsed). Any further cleaning is done in each method.
- `group_id`: ground-truth duplicate group. Questions with the same `group_id` are duplicates.
- `split`: `dev` or `test`

`pairs`

- `qid1`, `qid2`: the two questions, with `qid1 < qid2`
- `conflicting_pair`: labelled 0, but both questions are in the same duplicate group
- `split`: `dev` or `test`
- `is_duplicate`: target label, 1 or 0

### Rules for everyone

- Use the `split` column if split is needed, never a split of your own. Connected components of pairs are on one side, so no question is in both dev and test.
- Tune everything on `dev`. `test` is used once, for the final numbers in the report.
- Searches (LSH, HNSW) run over all questions, including test ones.
- The sample is for testing code. All reported numbers come from the full data.

### Loading

```python
import pandas as pd
questions = pd.read_parquet("data/processed/questions.parquet")
pairs = pd.read_parquet("data/processed/pairs.parquet")
dev_pairs = pairs[pairs["split"] == "dev"]
```

## Evaluation

`src/evaluate.py` scores the output of any method the same way.

### Candidate format

Every method returns a pandas DataFrame with one row per pair it thinks may be a duplicate:

| Column           | Type  | Meaning                                                       |
| ---------------- | ----- | ------------------------------------------------------------- |
| `qid_a`, `qid_b` | int   | the two questions (any order)                                 |
| `score`          | float | similarity (estimated Jaccard for LSH, cosine for embeddings) |
| `source`         | str   | `lsh` or `emb`                                                |

`evaluate` only uses `qid_a` and `qid_b`. `score` and `source` are needed to build the graph. Order, repeated pairs and self-pairs are handled inside. Clusters must be turned into pairs (every two questions in the same cluster) before evaluating.

### Usage

```python
import sys
sys.path.append("..")          # only when running from notebooks/
from src.evaluate import evaluate, recall_by_jaccard_bin

results = evaluate(candidates, pairs, questions, split="dev")
bins = recall_by_jaccard_bin(candidates, pairs, questions, split="dev")
```

`evaluate` returns a dictionary: `n_candidates`, `fraction_of_all_pairs`, `labelled_candidates`, `recall`, `precision`, `f1`, `recall_implied`.

- `recall`, `precision`, `f1` are computed on labelled pairs only. Precision ignores unlabelled candidates.
- `recall_implied` counts all duplicate pairs implied by the groups, labelled or not.
- `recall_by_jaccard_bin` gives recall for duplicates that share few or many words.

Use `split="dev"` while working. `split="test"` is used once, for the final numbers.

## P2: MinHash-LSH

MinHash-LSH finds likely duplicate pairs without comparing every pair of questions. Each question is represented as its set of words, and each set is summarised by a signature of **96 MinHash values**. The signature is split into bands, and two questions become a candidate pair if all values in at least one band match.

We compared four settings of bands and rows (6×16, 8×12, 12×8 and 16×6) on the dev split. We chose **16 bands of 6 rows** because it gave the highest F1. Word pairs (q=2) gave a lower F1 than single words with the same setting (17.1% against 38.6% on dev), so we kept single words.

On the **test split**, the chosen setting produced **26,906,047 candidate pairs**. That is **0.0186% of all possible pairs**. On the labelled test pairs it reached **28.6% recall** and **50.0% precision**, an **F1 of 36.4%**. Implied recall, which also counts duplicate pairs that have no label, was **18.9%**. Precision is measured only on the 17,089 candidates that have a test label, so the precision of the full candidate set is unknown.

Recall depends strongly on how many words the two questions share (test split):

| Word Jaccard    | 0–0.2  | 0.2–0.4  | 0.4–0.6   | 0.6–0.8   | 0.8–1.0   |
| --------------- | ------ | -------- | --------- | --------- | --------- |
| Duplicate pairs | 2,726  | 11,274   | 8,709     | 5,137     | 1,996     |
| Recall          | **0%** | **1.9%** | **23.4%** | **83.6%** | **99.9%** |

LSH therefore finds near-copies but misses most paraphrases. It works well as a cheap first filter, but it is not enough on its own, which is why the next stages use embeddings and graph clustering.

### P3 — Embeddings + HNSW

Encodes every normalised question with the sentence-transformers model `all-MiniLM-L6-v2` (384-dim, L2-normalised, cached in `data/processed/embeddings.npy`, not tracked by git) and indexes them with a FAISS HNSW index (M=32, inner product = cosine similarity). Each question retrieves its top-20 neighbours. `efSearch=64` was chosen by comparing against exact search on 2,000 random queries. The similarity threshold (0.75) was chosen by best F1 on the **dev** split. Output: `data/processed/emb_candidates.parquet` (~1.42M pairs, columns `qid_a < qid_b`, `score`, `source="emb"`). On the **test** split it reaches recall 0.81, precision 0.63, F1 0.71 (LSH: 0.29 / 0.50 / 0.36), and it recovers duplicates with low word overlap that LSH misses.
