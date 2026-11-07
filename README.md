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

In P2, we used MinHash-LSH to find likely duplicate-question pairs without
comparing every possible pair directly. We tested several combinations of
bands and rows, and selected `(bands=16, rows=6)` as the best configuration
on the development set with 96 hash functions.

This configuration generated 26,906,047 candidate pairs, only `0.0186%` of
all possible question pairs. It achieved 30.6% recall and 52.2% precision on
the labelled pairs, for an F1 score of 38.6%. The implied recall, which also
accounts for duplicate pairs not present in the labelled data, was 20.9%.
The run took about 6 minutes.

The result is useful as a scalable first-stage filter: it reduces the search
space dramatically while retaining a meaningful share of duplicates. However,
the moderate recall means that LSH alone is not sufficient for the final
duplicate-detection goal, which is why the later stages use embeddings and
graph clustering to recover additional paraphrases.
