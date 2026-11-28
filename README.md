# Duplicate questions on Quora: MinHash-LSH, embeddings and graph clustering

02807 Computational Tools for Data Science, final project, Group 22.
Gabriele Armani, Giovanni Carpenedo, Marcelo Sànchez, Alberto Iurlo.

We measure how many duplicate questions on Quora MinHash-LSH misses because they are paraphrases. We then test whether sentence embeddings with approximate nearest-neighbour search (HNSW), followed by graph clustering, recover those duplicates without comparing all pairs of questions. The method and results are in the report. Each notebook explains its own step.

## Requirements

- Python 3.13
- Packages: `pip install -r requirements.txt`. Use pandas 2.x: with pandas 3, notebook 01 lower-cases a few texts differently.
- The first run of notebook 03 downloads the embedding model from Hugging Face, so it needs internet access.
- A GPU is used automatically when one is available (Apple MPS or NVIDIA CUDA). Otherwise the code runs on the CPU.

All runs were done on a MacBook with an Apple M1 chip, using MPS.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The dataset is the [Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset) published by Quora on Kaggle. It has 404,351 labelled question pairs. To rebuild everything from scratch, download `questions.csv` and put it in `data/raw/`.

The processed files in `data/processed/` are already in the repository, so any step whose output is already there can be skipped. The only exception is `embeddings.npy` (825 MB, too large for GitHub): notebook 03 creates it the first time it runs, in about 10 minutes on the machine above. Nothing has to be downloaded by hand apart from `questions.csv`.

## How to run

Start Jupyter from the `notebooks/` folder and run the notebooks in order.
Times are for the machine above.

| Notebook                                                       | Reads                                                                          | Writes                                                                 | Time                                                    |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------- |
| `01_data.ipynb`: cleaning, ground-truth groups, dev/test split | `data/raw/questions.csv`                                                       | `questions.parquet`, `pairs.parquet`, 10% samples                      | about 1 min                                             |
| `02_lsh.ipynb`: MinHash-LSH candidates                         | `questions.parquet`, `pairs.parquet`                                           | `lsh_candidates.parquet`                                               | about 6 min                                             |
| `03_embeddings.ipynb`: embeddings and HNSW candidates          | `questions.parquet`, `pairs.parquet`, `lsh_candidates.parquet`                 | `embeddings.npy`, `emb_candidates.parquet`                             | about 10 min to encode the first time, then about 5 min |
| `04_graph.ipynb`: similarity graph and clustering              | `questions.parquet`, `pairs.parquet`, both candidate files                     | `p4_sweep_dev.csv` (only with `RUN_SWEEP = True`)                      | about 3 min; about 35 min with `RUN_SWEEP = True` (\*)  |
| `05_manual_check.ipynb`: manual check of 100 unlabelled pairs  | `questions.parquet`, `pairs.parquet`, both candidate files, `p4_sweep_dev.csv` | `manual_check.csv`, `manual_check_key.csv` (only if they do not exist) | about 1 min                                             |

Notes:

- Notebook 03 caches the embeddings in `data/processed/embeddings.npy`. They are encoded again only if the model or the list of questions changes.
- Notebook 02 keeps up to 27 million candidate pairs in memory.
- (\*) Notebook 04 was timed on a 2-core Linux machine, not the MacBook above. By default it reloads the parameter sweep from `data/processed/p4_sweep_dev.csv`. Set `RUN_SWEEP = True` to recompute it.
- Notebook 05 rebuilds the final clusters of notebook 04 and draws 50 embedding only pairs and 50 Louvain transitive pairs. `manual_check.csv` holds the pairs with our hand labels. `manual_check_key.csv` says which method produced each pair. It was kept closed while labelling.
- All tuning uses the `dev` split. The `test` split is used once, at the end of notebooks 02, 03 and 04, for the numbers in the report.

### Faster check

To check the code without the full runtimes:

- `questions_sample10.parquet` and `pairs_sample10.parquet` contain 10% of the questions, with the same qids and dev/test split.

## Main results

All numbers are on the `test` split. Recall, precision and F1 are measured on the labelled pairs only. For the two clustering rows, the predicted pairs are all pairs of questions inside the same cluster.

| System                       | Predicted pairs | Recall | Precision | F1     |
| ---------------------------- | --------------- | ------ | --------- | ------ |
| LSH pairs                    | 26.9 million    | 0.2863 | 0.4999    | 0.3641 |
| Embedding pairs              | 2.3 million     | 0.8051 | 0.6060    | 0.6915 |
| Union of both                | 28.8 million    | 0.8163 | 0.5735    | 0.6737 |
| Union + connected components | 12.9 billion    | 0.8825 | 0.6039    | 0.7171 |
| Union + Louvain (γ = 20)     | 24.7 million    | 0.8447 | 0.6179    | 0.7137 |

LSH misses 71.4% of the 29,842 labelled duplicate pairs in `test`. The embedding pairs recover 74.3% of the missed pairs, Union + Louvain increases that number to 80.1%. Connected components reach a similar F1, but one component holds 160,685 questions. With Louvain the largest cluster has 852 questions.

In notebook `05_manual_check` we check by hand 100 pairs that have no Quora label. We take 50 pairs found only by the embeddings: 26 of them are real duplicates (52%, 95% CI 38% to 65%). We also take 50 pairs that Louvain puts in the same cluster, although no edge links them: none of them is a duplicate (95% CI 0% to 7%). So the embeddings find many duplicates that Quora never labelled. Louvain, with this setting, instead joins questions that are not related.

## Repository structure

```
data/raw/         questions.csv from Kaggle (never modified)
data/processed/   files written by the notebooks
src/              functions used by the notebooks
  data.py         cleaning, ground truth, dev/test split
  evaluate.py     shared evaluation: recall, precision and F1 on labelled pairs, for candidate pairs and clusters
  minhash.py      our MinHash signatures and LSH banding
  embeddings.py   sentence embeddings, FAISS HNSW index, candidate search
  graph.py        similarity graph and clustering
  manual_check.py sample, export and score the pairs for the manual check
notebooks/        01 to 05, run in order
```

## Group contributions

Each member owned one part of the pipeline, with its module and notebook.

| Member             | Part                 | Code and notebooks                                                           | Work                                                                                                  |
| ------------------ | -------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Gabriele Armani    | Data and evaluation  | `src/data.py`, `src/evaluate.py`, `src/manual_check.py`, notebooks 01 and 05 | Cleaning, ground-truth groups, dev/test split, EDA, shared evaluation code, manual check of 100 pairs |
| Giovanni Carpenedo | MinHash-LSH          | `src/minhash.py`, notebook 02                                                | Vectorised MinHash and LSH banding, parameter study, check against datasketch, cost against all pairs |
| Alberto Iurlo      | Embeddings and HNSW  | `src/embeddings.py`, notebook 03                                             | Choice of the sentence encoder, FAISS HNSW index, tuning of K, efSearch and the cosine threshold      |
| Marcelo Sànchez    | Graph and clustering | `src/graph.py`, notebook 04                                                  | Similarity graph, Louvain clustering and modularity, parameter sweep on dev, final test results       |

All four members wrote the report together.
