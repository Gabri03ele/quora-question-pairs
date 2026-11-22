# Duplicate questions on Quora: MinHash-LSH, embeddings and graph clustering

02807 Computational Tools for Data Science, final project, Group 22.
Gabriele Armani, Giovanni Carpenedo, Marcelo Sànchez, Alberto Iurlo.

We measure how many duplicate questions on Quora MinHash-LSH misses because they are paraphrases. We then test whether sentence embeddings with approximate nearest-neighbour search (HNSW), followed by graph clustering, recover those duplicates without comparing all pairs of questions. The method and results are in the report. Each notebook explains its own step.

## Requirements

- Python 3.13
- Packages: `pip install -r requirements.txt`. Use pandas 2.x: with pandas 3, notebook 01 lower-cases a few texts differently.
- The first run of notebook 03 downloads the embedding model from Hugging Face, so it needs internet access.
- A GPU is used automatically when one is available (Apple MPS or NVIDIA CUDA). Otherwise the code runs on the CPU.

All runs were done on a MacBook with an Apple M1 chip, using MPS. <!-- TODO: add RAM, e.g. 16 GB -->

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The dataset is the [Question Pairs Dataset](https://www.kaggle.com/datasets/quora/question-pairs-dataset) published by Quora on Kaggle. It has 404,351 labelled question pairs. To rebuild everything from scratch, download `questions.csv` and put it in `data/raw/`.

The processed files in `data/processed/` are already in the repository, so any step whose output is already there can be skipped. The only exception is `embeddings.npy` (825 MB, too large for GitHub): notebook 03 creates it the first time it runs, in about 10 minutes on the machine above. Nothing has to be downloaded by hand apart from `questions.csv`.

## How to run

Start Jupyter from the `notebooks/` folder and run the notebooks in order. Each notebook reads only files written by the ones before it. Times are for the machine above.

| Notebook                                                       | Reads                                                          | Writes                                            | Time                                                    |
| -------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------- |
| `01_data.ipynb`: cleaning, ground-truth groups, dev/test split | `data/raw/questions.csv`                                       | `questions.parquet`, `pairs.parquet`, 10% samples | about 1 min                                             |
| `02_lsh.ipynb`: MinHash-LSH candidates                         | `questions.parquet`, `pairs.parquet`                           | `lsh_candidates.parquet`                          | about 6 min                                             |
| `03_embeddings.ipynb`: embeddings and HNSW candidates          | `questions.parquet`, `pairs.parquet`, `lsh_candidates.parquet` | `embeddings.npy`, `emb_candidates.parquet`        | about 10 min to encode the first time, then about 5 min |
| `04_graph.ipynb`: similarity graph and clustering              | `questions.parquet`, `pairs.parquet`, both candidate files     | `p4_sweep_dev.csv` (only with `RUN_SWEEP = True`) | about 3 min; about 35 min with `RUN_SWEEP = True` (\*)  |

Notes:

- Notebook 03 caches the embeddings in `data/processed/embeddings.npy`. They are encoded again only if the model or the list of questions changes.
- Notebook 02 keeps up to 27 million candidate pairs in memory.
- (\*) Notebook 04 was timed on a 2-core Linux machine, not the MacBook above. By default it reloads the parameter sweep from `data/processed/p4_sweep_dev.csv` (96 settings of thresholds and Louvain resolution); set `RUN_SWEEP = True` to recompute it.
- All tuning uses the `dev` split. The `test` split is used once, at the end of notebooks 02, 03 and 04, for the numbers in the report.

### Faster check

To check the code without the full runtimes:

- `questions_sample10.parquet` and `pairs_sample10.parquet` contain 10% of the questions, with the same qids and dev/test split. The first cell of the sweep in notebook 02 already runs on them.
- Notebook 04 only reads the two candidate files in the repository, so it can be run without running 02 and 03 again.

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
notebooks/        01 to 04, run in order
```

## Group contributions

All four members contributed equally to the project.
