"""Sentence embeddings and HNSW search (owner: P3).

TODO: encode questions, cache embeddings to disk, FAISS HNSW index, neighbour search, check against exact search.
"""

"""Sentence embeddings + HNSW candidate generation (owner: P3)."""

import os
import sys
import time
import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .evaluate import evaluate

def get_device():
    """Use the GPU if there is one: mps (Mac) or cuda (NVIDIA), otherwise cpu."""
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

def load_or_compute_embeddings(texts, qids, emb_path, qid_path,
                               model_name='all-MiniLM-L6-v2', batch_size=256):
    #L2-normalised embeddings, one row per qid cached
    # cache is reused only if it was computed for the same qids in the same order
    
    qids = np.asarray(qids)
    if os.path.exists(emb_path) and os.path.exists(qid_path) \
            and np.array_equal(np.load(qid_path), qids):
        print('Loaded cached embeddings')
        return np.load(emb_path)

    device = get_device()
    print('Encoding on', device)
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    np.save(emb_path, embeddings)
    np.save(qid_path, qids)
    return embeddings

def build_hnsw(embeddings, M=32):
    #FAISS HNSW index on normalised vectors (inner product equals cosine)
    index = faiss.IndexHNSWFlat(embeddings.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.add(embeddings)
    print(f'Index build time (M={M}): {time.time() - t0:.1f}s')
    return index


def recall_at_k(approx, exact):
    """Share of the exact top-k neighbours that the approximate search also found."""
    return np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)])


def tune_ef_search(index, embeddings, k, ef_values=(16, 32, 64, 128, 256),
                   sample_size=2000, repeats=1, seed=42):
    """Recall@k and query time of HNSW vs exact search, for each efSearch value."""
    rng = np.random.default_rng(seed)
    queries = embeddings[rng.choice(len(embeddings), size=sample_size, replace=False)]

    exact = faiss.IndexFlatIP(embeddings.shape[1])
    exact.add(embeddings)
    t0 = time.time()
    _, exact_idx = exact.search(queries, k)
    t_exact = time.time() - t0

    index.search(queries[:100], k)   # warm-up
    rows = []
    for ef in ef_values:
        index.hnsw.efSearch = ef
        times = []
        for _ in range(repeats):
            t0 = time.time()
            _, approx_idx = index.search(queries, k)
            times.append(time.time() - t0)
        t = float(np.median(times))
        rows.append({'efSearch': ef, 'recall@k': recall_at_k(approx_idx, exact_idx),
                     'time_s': t, 'speedup_vs_exact': t_exact / t})
    return pd.DataFrame(rows)

def build_candidates(scores, indices, qids, threshold):
    qids = np.asarray(qids)
    n, k = indices.shape
    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    sc = scores.ravel()

    keep = (cols >= 0) & (cols != rows) & (sc >= threshold)
    a, b = qids[rows[keep]], qids[cols[keep]]

    out = pd.DataFrame({
        'qid_a': np.minimum(a, b), 'qid_b': np.maximum(a, b),
        'score': sc[keep], 'source': 'emb'})
    return out.drop_duplicates(['qid_a', 'qid_b']).reset_index(drop=True)

def search_candidates(index, embeddings, qids, k, ef_search, threshold):
    """Search every question's k nearest neighbours and return the candidate pairs."""
    index.hnsw.efSearch = ef_search
    index.search(embeddings[:100], k)   # warm-up
    t0 = time.time()
    scores, indices = index.search(embeddings, k)
    print(f'Search time (efSearch={ef_search}): {time.time() - t0:.1f}s')
    return build_candidates(scores, indices, qids, threshold)

def tune_threshold(cands, pairs, questions, thresholds, split='dev'):
    """Evaluate the candidates at each threshold on one split; returns the table and the best-F1 threshold."""
    rows = [{'threshold': t, **evaluate(cands[cands['score'] >= t], pairs, questions, split=split)}
            for t in thresholds]
    table = pd.DataFrame(rows)
    best_t = table.loc[table['f1'].idxmax(), 'threshold']
    return table, best_t