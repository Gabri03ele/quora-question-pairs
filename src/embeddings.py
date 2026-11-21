"""Sentence embeddings and HNSW candidate generation (owner: P3)."""

import os
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
    """L2-normalised embeddings, one row per qid, cached on disk.

    The cache is reused only if it was computed with the same model
    for the same qids in the same order.
    """
    qids = np.asarray(qids)
    model_path = emb_path + '.model.txt'   # name of the model that made the cache
    cached_model = open(model_path).read().strip() if os.path.exists(model_path) else None
    if os.path.exists(emb_path) and os.path.exists(qid_path) \
            and cached_model == model_name \
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
    with open(model_path, 'w') as f:
        f.write(model_name)
    return embeddings

def build_hnsw(embeddings, M=32):
    """FAISS HNSW index on normalised vectors, so inner product equals cosine similarity."""
    index = faiss.IndexHNSWFlat(embeddings.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    start_time = time.time()
    index.add(embeddings)
    print(f'Index build time (M={M}): {time.time() - start_time:.1f}s')
    return index


def recall_at_k(approx, exact):
    """Share of the exact top-k neighbours that the approximate search also found."""
    return np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)])


def tune_ef_search(index, embeddings, k, ef_values=(16, 32, 64, 128, 256),
                   sample_size=2000, repeats=1, seed=42):
    """Recall@k and query time of HNSW vs exact search, for each efSearch value."""
    rng = np.random.default_rng(seed)
    queries = embeddings[rng.choice(len(embeddings), size=sample_size, replace=False)]

    exact_index = faiss.IndexFlatIP(embeddings.shape[1])
    exact_index.add(embeddings)
    start_time = time.time()
    _, exact_neighbours = exact_index.search(queries, k)
    exact_seconds = time.time() - start_time

    index.search(queries[:100], k)   # warm-up
    rows = []
    for ef_search in ef_values:
        index.hnsw.efSearch = ef_search
        times = []
        for _ in range(repeats):
            start_time = time.time()
            _, approx_neighbours = index.search(queries, k)
            times.append(time.time() - start_time)
        hnsw_seconds = float(np.median(times))
        rows.append({'efSearch': ef_search,
                     'recall@k': recall_at_k(approx_neighbours, exact_neighbours),
                     'time_s': hnsw_seconds,
                     'speedup_vs_exact': exact_seconds / hnsw_seconds})
    return pd.DataFrame(rows)


def build_candidates(scores, indices, qids, threshold):
    """Turn a search result into candidate pairs (qid_a < qid_b) with cosine score >= threshold.

    scores and indices have one row per question and one column per neighbour, as returned
    by index.search. Self-matches, empty results (-1) and pairs found from both sides are removed.
    """
    qids = np.asarray(qids)
    n_questions, k = indices.shape
    query_positions = np.repeat(np.arange(n_questions), k)
    neighbour_positions = indices.ravel()
    pair_scores = scores.ravel()

    keep = ((neighbour_positions >= 0) & (neighbour_positions != query_positions)
            & (pair_scores >= threshold))
    query_qids = qids[query_positions[keep]]
    neighbour_qids = qids[neighbour_positions[keep]]

    candidates = pd.DataFrame({
        'qid_a': np.minimum(query_qids, neighbour_qids),
        'qid_b': np.maximum(query_qids, neighbour_qids),
        'score': pair_scores[keep], 'source': 'emb'})
    return candidates.drop_duplicates(['qid_a', 'qid_b']).reset_index(drop=True)


def search_candidates(index, embeddings, qids, k, ef_search, threshold):
    """Search every question's k nearest neighbours and return the candidate pairs."""
    index.hnsw.efSearch = ef_search
    index.search(embeddings[:100], k)   # warm-up
    start_time = time.time()
    scores, indices = index.search(embeddings, k)
    print(f'Search time (efSearch={ef_search}): {time.time() - start_time:.1f}s')
    return build_candidates(scores, indices, qids, threshold)


def tune_threshold(candidates, pairs, questions, thresholds, split='dev'):
    """Evaluate the candidates at each threshold on one split; returns the table and the best-F1 threshold."""
    rows = [{'threshold': threshold,
             **evaluate(candidates[candidates['score'] >= threshold], pairs, questions, split=split)}
            for threshold in thresholds]
    table = pd.DataFrame(rows)
    best_threshold = table.loc[table['f1'].idxmax(), 'threshold']
    return table, best_threshold
