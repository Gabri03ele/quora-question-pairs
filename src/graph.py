"""
Turns the candidate pairs of LSH (notebook 02_lsh) and embeddings (notebook 03_embeddings) into one weighted graph,
splits it into duplicate clusters (connected components or Louvain) and scores the
clusters with the same metrics as src/evaluate.py, without listing every pair.

Node ids are the qids, which are 0..n-1, so they index arrays directly.
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as _scipy_cc

from .evaluate import word_jaccard
from .evaluate import evaluate_clusters as _shared_evaluate_clusters


# ---------------------------------------------------------------------------
# 1. Graph construction
# ---------------------------------------------------------------------------

def _filter_and_weight(candidates, threshold, weighting):
    """Keep the pairs with score >= threshold and give each one a weight.

    weighting="binary": every edge weighs 1.
    weighting="scaled": the score is rescaled inside its own source so that the
    threshold maps to 0.1 and a score of 1 maps to 1. Jaccard and cosine live on
    different scales (a Jaccard of 0.5 is high, a cosine of 0.5 is low for MiniLM),
    so raw scores cannot be compared across sources.
    """
    c = candidates[candidates["score"] >= threshold]
    if weighting == "binary":
        w = np.ones(len(c), dtype=np.float64)
    elif weighting == "scaled":
        span = max(1.0 - threshold, 1e-9)
        w = 0.1 + 0.9 * (c["score"].to_numpy(np.float64) - threshold) / span
    else:
        raise ValueError("weighting must be 'binary' or 'scaled'")
    return pd.DataFrame({"u": c["qid_a"].to_numpy(np.int64),
                         "v": c["qid_b"].to_numpy(np.int64),
                         "w": w})


def build_edges(lsh=None, emb=None, lsh_threshold=0.0, emb_threshold=0.75,
                weighting="binary"):
    """One row per undirected edge: u < v, weight w, source in {lsh, emb, both}.

    A pair returned by both methods becomes a single edge with source "both"
    and the larger of the two weights.
    """
    parts = []
    if lsh is not None:
        parts.append(_filter_and_weight(lsh, lsh_threshold, weighting).assign(src=1))
    if emb is not None:
        parts.append(_filter_and_weight(emb, emb_threshold, weighting).assign(src=2))
    edges = pd.concat(parts, ignore_index=True)

    u = np.minimum(edges["u"].to_numpy(), edges["v"].to_numpy())
    v = np.maximum(edges["u"].to_numpy(), edges["v"].to_numpy())
    edges = pd.DataFrame({"u": u, "v": v, "w": edges["w"].to_numpy(),
                          "src": edges["src"].to_numpy(np.int8)})
    edges = edges[edges["u"] != edges["v"]]

    # bit-or of src: 1 = lsh, 2 = emb, 3 = both
    edges = edges.groupby(["u", "v"], sort=False, as_index=False).agg(w=("w", "max"), src=("src", "sum"))
    edges["source"] = pd.Categorical(edges["src"].map({1: "lsh", 2: "emb", 3: "both"}),
                                     categories=["lsh", "emb", "both"])
    return edges.drop(columns="src")


def to_sparse(edges, n_nodes):
    """Symmetric scipy CSR adjacency matrix."""
    u, v, w = edges["u"].to_numpy(), edges["v"].to_numpy(), edges["w"].to_numpy()
    A = sp.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n_nodes, n_nodes))
    return A.tocsr()


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------

def connected_components(edges, n_nodes):
    """Cluster label per node = connected component (transitive closure of the edges)."""
    _, labels = _scipy_cc(to_sparse(edges, n_nodes), directed=False)
    return labels.astype(np.int64)


def louvain(edges, n_nodes, resolution=1.0, seed=0, backend="igraph"):
    """Louvain communities. Isolated nodes stay singletons.

    backend="igraph" (fast, used on the full data) or "networkx" (slow, for checks).
    Louvain is randomised, so the seed is fixed for reproducibility.
    """
    if backend == "igraph":
        import igraph as ig
        import random
        random.seed(seed)          # igraph uses Python's RNG
        ig.set_random_number_generator(random)
        g = ig.Graph(n=n_nodes, edges=np.c_[edges["u"].to_numpy(), edges["v"].to_numpy()],
                     directed=False)
        part = g.community_multilevel(weights=edges["w"].to_numpy(), resolution=resolution)
        return np.asarray(part.membership, dtype=np.int64)

    if backend == "networkx":
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_weighted_edges_from(zip(edges["u"], edges["v"], edges["w"]))
        comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
        labels = np.empty(n_nodes, dtype=np.int64)
        for k, c in enumerate(comms):
            labels[list(c)] = k
        return labels

    raise ValueError("backend must be 'igraph' or 'networkx'")


# ---------------------------------------------------------------------------
# 3. Modularity (own implementation)
# ---------------------------------------------------------------------------

def modularity(edges, labels, resolution=1.0):
    """Weighted modularity  Q = sum_c [ L_c / m - gamma * (d_c / 2m)^2 ].

    m   = total edge weight
    L_c = weight of the edges inside community c
    d_c = sum of the weighted degrees of the nodes in c
    The per-community form avoids the double sum over all node pairs, so it runs
    in O(edges) and is vectorised with NumPy.
    """
    u, v, w = edges["u"].to_numpy(), edges["v"].to_numpy(), edges["w"].to_numpy(np.float64)
    labels = np.asarray(labels)
    m = w.sum()
    if m == 0:
        return 0.0
    n_comm = labels.max() + 1
    inside = labels[u] == labels[v]
    L = np.bincount(labels[u][inside], weights=w[inside], minlength=n_comm)
    d = np.bincount(labels[u], weights=w, minlength=n_comm) + np.bincount(labels[v], weights=w, minlength=n_comm)
    return float(np.sum(L / m - resolution * (d / (2 * m)) ** 2))


# ---------------------------------------------------------------------------
# 4. Evaluation of clusters (same metrics as src/evaluate.evaluate)
# ---------------------------------------------------------------------------

def _pairs_in_clusters(sizes):
    sizes = np.asarray(sizes, dtype=np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def evaluate_clusters(labels, pairs, questions, split="dev"):
    """Score a clustering as if every two questions in the same cluster were a candidate pair.

    The metrics come from the shared evaluate_clusters in src/evaluate.py, so every system in the
    project is scored by the same code. On top of them this adds precision_implied:
    the share of ALL predicted pairs that lie inside a true group. Labelled precision ignores
    unlabelled pairs, so it cannot see a giant cluster; this one can. It is a lower bound
    (unlabelled true duplicates count as errors), useful only to compare systems.
    """
    labels = np.asarray(labels)
    res = _shared_evaluate_clusters(pd.Series(labels), pairs, questions, split=split)
    qid = questions["qid"].to_numpy()
    found_all = _pairs_in_clusters(pd.DataFrame({"g": questions["group_id"].to_numpy(),
                                                 "c": labels[qid]}).value_counts())
    res["precision_implied"] = found_all / res["n_candidates"] if res["n_candidates"] else 0.0
    return res


def implied_precision_pairs(candidates, questions):
    """Implied precision for a plain candidate list (to compare pair methods with clusterings)."""
    group = questions.set_index("qid")["group_id"].to_numpy()
    a = candidates["qid_a"].to_numpy()
    b = candidates["qid_b"].to_numpy()
    keys = np.unique(np.minimum(a, b) * len(group) + np.maximum(a, b))
    a, b = keys // len(group), keys % len(group)
    keys = keys[a != b]
    a, b = keys // len(group), keys % len(group)
    return float((group[a] == group[b]).mean())


def cluster_to_pairs(labels, max_cluster_size=None):
    """All pairs inside each cluster as a candidate DataFrame (only for small clusterings / checks)."""
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    bounds = np.flatnonzero(np.diff(sorted_labels)) + 1
    a_list, b_list = [], []
    for members in np.split(order, bounds):
        k = len(members)
        if k < 2 or (max_cluster_size and k > max_cluster_size):
            continue
        i, j = np.triu_indices(k, 1)
        a_list.append(members[i])
        b_list.append(members[j])
    if not a_list:
        return pd.DataFrame({"qid_a": [], "qid_b": []}, dtype=np.int64)
    return pd.DataFrame({"qid_a": np.concatenate(a_list), "qid_b": np.concatenate(b_list)})


def recall_by_jaccard_bin_clusters(labels, pairs, questions, split="dev",
                                   bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Recall per word-Jaccard bin for a clustering (same bins as evaluate.recall_by_jaccard_bin)."""
    labels = np.asarray(labels)
    text = questions.set_index("qid")["text_norm"]
    t = pairs[(pairs["split"] == split) & (pairs["is_duplicate"] == 1)].copy()
    t["jaccard"] = [word_jaccard(a, b) for a, b in zip(t["qid1"].map(text), t["qid2"].map(text))]
    t["bin"] = pd.cut(t["jaccard"], bins=list(bins), include_lowest=True)
    t["found"] = labels[t["qid1"].to_numpy()] == labels[t["qid2"].to_numpy()]
    return t.groupby("bin", observed=False).agg(n_pairs=("found", "size"), recall=("found", "mean"))


# ---------------------------------------------------------------------------
# 5. Chaining analysis
# ---------------------------------------------------------------------------

def cluster_sizes(labels, nodes=None):
    """Size of every cluster (optionally counting only the given nodes)."""
    labels = np.asarray(labels)
    if nodes is not None:
        labels = labels[nodes]
    s = np.bincount(labels)
    return s[s > 0]


def size_summary(labels):
    s = cluster_sizes(labels)
    return {"n_clusters_ge2": int((s >= 2).sum()),
            "largest": int(s.max()),
            "nodes_in_largest_pct": 100 * s.max() / s.sum(),
            "nodes_in_clusters_ge100": int(s[s >= 100].sum())}


def cluster_purity(labels, questions):
    """For each predicted cluster of size >= 2: share of its pairs that are true duplicates
    (pairs in the same ground-truth group). Returns a DataFrame indexed by cluster."""
    labels = np.asarray(labels)
    df = pd.DataFrame({"c": labels[questions["qid"].to_numpy()], "g": questions["group_id"].to_numpy()})
    size = df["c"].value_counts()
    size = size[size >= 2]
    df = df[df["c"].isin(size.index)]
    same = df.value_counts().rename("k").reset_index()
    same["p"] = same["k"] * (same["k"] - 1) // 2
    true_pairs = same.groupby("c")["p"].sum()
    out = pd.DataFrame({"size": size})
    out["pairs"] = out["size"] * (out["size"] - 1) // 2
    out["true_pairs"] = true_pairs.reindex(out.index).fillna(0).astype(np.int64)
    out["purity"] = out["true_pairs"] / out["pairs"]
    return out.sort_values("size", ascending=False)


def bridges_in_cluster(edges, labels, cluster):
    """Bridge edges inside one cluster: removing any of them splits the cluster in two.

    Returns one row per bridge with its source and the size of the smaller side.
    Works on large clusters (100k+ nodes): all bridges are removed at once, the remaining
    2-edge-connected blocks form a tree joined by the bridges, and the size of each side
    is a subtree sum in that tree (linear time, no graph copy per bridge).
    """
    import networkx as nx
    labels = np.asarray(labels)
    inside = (labels[edges["u"].to_numpy()] == cluster) & (labels[edges["v"].to_numpy()] == cluster)
    sub = edges[inside]
    G = nx.Graph()
    G.add_edges_from(zip(sub["u"].tolist(), sub["v"].tolist()))
    source = dict(zip(zip(sub["u"].tolist(), sub["v"].tolist()), sub["source"].astype(str)))
    bridges = list(nx.bridges(G))
    if not bridges:
        return pd.DataFrame(columns=["u", "v", "source", "smaller_side"])

    # blocks = components after removing every bridge
    G.remove_edges_from(bridges)
    block = {}
    block_size = []
    for k, comp in enumerate(nx.connected_components(G)):
        block_size.append(len(comp))
        for node in comp:
            block[node] = k
    T = nx.Graph()
    T.add_nodes_from(range(len(block_size)))
    T.add_edges_from((block[u], block[v]) for u, v in bridges)

    # subtree sums in every tree of the block forest
    subtree, parent, total = {}, {}, {}
    for tree in nx.connected_components(T):
        root = next(iter(tree))
        order = list(nx.dfs_preorder_nodes(T, root))
        tree_total = sum(block_size[b] for b in tree)
        for b in order:
            total[b] = tree_total
        parent.update(nx.dfs_predecessors(T, root))
        for b in reversed(order):
            subtree[b] = block_size[b] + sum(subtree[c] for c in T[b] if parent.get(c) == b)

    rows = []
    for u, v in bridges:
        a, b = block[u], block[v]
        child = b if parent.get(b) == a else a
        side = subtree[child]
        rows.append({"u": u, "v": v,
                     "source": source.get((min(u, v), max(u, v))),
                     "smaller_side": min(side, total[child] - side)})
    return pd.DataFrame(rows).sort_values("smaller_side", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# 6. Experiments: resources, sweep, choice of setting
# ---------------------------------------------------------------------------

class Timer:
    """Wall time and peak extra memory of a block.

    with Timer() as t: ...
    t.seconds, t.peak_mb   (peak process memory above the level at the start, sampled every 20 ms;
                            includes memory used by C libraries such as igraph)
    """
    def __enter__(self):
        import threading
        try:
            import psutil
            self._proc = psutil.Process()
        except ImportError:
            self._proc = None
        self._start_rss = self._rss()
        self._peak = self._start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.t0 = time.time()
        return self

    def _rss(self):
        return self._proc.memory_info().rss if self._proc else 0

    def _sample(self):
        while not self._stop.wait(0.02):
            self._peak = max(self._peak, self._rss())

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._rss())
        self.peak_mb = (self._peak - self._start_rss) / 2**20 if self._proc else float("nan")


def run_sweep(lsh, emb, pairs, questions, lsh_thresholds=(0.6, 0.7, 0.8, 0.9),
              emb_thresholds=(0.70, 0.75, 0.8, 0.85), gammas=(1, 5, 20, 100, 300), split="dev", seed=0):
    """Connected components and Louvain for every threshold / resolution setting.

    For each partition two modularities are stored:
      Q_gamma : modularity with the same gamma Louvain optimised (what Louvain maximises)
      Q1      : standard modularity (gamma = 1), comparable across ALL rows
    Q_gamma always falls when gamma grows, by construction, so only Q1 can be compared
    across different gammas.
    """
    n = len(questions)
    rows = []
    for lt in lsh_thresholds:
        for et in emb_thresholds:
            E = build_edges(lsh, emb, lt, et)
            runs = [("components", np.nan)] + [("louvain", g) for g in gammas]
            for method, g in runs:
                with Timer() as t:
                    lab = connected_components(E, n) if method == "components" else louvain(E, n, g, seed=seed)
                r = evaluate_clusters(lab, pairs, questions, split)
                r.update(size_summary(lab))
                r.update(method=method, lsh_thr=lt, emb_thr=et, gamma=g, n_edges=len(E),
                         Q_gamma=modularity(E, lab, 1.0 if method == "components" else g),
                         Q1=modularity(E, lab, 1.0), sec=t.seconds)
                rows.append(r)
                print(f"J>={lt} cos>={et} {method:10s} gamma={g!s:5s} F1={r['f1']:.4f} largest={r['largest']}",
                      flush=True)
    return pd.DataFrame(rows)


def choose_setting(sweep, tol=0.005):
    """Pick the setting to report.

    Several settings reach practically the same labelled F1 (differences in the third decimal),
    while their largest cluster differs by a factor of 3 or more. Labelled F1 hardly penalises
    big clusters (it only looks at labelled pairs), so among the settings within `tol` of the
    best F1 we take the one with the smallest largest cluster (ties: higher F1).
    """
    best = sweep["f1"].max()
    near = sweep[sweep["f1"] >= best - tol]
    return near.sort_values(["largest", "f1"], ascending=[True, False]).iloc[0]
