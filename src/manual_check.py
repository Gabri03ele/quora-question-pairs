"""Manual check of unlabelled candidate pairs 
samples unlabelled pairs from the final system of clustering, 
writes them to a sheet for labelling by hand, and scores the labels. 
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .graph import build_edges, choose_setting, louvain

VALID_LABELS = ("1", "0")      # 1 = duplicate, 0 = not a duplicate


def pair_keys(qid_a, qid_b, n_questions):
    qid_a = np.asarray(qid_a, dtype=np.int64)
    qid_b = np.asarray(qid_b, dtype=np.int64)
    return np.minimum(qid_a, qid_b) * n_questions + np.maximum(qid_a, qid_b)

def is_unlabelled(qid_a, qid_b, pairs, questions):
    """True for pairs that have no Quora label (neither 1 nor 0) and are not in the same true group."""
    n_questions = len(questions)
    labelled_keys = pair_keys(pairs["qid1"], pairs["qid2"], n_questions)
    has_label = np.isin(pair_keys(qid_a, qid_b, n_questions), labelled_keys)
    group_of_question = questions["group_id"].to_numpy() # qids are 0..n-1
    same_group = group_of_question[qid_a] == group_of_question[qid_b]
    return ~has_label & ~same_group


def rebuild_final_clusters(lsh_candidates, emb_candidates, n_questions, sweep_path, seed=0):
    chosen = choose_setting(pd.read_csv(sweep_path))
    setting = {"lsh_thr": float(chosen["lsh_thr"]), "emb_thr": float(chosen["emb_thr"]),
               "gamma": float(chosen["gamma"])}
    edges = build_edges(lsh_candidates, emb_candidates, setting["lsh_thr"], setting["emb_thr"])
    labels = louvain(edges, n_questions, setting["gamma"], seed=seed)
    return setting, edges, labels


def embedding_only_pool(emb_candidates, lsh_candidates, pairs, questions, emb_threshold):
    """Unlabelled embedding pairs above the threshold that LSH never returned (at any score)."""
    n_questions = len(questions)
    strong = emb_candidates[emb_candidates["score"] >= emb_threshold]
    keys = pair_keys(strong["qid_a"], strong["qid_b"], n_questions)
    keys, first_row = np.unique(keys, return_index=True)      # one row per unordered pair
    cosine = strong["score"].to_numpy()[first_row]

    lsh_keys = pair_keys(lsh_candidates["qid_a"], lsh_candidates["qid_b"], n_questions)
    not_in_lsh = ~np.isin(keys, lsh_keys)
    keys, cosine = keys[not_in_lsh], cosine[not_in_lsh]

    pool = pd.DataFrame({"qid_a": keys // n_questions, "qid_b": keys % n_questions, "cosine": cosine})
    return pool[is_unlabelled(pool["qid_a"], pool["qid_b"], pairs, questions)].reset_index(drop=True)


def louvain_transitive_sample(labels, edges, pairs, questions, n_sample, seed, n_draws=20_000):
    """Uniform sample of unlabelled pairs that share a Louvain cluster but are not an edge.

    so pairs are drawn: cluster with probability proportional to its number of pairs 
    Draws that are edges or labelled are rejected, which keeps the sample uniform
    Returns the sample and the share of draws kept (the share of cluster pairs in the pool).
    """
    rng = np.random.default_rng(seed)
    n_questions = len(questions)
    labels = np.asarray(labels)
    members_by_cluster = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels)
    first_member = np.r_[0, np.cumsum(sizes)[:-1]]
    pairs_per_cluster = sizes * (sizes - 1) // 2

    cluster = rng.choice(len(sizes), size=n_draws, p=pairs_per_cluster / pairs_per_cluster.sum())
    i = rng.integers(0, sizes[cluster])
    j = rng.integers(0, sizes[cluster] - 1)
    j = j + (j >= i)                                           # second member different from the first
    qid_a = members_by_cluster[first_member[cluster] + i]
    qid_b = members_by_cluster[first_member[cluster] + j]

    is_edge = np.isin(pair_keys(qid_a, qid_b, n_questions), pair_keys(edges["u"], edges["v"], n_questions))
    keep = ~is_edge & is_unlabelled(qid_a, qid_b, pairs, questions)

    kept_keys = pair_keys(qid_a[keep], qid_b[keep], n_questions)
    _, first_draw = np.unique(kept_keys, return_index=True)
    kept_keys = kept_keys[np.sort(first_draw)][:n_sample]      # drop repeats, keep draw order
    if len(kept_keys) < n_sample:
        raise ValueError(f"only {len(kept_keys)} pairs kept; increase n_draws")
    sample = pd.DataFrame({"qid_a": kept_keys // n_questions, "qid_b": kept_keys % n_questions})
    return sample, keep.mean()


def build_sheet(samples_by_method, questions, seed):
    text = questions["text_norm"].to_numpy()# qids are 0..n-1
    sheet = pd.concat([s[["qid_a", "qid_b"]].assign(method=m) for m, s in samples_by_method.items()],
                      ignore_index=True)
    sheet = sheet.sample(frac=1, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "id", np.arange(1, len(sheet) + 1))
    sheet["q1"] = text[sheet["qid_a"]]
    sheet["q2"] = text[sheet["qid_b"]]
    sheet["label"] = ""
    return sheet


def export_sheet(sheet, sheet_path, key_path):
    """Write the sheet to label and the key.
    Never overwrites an existing sheet, so labels already typed in are safe.
    """
    sheet_path, key_path = Path(sheet_path), Path(key_path)
    if sheet_path.exists():
        print(f"{sheet_path.name} already exists: not overwritten")
        return
    sheet[["id", "q1", "q2", "label"]].to_csv(sheet_path, index=False)
    sheet[["id", "method", "qid_a", "qid_b"]].to_csv(key_path, index=False)
    print(f"wrote {sheet_path.name} ({len(sheet)} pairs to label) and {key_path.name}")


def load_labels(sheet_path, key_path):
    """Labelled sheet joined with the key. None if the sheet is missing or has no labels yet."""
    sheet_path = Path(sheet_path)
    if not sheet_path.exists():
        return None
    # sep=None and utf-8-sig: also reads a sheet saved by Excel or Numbers (";" separator, BOM)
    sheet = pd.read_csv(sheet_path, dtype={"label": str}, keep_default_na=False,
                        sep=None, engine="python", encoding="utf-8-sig")
    sheet["label"] = sheet["label"].str.strip()
    if (sheet["label"] == "").all():
        return None
    invalid = ~sheet["label"].isin(VALID_LABELS + ("",))
    if invalid.any():
        raise ValueError(f"labels must be 1 or 0; check ids {sheet.loc[invalid, 'id'].tolist()}")
    return sheet.merge(pd.read_csv(key_path), on="id", validate="one_to_one")


def wilson_interval(successes, n, z=1.96):
    """Wilson score interval for a proportion (95% for z = 1.96). Better than p ± z·se for small n."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half_width, centre + half_width


def summarise_labels(labelled):
    """Per method: share of true duplicates among the labelled pairs, with its Wilson 95% CI."""
    rows = []
    for method, group in labelled.groupby("method"):
        n_labelled = int(group["label"].isin(VALID_LABELS).sum())
        n_duplicates = int((group["label"] == "1").sum())
        ci_low, ci_high = wilson_interval(n_duplicates, n_labelled)
        rows.append({"method": method,
                     "n_labelled": n_labelled,
                     "n_not_labelled": int((group["label"] == "").sum()),
                     "share_duplicates": n_duplicates / n_labelled if n_labelled else np.nan,
                     "ci95_low": ci_low,
                     "ci95_high": ci_high})
    return pd.DataFrame(rows).set_index("method")
