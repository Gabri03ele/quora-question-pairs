"""Evaluation (owner: P1).

evaluate(candidates, truth) with pair precision/recall/F1, cluster metrics, recall by Jaccard bin.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

def word_jaccard(text1, text2):
    """percentage of words the two questions have in common (punctuation ignored)"""
    words1 = set(re.findall(r"\w", text1))
    words2 = set(re.findall(r"\w", text2))

    if len(words1 | words2) == 0:
        return 0.0
    
    return len(words1 & words2) / len(words1 | words2)


def recall_by_jaccard_bin(candidates, pairs, questions, split="dev", bins=(0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Recall on the labelled duplicate pairs, separately for pairs that share few or many words."""
    candidates = sort_candidates(candidates)
    text_of_question = questions.set_index("qid")["text_norm"]

    true_pairs = pairs[(pairs["split"] == split) & (pairs["is_duplicate"] == 1)].copy()
    text1 = true_pairs["qid1"].map(text_of_question)
    text2 = true_pairs["qid2"].map(text_of_question)
    true_pairs["jaccard"] = [word_jaccard(t1, t2) for t1, t2 in zip(text1, text2)]
    true_pairs["bin"] = pd.cut(true_pairs["jaccard"], bins=list(bins), include_lowest=True)

    matched = true_pairs.merge(candidates, on=["qid1", "qid2"], how="left", indicator=True)
    matched["found"] = matched["_merge"] == "both"        # True if the method returned this pair

    result = matched.groupby("bin", observed=False).agg(n_pairs=("found", "size"), recall=("found", "mean"))
    return result

def sort_candidates(candidates):
    """Smaller qid first, self-pairs and repeated pairs removed"""
    a = candidates["qid_a"].to_numpy()
    b = candidates["qid_b"].to_numpy()

    qid1 = np.minimum(a, b)
    qid2 = np.maximum(a, b)

    keep = qid1 != qid2 #no self-pair

    sorted_candidates = pd.DataFrame({"qid1": qid1[keep], "qid2": qid2[keep]})

    return sorted_candidates.drop_duplicates(ignore_index=True)

def labelled_metrics(candidates, pairs, split):
    """Recall, precision and F1 on the labelled pairs of one split."""
    split_pairs = pairs[pairs["split"] == split]
    labelled = candidates.merge(split_pairs, on=["qid1", "qid2"], how="inner")   # candidates that have a label

    true_positives = labelled["is_duplicate"].sum()
    all_duplicates = split_pairs["is_duplicate"].sum()

    recall = true_positives / all_duplicates
    precision = true_positives / len(labelled) if len(labelled) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"labelled_candidates": len(labelled), "recall": recall, "precision": precision, "f1": f1}

def implied_recall(candidates, questions, split):
    """Recall on all pairs implied by the duplicate groups, labelled or not."""
    group_of_question = questions.set_index("qid")["group_id"]
    split_of_question = questions.set_index("qid")["split"]

    same_group = candidates["qid1"].map(group_of_question) == candidates["qid2"].map(group_of_question)
    in_split = candidates["qid1"].map(split_of_question) == split # same group means same split
    found_implied = (same_group & in_split).sum()

    sizes = questions[questions["split"] == split]["group_id"].value_counts()
    all_implied = (sizes * (sizes - 1) // 2).sum()

    return {"recall_implied": found_implied / all_implied}

def cost(candidates, questions):
    """Number of candidates, and their share of all possible pairs."""
    n = len(questions)
    all_pairs = n * (n - 1) // 2

    return {"n_candidates": len(candidates), "fraction_of_all_pairs": len(candidates) / all_pairs}

def evaluate(candidates, pairs, questions, split="dev"):
    """All metrics in one dictionary"""
    candidates = sort_candidates(candidates)
    results = {}
    results.update(cost(candidates, questions))
    results.update(labelled_metrics(candidates, pairs, split))
    results.update(implied_recall(candidates, questions, split))

    return results

def cluster_labels_for_questions(cluster_of_question, questions):
    """One integer cluster label per question, in the order of questions["qid"].
    Questions missing from cluster_of_question become clusters of size one."""
    labels = cluster_of_question.reindex(questions["qid"])
    codes, _ = pd.factorize(labels)              # missing labels get code -1
    is_missing = codes == -1
    codes[is_missing] = codes.max() + 1 + np.arange(is_missing.sum())   # a new cluster for each
    return pd.Series(codes, index=questions["qid"].to_numpy())


def pairs_inside(sizes):
    """Number of pairs inside groups of the given sizes: s(s-1)/2 each."""
    sizes = np.asarray(sizes, dtype=np.int64)
    return int((sizes * (sizes - 1) // 2).sum())


def evaluate_clusters(cluster_of_question, pairs, questions, split="dev", large_cluster_size=50):
    """Same metrics as evaluate, for a clustering, without listing the pairs inside each cluster.

    cluster_of_question: pd.Series indexed by qid, one cluster label per question.
    Two questions count as a candidate pair when they are in the same cluster.
    """
    cluster = cluster_labels_for_questions(cluster_of_question, questions)
    cluster_sizes = np.bincount(cluster.to_numpy())
    results = {}

    # cost: every two questions in the same cluster are one candidate pair
    n = len(questions)
    n_candidates = pairs_inside(cluster_sizes)
    results["n_candidates"] = n_candidates
    results["fraction_of_all_pairs"] = n_candidates / (n * (n - 1) // 2)

    # labelled pairs: a pair is a candidate when both questions have the same cluster
    split_pairs = pairs[pairs["split"] == split]
    same_cluster = split_pairs["qid1"].map(cluster).to_numpy() == split_pairs["qid2"].map(cluster).to_numpy()
    is_duplicate = split_pairs["is_duplicate"].to_numpy() == 1

    true_positives = int((same_cluster & is_duplicate).sum())
    labelled_candidates = int(same_cluster.sum())
    recall = true_positives / is_duplicate.sum()
    precision = true_positives / labelled_candidates if labelled_candidates > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    results.update({"labelled_candidates": labelled_candidates, "recall": recall,
                    "precision": precision, "f1": f1})

    # implied recall: pairs in the same true group AND the same cluster.
    # Count the questions in each (group, cluster) cell; a cell with c questions holds c(c-1)/2 such pairs.
    in_split = questions["split"].to_numpy() == split
    group_and_cluster = pd.DataFrame({"group_id": questions["group_id"].to_numpy()[in_split],
                                      "cluster": cluster.to_numpy()[in_split]})
    found_implied = pairs_inside(group_and_cluster.value_counts())
    all_implied = pairs_inside(group_and_cluster["group_id"].value_counts())
    results["recall_implied"] = found_implied / all_implied

    # chaining: very large clusters usually mean unrelated groups were merged
    is_large = cluster_sizes >= large_cluster_size
    results["largest_cluster"] = int(cluster_sizes.max())
    results["n_large_clusters"] = int(is_large.sum())
    results["fraction_in_large_clusters"] = cluster_sizes[is_large].sum() / n

    return results