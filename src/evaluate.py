"""Evaluation (owner: P1).

TODO: evaluate(candidates, truth) with pair precision/recall/F1, cluster metrics, recall by Jaccard bin.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

def word_jaccard(text1, text2):
    """percentage of words the two questions have in common (punctuation ignored)"""
    words1 = set(re.findall(r"\w+", text1))
    words2 = set(re.findall(r"\w+", text2))

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
