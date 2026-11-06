"""Evaluation (owner: P1).

TODO: evaluate(candidates, truth) with pair precision/recall/F1, cluster metrics, recall by Jaccard bin.
"""

from pathlib import Path
import pandas as pd
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
    """smaller qid first"""
    sorted_candidates = pd.DataFrame({
        "qid1": candidates[["qid_a", "qid_b"]].min(axis=1),
        "qid2": candidates[["qid_a", "qid_b"]].max(axis=1),
    })

    sorted_candidates = sorted_candidates[sorted_candidates["qid1"] != sorted_candidates["qid2"]]
    sorted_candidates = sorted_candidates.drop_duplicates().reset_index(drop=True)

    return sorted_candidates

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


def make_toy_data():
    """A tiny dataset where every result can be worked out by hand."""
    questions = pd.DataFrame({
        "qid": [0, 1, 2, 3, 4, 5, 6],
        "text_norm": [
            "how do i learn python",
            "how can i learn python",
            "what is the best way to start with python?",
            "how do i cook rice",
            "what is the capital of france?",
            "how do i lose weight",
            "how can i lose weight fast",
        ],
        "group_id": [5, 5, 5, 8, 9, 10, 10],     # 0, 1, 2 are duplicates; 5 and 6 are duplicates
        "split": ["dev", "dev", "dev", "dev", "dev", "test", "test"],
    })
    pairs = pd.DataFrame({
        "qid1": [0, 1, 0, 5],
        "qid2": [1, 2, 3, 6],
        "conflicting_pair": [False, False, False, False],
        "split": ["dev", "dev", "dev", "test"],
        "is_duplicate": [1, 1, 0, 1],
    })
    candidates = pd.DataFrame({
        "qid_a": [1, 0, 0, 0, 3, 2],
        "qid_b": [0, 1, 2, 3, 3, 4],
        # (1,0) and (0,1) are the same pair; (3,3) is a self-pair; (0,2) is a duplicate without a label
    })
    return questions, pairs, candidates
 
 
def test_on_toy_data():
    questions, pairs, candidates = make_toy_data()
 
    sorted_candidates = sort_candidates(candidates)
    print("Sorted candidates:")
    print(sorted_candidates)
    assert sorted_candidates.values.tolist() == [[0, 1], [0, 2], [0, 3], [2, 4]]
 
    results = evaluate(candidates, pairs, questions, split="dev")
    print(results)
    # 4 candidates out of 7 * 6 / 2 = 21 pairs
    assert results["n_candidates"] == 4
    assert abs(results["fraction_of_all_pairs"] - 4 / 21) < 1e-9
    # labelled candidates: (0,1) is 1, (0,3) is 0. Dev has 2 duplicates: (0,1), (1,2)
    assert results["labelled_candidates"] == 2
    assert results["recall"] == 0.5
    assert results["precision"] == 0.5
    assert results["f1"] == 0.5
    # group 5 has 3 pairs; found (0,1) and (0,2)
    assert abs(results["recall_implied"] - 2 / 3) < 1e-9
 
    bins = recall_by_jaccard_bin(candidates, pairs, questions, split="dev")
    print(bins)
    # (0,1): jaccard 4/6 = 0.67, found. (1,2): jaccard 1/13 = 0.08, not found
    assert abs(word_jaccard(questions.text_norm[0], questions.text_norm[1]) - 4 / 6) < 1e-9
    assert bins["n_pairs"].tolist() == [1, 0, 0, 1, 0]
    assert bins["recall"].iloc[0] == 0.0
    assert bins["recall"].iloc[3] == 1.0
 
    print("Toy data: all checks passed\n")
 
 
def test_on_real_data(data_dir):
    questions = pd.read_parquet(data_dir / "questions_sample10.parquet")
    pairs = pd.read_parquet(data_dir / "pairs_sample10.parquet")
 
    # 1. the true duplicate pairs as candidates: a perfect method
    perfect = pairs[(pairs["split"] == "dev") & (pairs["is_duplicate"] == 1)]
    perfect = perfect.rename(columns={"qid1": "qid_a", "qid2": "qid_b"})
    results = evaluate(perfect, pairs, questions, split="dev")
    print("Perfect candidates:", results)
    assert results["recall"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall_implied"] <= 1.0
 
    # 2. the same pairs in reversed order must give the same result
    reversed_pairs = perfect.rename(columns={"qid_a": "qid_b", "qid_b": "qid_a"})
    assert evaluate(reversed_pairs, pairs, questions, split="dev") == results
 
    # 3. no candidates: everything is 0
    empty = pd.DataFrame({"qid_a": pd.Series(dtype="int64"), "qid_b": pd.Series(dtype="int64")})
    results = evaluate(empty, pairs, questions, split="dev")
    print("No candidates:", results)
    assert results["recall"] == 0 and results["precision"] == 0 and results["recall_implied"] == 0
 
    # 4. recall by bin of the perfect method is 1 in every non-empty bin
    bins = recall_by_jaccard_bin(perfect, pairs, questions, split="dev")
    print(bins)
    assert (bins.loc[bins["n_pairs"] > 0, "recall"] == 1.0).all()
 
    print("Real data (sample): all checks passed")
 
 
if __name__ == "__main__":
    test_on_toy_data()
 
    data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"
    if (data_dir / "questions_sample10.parquet").exists():
        test_on_real_data(data_dir)
    else:
        print(f"Skipped the real-data checks: no sample files in {data_dir}")