"""Data loading and cleaning (owner: P1).

Functions used by notebooks/01_data.ipynb to build questions.parquet and pairs.parquet
from the raw Kaggle file. The notebook shows the outputs and explains each step.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def stack_questions(df_raw):
    q1 = df_raw[ ["qid1", "question1"] ]
    q2 = df_raw[ ["qid2", "question2"] ]

    q1 = q1.rename(columns={"qid1": "qid", "question1": "question"})
    q2 = q2.rename(columns={"qid2": "qid", "question2": "question"})

    q1_with_q2 = pd.concat([q1, q2]).dropna()
    return q1_with_q2


def report_id_problems(df_raw):
    """Print why id, qid1 and qid2 are not useful"""
    q1_with_q2 = stack_questions(df_raw)
    ids_per_text = q1_with_q2.groupby("question")["qid"].nunique()

    print(f"Distinct qids: {q1_with_q2['qid'].nunique()}")
    print(f"Distinct question: {q1_with_q2['question'].nunique()}")
    
    print("\nQuestions with most different ids:")
    print(ids_per_text.sort_values(ascending=False).head())


def drop_unused_columns(df_raw):
    """Drop columns id, qid1 and qid2"""
    return df_raw.drop(columns=["id", "qid1", "qid2"])


def drop_missing(df):
    """Drop rows with a missing value"""
    df = df.copy()  
    df_null = df[df.isna().any(axis=1)]

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, df_null


def normalise_questions(df):
    df = df.copy()
    df["question1_norm"] = df["question1"]
    df["question2_norm"] = df["question2"]

    for col in ["question1_norm", "question2_norm"]:
        df[col] = df[col].str.normalize("NFKC")
        df[col] = df[col].str.lower()
        df[col] = df[col].str.split().str.join(" ")

    target = df.pop("is_duplicate")
    df["is_duplicate"] = target
    return df


def build_questions_table(df):
    """One row per distinct normalised question"""
    q1_norm = df["question1_norm"].rename("question")
    q2_norm = df["question2_norm"].rename("question")

    df_unique_questions = pd.concat([q1_norm, q2_norm], ignore_index=True).drop_duplicates().reset_index(drop=True)
    return df_unique_questions


def build_pairs_table(df, df_unique_questions):
    """Replaces each question text by its qid."""
    unique_ids = df_unique_questions.to_dict()
    unique_ids = dict((v, k) for k, v in unique_ids.items())

    df_pair_questions = pd.DataFrame({"qid1": df["question1_norm"].map(unique_ids),
                                      "qid2": df["question2_norm"].map(unique_ids),
                                      "is_duplicate": df["is_duplicate"]})
    
    return df_pair_questions, unique_ids


def check_pairs(df_pair_questions):
    """Detect identical, repeated and inconsistent pairs"""
    df_questions = df_pair_questions.copy()

    # identical, both sides are the same question
    identical_pairs = df_questions["qid1"] == df_questions["qid2"]
    print(f"Number of Identical Pairs: {identical_pairs.sum()}")

    # repeated, same two questions in more than one row, in either order
    sorted_pairs = df_questions[["qid1", "qid2"]].apply(lambda row: tuple(sorted(row)), axis=1)
    sorted_pairs = sorted_pairs.rename("sorted_pairs")

    repeated_pairs = sorted_pairs.duplicated()
    print(f"Number of Repeated Pairs: {repeated_pairs.sum()}")

    # inconsistent, the same pair appears with different labels
    sorted_pairs = sorted_pairs.to_frame()
    sorted_pairs["is_duplicate"] = df_questions["is_duplicate"]

    inconsistent_pairs = sorted_pairs.groupby("sorted_pairs")["is_duplicate"].transform("nunique") > 1
    print(f"Number of Inconsistent Pairs: {inconsistent_pairs.sum()}")

    return identical_pairs, repeated_pairs, inconsistent_pairs, sorted_pairs["sorted_pairs"]


def clean_pairs(df_pair_questions, identical_pairs, repeated_pairs, inconsistent_pairs, sorted_pairs):
    """Drops identical, repeated and inconsistent pairs"""
    remove_rows = identical_pairs | repeated_pairs | inconsistent_pairs
    df_clean = df_pair_questions[~remove_rows].copy()

    df_clean[["qid1", "qid2"]] = sorted_pairs.loc[~remove_rows].tolist()
    df_clean = df_clean.reset_index(drop=True)

    print(f"Rows removed: {remove_rows.sum()}, rows kept: {len(df_clean)}")
    return df_clean


def build_groups(df_clean, df_unique_questions):
    """Connected components of the duplicate pairs are ground-truth groups"""
    all_duplicates = df_clean[df_clean["is_duplicate"] == 1]
    n = len(df_unique_questions)

    graph = coo_matrix((np.ones(len(all_duplicates)), (all_duplicates["qid1"], all_duplicates["qid2"])), shape=(n, n))
    number_of_groups, group_id = connected_components(graph, directed=False)
    sizes = pd.Series(group_id).value_counts()

    print(f"Number of Groups: {number_of_groups} \nGroups with Size greater than 1: {(sizes > 1).sum()}")
    print(f"Largest group: {sizes.max()} questions")
    print(f"Labelled duplicate pairs: {len(all_duplicates)}")
    print(f"Duplicate pairs implied by the groups: {(sizes * (sizes - 1) // 2).sum()}")

    questions = df_unique_questions.to_frame(name="text_norm")
    questions.index.name = "qid"
    questions = questions.reset_index()
    questions["group_id"] = group_id

    df_clean = df_clean.copy()
    
    df_clean["conflicting_pair"] = (
        (df_clean["is_duplicate"] == 0)
        & (group_id[df_clean["qid1"]] == group_id[df_clean["qid2"]])
    )
    print(f"Number of Conflicting Pairs: {df_clean['conflicting_pair'].sum()}")
    return questions, df_clean, sizes


def split_dev_test(df_clean, questions, test_fraction=0.2, seed=42):
    """Splits whole connected components of the labelled pairs into dev and test"""
    n = len(questions)

    all_pairs_graph = coo_matrix((np.ones(len(df_clean)), (df_clean["qid1"], df_clean["qid2"])), shape=(n, n))
    number_of_components, component_id = connected_components(all_pairs_graph, directed=False)
    component_size = np.bincount(component_id) # questions in each component

    rng = np.random.default_rng(seed)
    test_components = []
    n_test = 0
    for component in rng.permutation(number_of_components):
        if n_test >= test_fraction * n:
            break
        test_components.append(component) # this component, with ALL its pairs, goes to test
        n_test += component_size[component]

    questions = questions.copy()
    df_clean = df_clean.copy()
    questions["component_id"] = component_id
    questions["split"] = "dev"  # everybody starts in dev
    questions.loc[questions["component_id"].isin(test_components), "split"] = "test"

    split_of_question = questions.set_index("qid")["split"] 
    df_clean["split"] = df_clean["qid1"].map(split_of_question) 

    target = df_clean.pop("is_duplicate")
    df_clean["is_duplicate"] = target

    print(f"Components: {number_of_components}")
    print(df_clean["split"].value_counts())
    return questions, df_clean


def save_tables(questions, df_clean, out_dir):
    """Saves tables in files"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions_out = questions[["qid", "text_norm", "group_id", "split"]]
    pairs_out = df_clean[["qid1", "qid2", "conflicting_pair", "split", "is_duplicate"]]

    questions_out.to_parquet(out_dir / "questions.parquet", index=False)
    pairs_out.to_parquet(out_dir / "pairs.parquet", index=False)

    q_back = pd.read_parquet(out_dir / "questions.parquet")
    p_back = pd.read_parquet(out_dir / "pairs.parquet")
    print(f"questions: {q_back.shape}, pairs: {p_back.shape}")
    # dtypes may differ (pandas 3)
    pd.testing.assert_frame_equal(q_back, questions_out, check_dtype=False)
    pd.testing.assert_frame_equal(p_back, pairs_out, check_dtype=False)


def make_sample(questions, df_clean, fraction=0.1, seed=42):
    rng = np.random.default_rng(seed)
    sample_components = []

    for split in ["dev", "test"]:
        split_questions = questions[questions["split"] == split]
        component_size = split_questions["component_id"].value_counts()   # questions in each component
        n_target = fraction * len(split_questions)
        n_taken = 0
        for component in rng.permutation(component_size.index):          # components in random order
            if n_taken >= n_target:                                       # this side is full: stop
                break
            if n_taken + component_size[component] > 1.05 * n_target:     # too big to fit: skip it
                continue
            sample_components.append(component)
            n_taken += component_size[component]

    questions_sample = questions[questions["component_id"].isin(sample_components)]
    pairs_sample = df_clean[df_clean["qid1"].isin(questions_sample["qid"])]

    print(f"Questions: {len(questions_sample)} ({len(questions_sample) / len(questions):.1%})")
    print(f"Pairs: {len(pairs_sample)} ({len(pairs_sample) / len(df_clean):.1%})")
    print(questions_sample["split"].value_counts(normalize=True))
    print(f"Duplicate rate: {pairs_sample['is_duplicate'].mean():.2f}")
    return questions_sample, pairs_sample
