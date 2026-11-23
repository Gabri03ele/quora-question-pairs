"""MinHash signatures, LSH candidate generation and the P2 validation studies."""

import re # Regular expressions
import time

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"\w+")
_PRIME = 2**31 - 1
_EMPTY_SIGNATURE = np.iinfo(np.uint32).max
_PANDAS_HASH_KEY = "0123456789123456"


def tokenize(text):
    """Return lowercase word tokens (same definition as P1's Jaccard)."""
    return _TOKEN_RE.findall(str(text).lower())


def shingles(text, q=1, stop_words=None):
    """Return the set-like list of distinct consecutive q-word shingles.
    Words in ``stop_words`` are removed before the shingles are formed.
    """
    if q < 1:
        raise ValueError("q must be a positive integer")
    
    words = tokenize(text)
    if stop_words is not None:
        words = [word for word in words if word not in stop_words]
    if len(words) < q:
        return []
    if q == 1:
        return list(dict.fromkeys(words))
    
    return list(
        dict.fromkeys("\x1f".join(words[i : i + q]) for i in range(len(words) - q + 1))
    )


def character_shingles(text, k=3):
    """Return distinct character k-shingles from normalised text."""
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    normalized = " ".join(str(text).lower().split())
    if len(normalized) < k:
        return []
    return list(dict.fromkeys(normalized[i:i + k] for i in range(len(normalized) - k + 1)))


def _question_texts(questions):
    if isinstance(questions, pd.DataFrame):
        if "text_norm" not in questions:
            raise ValueError("questions DataFrame must have a 'text_norm' column")
        return questions["text_norm"].tolist()
    
    return list(questions)


def minhash_signatures(
    questions,
    q=1,
    num_hashes=64,
    seed=42,
    batch_size=512,
    shingle_type="word",
    char_size=3,
    stop_words=None,
):
    """Build one deterministic MinHash signature per question. Shingles are q-word
    sequences, or character ``char_size``-grams if ``shingle_type="character"``
    (``stop_words`` only applies to words). Each distinct shingle gets a stable 64-bit hash, then each signature position applies its own 
    affine hash modulo a prime. Batching keeps the temporary arrays small.
    """
    if not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError("q must be a positive integer")
    if shingle_type not in {"word", "character"}:
        raise ValueError("shingle_type must be 'word' or 'character'")
    if not isinstance(num_hashes, (int, np.integer)) or num_hashes < 1:
        raise ValueError("num_hashes must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    texts = _question_texts(questions)
    n_questions = len(texts)

    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_hashes, dtype=np.int64)
    b = rng.integers(0, _PRIME, size=num_hashes, dtype=np.int64)
    
    signatures = np.full((n_questions, num_hashes), _EMPTY_SIGNATURE, dtype=np.uint32)

    for start in range(0, n_questions, batch_size):
        stop = min(start + batch_size, n_questions)

        if shingle_type == "word":
            batch_shingles = [shingles(text, q=q, stop_words=stop_words) for text in texts[start:stop]]
        else:
            batch_shingles = [character_shingles(text, k=char_size) for text in texts[start:stop]]
        max_width = max((len(row) for row in batch_shingles), default=0)
        if max_width == 0:
            continue

        flattened = [item for row in batch_shingles for item in row]
        hashed = pd.util.hash_pandas_object(
            pd.Series(flattened, dtype=object), 
            index=False, 
            hash_key=_PANDAS_HASH_KEY
        ).to_numpy(dtype=np.uint64)

        base = np.zeros((stop - start, max_width), dtype=np.int64)
        valid = np.zeros((stop - start, max_width), dtype=bool)
        cursor = 0
        for row_index, row in enumerate(batch_shingles):
            width = len(row)
            if width:
                row_hashes = hashed[cursor : cursor + width] % np.uint64(_PRIME)
                base[row_index, :width] = row_hashes.astype(np.int64)
                valid[row_index, :width] = True
                cursor += width

        # Chunking the hash functions bounds the temporary array to roughly
        # batch_size x max_shingles_in_batch x 16 int64 values.
        for hash_start in range(0, num_hashes, 16):
            hash_stop = min(hash_start + 16, num_hashes)

            values = (
                base[:, :, None] * a[None, None, hash_start:hash_stop]
                + b[None, None, hash_start:hash_stop]
            ) % _PRIME

            values[~valid, :] = _PRIME
            minimum = values.min(axis=1)
            minimum[~valid.any(axis=1)] = _EMPTY_SIGNATURE
            signatures[start:stop, hash_start:hash_stop] = minimum.astype(np.uint32)

    return signatures


def banding_candidate_probability(similarity, bands, rows):
    """Theoretical probability that a pair becomes an LSH candidate."""
    if bands < 1 or rows < 1:
        raise ValueError("bands and rows must be positive integers")
    similarity = np.asarray(similarity, dtype=float)
    return 1.0 - (1.0 - similarity**rows) ** bands


def _jaccard(text1, text2):
    """Exact word Jaccard similarity of two texts."""
    words1, words2 = set(tokenize(text1)), set(tokenize(text2))
    union = words1 | words2
    return len(words1 & words2) / len(union) if union else 0.0


def labelled_pair_table(pairs, questions, split="dev"):
    """Labelled pairs of one split with exact word Jaccard and question row positions.

    The positions index the rows of a signature matrix built from ``questions``.
    """
    labelled = pairs.loc[pairs["split"] == split, ["qid1", "qid2", "is_duplicate"]].copy()
    position = pd.Series(np.arange(len(questions)), index=questions["qid"].to_numpy())
    labelled["pos1"] = labelled["qid1"].map(position).to_numpy()
    labelled["pos2"] = labelled["qid2"].map(position).to_numpy()
    texts = questions["text_norm"].to_numpy()
    labelled["jaccard"] = [
        _jaccard(texts[a], texts[b]) for a, b in zip(labelled["pos1"], labelled["pos2"])
    ]
    return labelled.reset_index(drop=True)


def band_collisions(signatures, pos1, pos2, bands, rows):
    """True for each given pair whose signatures agree on at least one whole band.

    This is exactly the LSH candidate rule, checked on given pairs only, so it
    gives the labelled metrics of a setting without generating its candidates.
    """
    if bands * rows > signatures.shape[1]:
        raise ValueError("bands * rows exceeds the signature length")
    pos1, pos2 = np.asarray(pos1), np.asarray(pos2)
    empty = signatures[:, 0] == _EMPTY_SIGNATURE
    found = np.zeros(len(pos1), dtype=bool)
    for band in range(bands):
        columns = slice(band * rows, (band + 1) * rows)
        found |= (signatures[pos1, columns] == signatures[pos2, columns]).all(axis=1)
    return found & ~empty[pos1] & ~empty[pos2]


def labelled_scores(found, labelled):
    """Recall, precision and F1 of a boolean 'found' mask over labelled pairs."""
    found = np.asarray(found)
    duplicate = labelled["is_duplicate"].to_numpy() == 1
    hits = int((found & duplicate).sum())
    recall = hits / duplicate.sum()
    precision = hits / found.sum() if found.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def validate_against_datasketch(labelled, questions, signatures, n_per_bin=200,
                                bands=None, rows=None, seed=42, n_bins=5):
    """Compare our MinHash and LSH with datasketch on a Jaccard-stratified sample.

    Random question pairs almost always have Jaccard near 0, so we draw
    ``n_per_bin`` labelled pairs from each of ``n_bins`` equal-width Jaccard bins.
    Both methods use ``signatures.shape[1]`` hashes. With ``bands`` and ``rows``
    the result also says whether each method makes the pair a candidate;
    datasketch's answer comes from its own ``MinHashLSH`` index.
    """
    from datasketch import MinHash, MinHashLSH

    num_hashes = signatures.shape[1]
    bin_index = np.minimum((labelled["jaccard"] * n_bins).astype(int), n_bins - 1)
    sample = pd.concat([
        group.sample(min(n_per_bin, len(group)), random_state=seed)
        for _, group in labelled.groupby(bin_index)
    ]).reset_index(drop=True)

    texts = questions["text_norm"].to_numpy()
    reference = {}
    for position in set(sample["pos1"]) | set(sample["pos2"]):
        sketch = MinHash(num_perm=num_hashes, seed=seed)
        for word in shingles(texts[position]):
            sketch.update(word.encode("utf-8"))
        reference[position] = sketch

    result = sample[["qid1", "qid2", "jaccard"]].copy()
    result["own_estimate"] = (
        signatures[sample["pos1"]] == signatures[sample["pos2"]]
    ).mean(axis=1)
    result["datasketch_estimate"] = [
        float(reference[a].jaccard(reference[b]))
        for a, b in zip(sample["pos1"], sample["pos2"])
    ]
    if bands is not None:
        index = MinHashLSH(num_perm=num_hashes, params=(bands, rows))
        for position, sketch in reference.items():
            index.insert(position, sketch)
        result["own_candidate"] = band_collisions(
            signatures, sample["pos1"], sample["pos2"], bands, rows
        )
        result["datasketch_candidate"] = [
            b in index.query(reference[a]) for a, b in zip(sample["pos1"], sample["pos2"])
        ]
    return result


def empirical_s_curve(labelled, signatures, bands, rows, n_bins=20):
    """Share of labelled pairs that become candidates, per exact-Jaccard bin.

    The theoretical value is the mean of ``1-(1-s^r)^b`` over the pairs in the
    bin. Using the bin midpoint instead would be biased because the curve is steep.
    """
    similarity = labelled["jaccard"].to_numpy()
    table = pd.DataFrame({
        "bin": np.minimum((similarity * n_bins).astype(int), n_bins - 1),
        "jaccard": similarity,
        "empirical": band_collisions(
            signatures, labelled["pos1"], labelled["pos2"], bands, rows
        ),
        "theoretical": banding_candidate_probability(similarity, bands, rows),
    })
    result = table.groupby("bin").agg(
        n_pairs=("jaccard", "size"),
        mean_jaccard=("jaccard", "mean"),
        empirical=("empirical", "mean"),
        theoretical=("theoretical", "mean"),
    )
    result.insert(0, "bin_left", result.index / n_bins)
    return result.reset_index(drop=True)


def short_question_mask(questions, min_words=4, min_chars=20):
    """True for questions with fewer than min_words words or min_chars characters."""
    texts = questions["text_norm"].fillna("")
    n_words = texts.map(lambda text: len(tokenize(text))).to_numpy()
    n_chars = texts.map(len).to_numpy()
    return (n_words < min_words) | (n_chars < min_chars)


def short_question_report(questions, pairs, split="dev", min_words=4, min_chars=20):
    """Count short questions and the labelled pairs of one split that contain one."""
    short = short_question_mask(questions, min_words, min_chars)
    short_ids = questions.loc[short, "qid"]
    split_pairs = pairs[pairs["split"] == split]
    affected = split_pairs[
        split_pairs["qid1"].isin(short_ids) | split_pairs["qid2"].isin(short_ids)
    ]
    return {
        "n_questions": len(questions),
        "n_short": int(short.sum()),
        "fraction_short": float(short.mean()),
        "n_affected_pairs": len(affected),
        "affected_duplicate_pairs": int(affected["is_duplicate"].sum()),
    }


def all_pairs_cost(questions, n_queries=1000, seed=42):
    """Extrapolated seconds for the exact word Jaccard of all pairs of questions.

    Compares a random sample of questions with all questions by a sparse matrix
    product (intersection sizes), then scales to the n(n-1)/2 distinct pairs.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        tokenizer=tokenize, token_pattern=None, lowercase=False, binary=True, dtype=np.int32
    )
    matrix = vectorizer.fit_transform(questions["text_norm"])
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    n = len(questions)
    queries = np.random.default_rng(seed).choice(n, size=min(n_queries, n), replace=False)
    transposed = matrix.T.tocsr()

    started = time.perf_counter()
    for start in range(0, len(queries), 100):
        batch = queries[start : start + 100]
        shared = (matrix[batch] @ transposed).tocoo()
        shared.data / (sizes[batch][shared.row] + sizes[shared.col] - shared.data)
    seconds = time.perf_counter() - started
    return {"sample_seconds": seconds, "sample_queries": len(queries),
            "estimated_seconds": seconds / len(queries) * (n - 1) / 2}


def bucket_pair_counts(signatures, bands, rows):
    """Number of bucket pairs in each band, without listing them.

    A pair that collides in several bands is counted several times, so the sum is
    an upper bound on the candidates and the maximum a lower bound.
    """
    valid = signatures[:, 0] != _EMPTY_SIGNATURE
    totals = []
    for band in range(bands):
        keys = signatures[valid, band * rows : (band + 1) * rows]
        counts = np.unique(keys, axis=0, return_counts=True)[1].astype(np.int64)
        totals.append(int((counts * (counts - 1) // 2).sum()))
    return np.array(totals)


def lsh_candidates(
    signatures,
    qids=None,
    bands=8,
    rows=8,
    source="lsh",
    max_band_pairs=20_000_000,
    score_batch_size=250_000,
    must_include=None,
):
    """Return each pair that shares at least one band. Pairs that collide in
    multiple bands are returned once, with ``score`` set to their fraction of
    matching signature values. Empty signatures never enter a bucket. If one
    band would exceed ``max_band_pairs``, this raises ``MemoryError`` instead
    of silently dropping candidates. If the boolean mask ``must_include`` is
    given, only pairs with at least one marked question are returned.
    """
    signatures = np.asarray(signatures)
    if signatures.ndim != 2:
        raise ValueError("signatures must be a two-dimensional array")
    if not isinstance(bands, (int, np.integer)) or bands < 1:
        raise ValueError("bands must be a positive integer")
    if not isinstance(rows, (int, np.integer)) or rows < 1:
        raise ValueError("rows must be a positive integer")
    if max_band_pairs is not None and max_band_pairs < 1:
        raise ValueError("max_band_pairs must be positive or None")
    if not isinstance(score_batch_size, (int, np.integer)) or score_batch_size < 1:
        raise ValueError("score_batch_size must be a positive integer")
    if bands * rows != signatures.shape[1]:
        raise ValueError("bands * rows must equal the signature length")

    n_questions = signatures.shape[0]
    if qids is None:
        qids = np.arange(n_questions, dtype=np.int64)
    else:
        qids = np.asarray(qids)
        if qids.ndim != 1 or len(qids) != n_questions:
            raise ValueError("qids must have one value per signature row")
        if len(np.unique(qids)) != n_questions:
            raise ValueError("qids must be unique")

    # Encode each positional pair as one int64. qids themselves can be
    # non-contiguous, so signatures are paired by row position then mapped back.
    encoded_pairs = []
    sentinel = _EMPTY_SIGNATURE

    for band in range(bands):
        start = band * rows
        band_signatures = signatures[:, start : start + rows]
        nonempty = np.any(band_signatures != sentinel, axis=1)
        valid_positions = np.flatnonzero(nonempty)
        if len(valid_positions) < 2:
            continue

        _, inverse, counts = np.unique(
            band_signatures[valid_positions],
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        band_pair_count = int(np.sum(counts.astype(np.int64) * (counts - 1) // 2))
        if max_band_pairs is not None and band_pair_count > max_band_pairs:
            raise MemoryError(
                f"band {band} contains {band_pair_count:,} bucket pairs, above the "
                f"max_band_pairs limit of {max_band_pairs:,}; use more rows per band"
            )
        
        # Positions sorted by bucket. Buckets of equal size are stacked into one
        # matrix, so all their pairs are produced without a Python loop per bucket.
        order = valid_positions[np.argsort(inverse, kind="stable")]
        offsets = np.concatenate(([0], np.cumsum(counts)))[:-1]
        for size in np.unique(counts[counts > 1]):
            buckets = order[offsets[counts == size][:, None] + np.arange(size)]
            if must_include is not None:
                buckets = buckets[must_include[buckets].any(axis=1)]
            left, right = np.triu_indices(int(size), k=1)
            first = buckets[:, left].ravel()
            second = buckets[:, right].ravel()
            if must_include is not None:
                keep = must_include[first] | must_include[second]
                first, second = first[keep], second[keep]
            # Positions inside a bucket are increasing, so first < second.
            encoded_pairs.append(first.astype(np.int64) * n_questions + second)

    # A pair that collides in several bands is kept once.
    pair_codes = np.concatenate(encoded_pairs) if encoded_pairs else np.empty(0, dtype=np.int64)
    del encoded_pairs
    pair_codes.sort()
    if len(pair_codes):
        pair_codes = pair_codes[np.concatenate(([True], pair_codes[1:] != pair_codes[:-1]))]

    columns = ["qid_a", "qid_b", "score", "source"]
    if len(pair_codes) == 0:
        return pd.DataFrame(
            {
                "qid_a": pd.Series(dtype=qids.dtype),
                "qid_b": pd.Series(dtype=qids.dtype),
                "score": pd.Series(dtype=float),
                "source": pd.Series(dtype=object),
            }
        )[columns]

    first = pair_codes // n_questions
    second = pair_codes % n_questions
    scores = np.empty(len(pair_codes), dtype=np.float32)
    for start in range(0, len(pair_codes), score_batch_size):
        stop = min(start + score_batch_size, len(pair_codes))
        scores[start:stop] = np.mean(
            signatures[first[start:stop]] == signatures[second[start:stop]], axis=1
        )
    result = pd.DataFrame(
        {
            "qid_a": np.minimum(qids[first], qids[second]),
            "qid_b": np.maximum(qids[first], qids[second]),
            "score": scores,
            "source": pd.Categorical.from_codes(
                np.zeros(len(pair_codes), dtype=np.int8), categories=[source]
            ),
        },
        columns=columns,
    )
    result.attrs["sorted_unique_pairs"] = True
    return result
