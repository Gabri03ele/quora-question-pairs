"""MinHash signatures and LSH candidate generation."""

import re # Regular expressions

import numpy as np
import pandas as pd

_TOKEN_RE = re.compile(r"\w+")
_PRIME = 2**31 - 1
_EMPTY_SIGNATURE = np.iinfo(np.uint32).max
_PANDAS_HASH_KEY = "0123456789123456"


def tokenize(text):
    """Return lowercase word tokens (same definition as P1's Jaccard)."""
    return _TOKEN_RE.findall(str(text).lower())


def shingles(text, q=1):
    """Return the set-like list of distinct consecutive q-word shingles."""
    if q < 1:
        raise ValueError("q must be a positive integer")
    
    words = tokenize(text)
    if len(words) < q:
        return []
    if q == 1:
        return list(dict.fromkeys(words))
    
    return list(
        dict.fromkeys("\x1f".join(words[i : i + q]) for i in range(len(words) - q + 1))
    )


def _question_texts(questions):
    if isinstance(questions, pd.DataFrame):
        if "text_norm" not in questions:
            raise ValueError("questions DataFrame must have a 'text_norm' column")
        return questions["text_norm"].tolist()
    
    return list(questions)


def minhash_signatures(questions, q=1, num_hashes=64, seed=42, batch_size=512):
    """Build one deterministic MinHash signature per question. Each distinct q-word 
    shingle gets a stable 64-bit hash, then each signature position applies its own 
    affine hash modulo a prime. Batching keeps the temporary arrays small.
    """
    if not isinstance(q, (int, np.integer)) or q < 1:
        raise ValueError("q must be a positive integer")
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

        batch_shingles = [shingles(text, q=q) for text in texts[start:stop]]
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


def lsh_candidates(
    signatures,
    qids=None,
    bands=8,
    rows=8,
    source="lsh",
    max_band_pairs=20_000_000,
    score_batch_size=250_000,
):
    """Return each pair that shares at least one band. Pairs that collide in
    multiple bands are returned once, with ``score`` set to their fraction of
    matching signature values. Empty signatures never enter a bucket. If one
    band would exceed ``max_band_pairs``, this raises ``MemoryError`` instead
    of silently dropping candidates.
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
    pair_codes = np.empty(0, dtype=np.int64)
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
        
        order = np.argsort(inverse, kind="stable")
        offsets = np.concatenate(([0], np.cumsum(counts)))
        encoded_band_pairs = []
        for bucket_index, count in enumerate(counts):
            if count < 2:
                continue
            bucket = valid_positions[order[offsets[bucket_index] : offsets[bucket_index + 1]]]
            left, right = np.triu_indices(int(count), k=1)
            first = np.minimum(bucket[left], bucket[right]).astype(np.int64)
            second = np.maximum(bucket[left], bucket[right]).astype(np.int64)
            encoded_band_pairs.append(first * n_questions + second)
        if not encoded_band_pairs:
            continue
        
        band_codes = np.sort(np.concatenate(encoded_band_pairs))
        pair_codes = np.union1d(pair_codes, band_codes)

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
