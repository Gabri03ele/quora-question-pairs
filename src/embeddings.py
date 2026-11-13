"""Sentence embeddings and HNSW search (owner: P3).

TODO: encode questions, cache embeddings to disk, FAISS HNSW index, neighbour search, check against exact search.
"""
# -*- coding: utf-8 -*-


import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
import numpy as np
import faiss
from torch.utils.data import DataLoader
import time
from scipy.spatial.distance import cdist

df = pd.read_parquet('/content/quora-question-pairs/data/processed/questions.parquet')
texts = df['text_norm'].fillna("").tolist()
qids = df['qid'].tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

np.save('embeddings.npy', embeddings)

d = embeddings.shape[1]
index_hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index_hnsw.add(embeddings)

k = 20
threshold = 0.80

scores, indices = index_hnsw.search(embeddings, k)

candidates = []
for i in range(len(qids)):
    for j in range(1, k):
        score = float(scores[i][j])
        if score >= threshold:
            candidates.append({
                'qid_a': qids[i],
                'qid_b': qids[indices[i][j]],
                'score': score,
                'source': 'emb'
            })

df_candidates = pd.DataFrame(candidates)
df_candidates.to_parquet('candidates_emb.parquet', index=False)

sample_size = min(1000, len(embeddings))
sample_queries = embeddings[:sample_size]

index_exact = faiss.IndexFlatIP(d)
index_exact.add(embeddings)

exact_scores, exact_indices = index_exact.search(sample_queries, k)
hnsw_scores, hnsw_indices = index_hnsw.search(sample_queries, k)

hits = 0
for i in range(sample_size):
    exact_set = set(exact_indices[i])
    hnsw_set = set(hnsw_indices[i])
    hits += len(exact_set.intersection(hnsw_set))

recall = hits / (sample_size * k)
print(f"HNSW Recall: {recall:.4f}")

"""## experimens

"""

df_questions = pd.read_parquet('/content/quora-question-pairs/data/processed/questions.parquet')
df_pairs = pd.read_parquet('/content/quora-question-pairs/data/processed/pairs.parquet')

df_train = df_pairs[df_pairs['split'] == 'dev']

qid_to_text = dict(zip(df_questions['qid'], df_questions['text_norm']))

train_examples = []
for _, row in df_train.iterrows():
    text_a = qid_to_text.get(row['qid1'], "")
    text_b = qid_to_text.get(row['qid2'], "")
    label = float(row['is_duplicate'])
    if text_a and text_b:
        train_examples.append(InputExample(texts=[text_a, text_b], label=label))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

model = SentenceTransformer('all-MiniLM-L6-v2')
train_loss = losses.ContrastiveLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path='./finetuned-minilm'
)

class CustomGraphIndex:
    def __init__(self, data):
        self.data = data

    def search(self, queries, k=20):
        distances = 1 - cdist(queries, self.data, metric='cosine')
        indices = np.argsort(-distances, axis=1)[:, :k]
        scores = np.take_along_axis(distances, indices, axis=1)
        return scores, indices

custom_index = CustomGraphIndex(embeddings)

start_time = time.time()
custom_scores, custom_indices = custom_index.search(sample_queries, k)
custom_time = time.time() - start_time

print(f"Custom implementation search time: {custom_time:.4f} seconds")