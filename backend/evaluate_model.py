# evaluate_model.py
import pandas as pd
import numpy as np
import pickle
import math
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# Load artifacts
with open('models/predicted_ratings.pkl', 'rb') as f:
    predicted_ratings = pickle.load(f)
with open('models/user_id_to_index.pkl', 'rb') as f:
    user_id_to_index = pickle.load(f)
with open('models/movie_id_to_index.pkl', 'rb') as f:
    movie_id_to_index = pickle.load(f)
with open('models/item_sim_matrix.pkl', 'rb') as f:
    item_sim_df = pickle.load(f)
with open('models/item_sim_genre.pkl', 'rb') as f:
    item_sim_genre_df = pickle.load(f)
with open('models/moviemeta.pkl', 'rb') as f:
    moviemeta = pickle.load(f)
with open('models/movie_stats.pkl', 'rb') as f:
    movie_stats = pickle.load(f)

ratings = pd.read_csv('data/ratings.csv')

# Leave-2-out split function
def leave_out_split(df, n=2, seed=42):
    train_idx, test_idx = [], []
    np.random.seed(seed)
    for uid, user_group in df.groupby('userId'):
        idx = user_group.index.tolist()
        if len(idx) <= n:
            train_idx += idx
        else:
            test_samples = np.random.choice(idx, size=n, replace=False)
            train_samples = list(set(idx) - set(test_samples))
            train_idx += train_samples
            test_idx += list(test_samples)
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)

train_ratings, test_ratings = leave_out_split(ratings, n=2)

def predict_svd_rating(user_id, movie_id):
    if user_id not in user_id_to_index or movie_id not in movie_id_to_index:
        return np.nan
    return predicted_ratings[user_id_to_index[user_id], movie_id_to_index[movie_id]]

# RMSE/MAE
y_true, y_pred = [], []
for _, row in test_ratings.iterrows():
    pred = predict_svd_rating(row['userId'], row['movieId'])
    if not np.isnan(pred):
        y_true.append(row['rating'])
        y_pred.append(pred)
if len(y_true) > 0:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print("Test RMSE:", round(rmse, 4))
    print("Test MAE:", round(mae, 4))
else:
    print("No overlapping test ratings for RMSE/MAE")

# Top-N evaluation utilities
def top_n_svd(user_id, N=10, train_df=None):
    if train_df is None: train_df = train_ratings
    user_rated = set(train_df[train_df['userId'] == user_id]['movieId'])
    scores = []
    for m in movie_id_to_index:
        if m in user_rated:
            continue
        s = predict_svd_rating(user_id, m)
        if not np.isnan(s):
            scores.append((m, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scores[:N]]

def metric_top_n(user_id, top_ids, test_df, threshold=4.0):
    rel = set(test_df[(test_df['userId'] == user_id) & (test_df['rating'] >= threshold)]['movieId'])
    if not rel:
        return None
    hits = len(set(top_ids) & rel)
    precision = hits / len(top_ids)
    recall = hits / len(rel)
    # NDCG
    dcg = 0.0
    for i, m in enumerate(top_ids):
        if m in rel:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(rel), len(top_ids)))])
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    return precision, recall, ndcg

# Evaluate top-k metrics on a sample of users
unique_test_users = test_ratings['userId'].unique()
sample_users = unique_test_users[:200]  # use up to first 200 users for speed

precisions, recalls, ndcgs = [], [], []
for uid in sample_users:
    top_ids = top_n_svd(uid, N=10, train_df=train_ratings)
    m = metric_top_n(uid, top_ids, test_ratings)
    if m:
        p, r, n = m
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

if precisions:
    print("Mean Precision@10:", round(np.mean(precisions), 4))
    print("Mean Recall@10:", round(np.mean(recalls), 4))
    print("Mean NDCG@10:", round(np.mean(ndcgs), 4))
else:
    print("No top-N metrics could be computed on sample")

# Coverage (fraction of catalog recommended at least once in all top lists)
all_recommended = set()
for uid in sample_users:
    all_recommended.update(top_n_svd(uid, N=10, train_df=train_ratings))
coverage = len(all_recommended) / len(movie_id_to_index)
print("Catalog coverage @10 (sample users):", round(coverage, 4))

# Diversity (average pairwise genre dissimilarity using item_sim_genre_df)
def diversity_of_list(top_list):
    if len(top_list) < 2: return 0.0
    sims = []
    for i in range(len(top_list)):
        for j in range(i+1, len(top_list)):
            mi, mj = top_list[i], top_list[j]
            try:
                s = item_sim_genre_df.at[mi, mj]
            except Exception:
                s = 0.0
            sims.append(s)
    if not sims:
        return 0.0
    avg_sim = np.mean(sims)
    # diversity = 1 - avg similarity
    return 1.0 - avg_sim

diversities = []
for uid in sample_users:
    top_ids = top_n_svd(uid, N=10, train_df=train_ratings)
    diversities.append(diversity_of_list(top_ids))
print("Mean Diversity (1 - avg genre-sim) @10:", round(np.mean(diversities), 4))

# Novelty (average inverse popularity rank)
pop_rank = movie_stats.sort_values('popularity_score', ascending=False).reset_index(drop=True)
movie_to_rank = {mid: idx+1 for idx, mid in enumerate(pop_rank['movieId'])}
novelties = []
for uid in sample_users:
    top_ids = top_n_svd(uid, N=10, train_df=train_ratings)
    ranks = [movie_to_rank.get(m, len(movie_to_rank)) for m in top_ids]
    # novelty = average log(rank) (higher = more novel because rank is worse)
    novelties.append(np.mean([math.log1p(r) for r in ranks]))
print("Mean Novelty (avg log rank) @10:", round(np.mean(novelties), 4))

# Personalization (1 - avg overlap between users recommendation lists)
def jaccard(a, b):
    if not a and not b: return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    uni = len(A | B)
    return inter / uni if uni > 0 else 0.0

pairs = 0
jacc_sums = 0.0
user_tops = {}
for uid in sample_users:
    user_tops[uid] = top_n_svd(uid, N=10, train_df=train_ratings)

u_list = list(user_tops.keys())
for i in range(len(u_list)):
    for j in range(i+1, len(u_list)):
        a = user_tops[u_list[i]]
        b = user_tops[u_list[j]]
        jacc_sums += jaccard(a, b)
        pairs += 1
if pairs > 0:
    avg_jacc = jacc_sums / pairs
    personalization = 1.0 - avg_jacc
    print("Personalization (1 - avg Jaccard overlap):", round(personalization, 4))
else:
    print("Not enough users to compute personalization")

print("\nEvaluation complete.")
