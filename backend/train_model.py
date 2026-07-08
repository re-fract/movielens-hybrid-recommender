# train_model.py
import pandas as pd
import numpy as np
import pickle
import os
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from collections import defaultdict
import math
import time

np.random.seed(42)

print("=" * 60)
print("Starting Model Training with SVD + Hybrid components...")
print("=" * 60)

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Load datasets
print("\n[1/8] Loading datasets...")
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

print(f"   - Loaded {len(ratings):,} ratings")
print(f"   - Loaded {len(movies):,} movies")

# Ensure consistent movie set
common_movies = sorted(set(ratings['movieId']).intersection(movies['movieId']))
ratings = ratings[ratings['movieId'].isin(common_movies)].reset_index(drop=True)
movies = movies[movies['movieId'].isin(common_movies)].sort_values('movieId').reset_index(drop=True)

print(f"   - Common movies: {len(common_movies):,}")

# ============================================================
# PART 1: Create User-Movie Matrix
# ============================================================
print("\n[2/8] Creating user-movie rating matrix...")

user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating',
    fill_value=0
)

print(f"   - Matrix shape: {user_movie_matrix.shape}")
print(f"   - Users: {user_movie_matrix.shape[0]:,}")
print(f"   - Movies: {user_movie_matrix.shape[1]:,}")

# Save user and movie mappings
user_ids = user_movie_matrix.index.tolist()
movie_ids = user_movie_matrix.columns.tolist()

user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

with open('models/user_id_to_index.pkl', 'wb') as f:
    pickle.dump(user_id_to_index, f)
with open('models/movie_id_to_index.pkl', 'wb') as f:
    pickle.dump(movie_id_to_index, f)

print("   ✓ Saved user and movie mappings")

# ============================================================
# PART 2: Normalize ratings (mean-centering)
# ============================================================
print("\n[3/8] Normalizing ratings (mean-centering)...")

R = user_movie_matrix.values.astype(float)
num_users, num_movies = R.shape

# Compute per-user mean using non-zero entries
user_ratings_mean = np.zeros(num_users)
for i in range(num_users):
    rated = R[i, :] > 0
    if rated.sum() > 0:
        user_ratings_mean[i] = R[i, rated].mean()
    else:
        user_ratings_mean[i] = 0.0

# Demean only existing ratings (leave zeros)
R_demeaned = R.copy()
for i in range(num_users):
    mask = R[i, :] > 0
    if mask.any():
        R_demeaned[i, mask] = R[i, mask] - user_ratings_mean[i]

print(f"   - Mean rating (across users with ratings): {user_ratings_mean[user_ratings_mean > 0].mean():.3f}")

with open('models/user_ratings_mean.pkl', 'wb') as f:
    pickle.dump(user_ratings_mean, f)
print("   ✓ Saved user_ratings_mean.pkl")

# ============================================================
# Helper: Per-user leave-k-out split (for hyperparam tuning)
# ============================================================
def leave_k_out(df, k=2, seed=42):
    np.random.seed(seed)
    train_rows = []
    test_rows = []
    grouped = df.groupby('userId')
    for uid, group in grouped:
        if len(group) <= k:
            train_rows += group.index.tolist()
        else:
            test_idx = np.random.choice(group.index, size=k, replace=False).tolist()
            train_idx = list(set(group.index.tolist()) - set(test_idx))
            train_rows += train_idx
            test_rows += test_idx
    return df.loc[train_rows].reset_index(drop=True), df.loc[test_rows].reset_index(drop=True)

train_ratings_cv, test_ratings_cv = leave_k_out(ratings, k=2)
print("   ✓ Created leave-2-out split for quick CV")

# ============================================================
# PART 3: Hyperparameter search + SVD training
# ============================================================
print("\n[4/8] SVD training & hyperparameter search...")

# We search a small grid for k (latent dims). Adjust as needed.
candidate_ks = [20, 50, 100]
best_k = candidate_ks[0]
best_ndcg = -1.0
eval_summary = {}

# Precompute train user-movie matrix for CV
train_matrix_cv = train_ratings_cv.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
# align to global user/movie ordering (some users/movies might be missing in leave-out train)
train_matrix_cv = train_matrix_cv.reindex(index=user_ids, columns=movie_ids, fill_value=0).values

for k in candidate_ks:
    t0 = time.time()
    print(f"   - Trying k={k} ...")
    try:
        U, sigma_vals, Vt = svds(train_matrix_cv.astype(float), k=k)
    except Exception as e:
        print(f"     ! SVD failed for k={k}: {e}")
        continue

    sigma = np.diag(sigma_vals)
    pred_matrix = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

    # Evaluate on test_ratings_cv
    y_true, y_pred = [], []
    for _, row in test_ratings_cv.iterrows():
        uid, mid, r = row['userId'], row['movieId'], row['rating']
        if uid in user_id_to_index and mid in movie_id_to_index:
            uidx = user_id_to_index[uid]
            midx = movie_id_to_index[mid]
            pred = pred_matrix[uidx, midx]
            y_true.append(r)
            y_pred.append(pred)
    if len(y_true) == 0:
        print("     - No test ratings to evaluate for this k")
        continue

    from sklearn.metrics import mean_squared_error
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    # Quick top-N NDCG estimate on a sample of users
    def ndcg_at_k_for_user(uid, k_rec=10):
        if uid not in user_id_to_index: return 0.0
        uidx = user_id_to_index[uid]
        # movies the user rated in train (exclude)
        rated_train = set(train_ratings_cv[train_ratings_cv['userId'] == uid]['movieId'])
        # generate top from pred_matrix
        scores = []
        for m in movie_ids:
            if m in rated_train: continue
            midx = movie_id_to_index[m]
            scores.append((m, pred_matrix[uidx, midx]))
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = [m for m, _ in scores[:k_rec]]
        # relevance from test set (>=4 considered relevant)
        rel = set(test_ratings_cv[(test_ratings_cv['userId'] == uid) & (test_ratings_cv['rating'] >= 4.0)]['movieId'])
        if not rel: return 0.0
        dcg = 0.0
        for i, m in enumerate(topk):
            if m in rel:
                dcg += 1.0 / np.log2(i + 2)
        ideal_dcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(rel), len(topk)))])
        return (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    sample_users = test_ratings_cv['userId'].unique()[:200]  # sample for speed
    ndcgs = [ndcg_at_k_for_user(uid, 10) for uid in sample_users]
    avg_ndcg = np.mean(ndcgs)

    eval_summary[k] = {'rmse_cv': rmse, 'ndcg10_cv': avg_ndcg}
    print(f"     -> RMSE: {rmse:.4f}, NDCG@10 (sample): {avg_ndcg:.4f} (time {time.time()-t0:.1f}s)")

    if avg_ndcg > best_ndcg:
        best_ndcg = avg_ndcg
        best_k = k
        best_U, best_sigma_vals, best_Vt = U, sigma_vals, Vt
        best_pred_matrix = pred_matrix.copy()

print(f"   ✓ Best k chosen: {best_k} (sample NDCG@10={best_ndcg:.4f})")

# Convert chosen sigma to proper diag matrix
sigma = np.diag(best_sigma_vals)

# Save SVD components (final)
with open('models/U.pkl', 'wb') as f:
    pickle.dump(best_U, f)
with open('models/sigma.pkl', 'wb') as f:
    pickle.dump(sigma, f)
with open('models/Vt.pkl', 'wb') as f:
    pickle.dump(best_Vt, f)

print("   ✓ SVD components saved (U, sigma, Vt)")

# Full predicted ratings matrix (using all data)
print("\n[5/8] Recomputing SVD on full demeaned matrix with best_k for final predicted matrix...")
k_final = best_k
U_full, sigma_vals_full, Vt_full = svds(R_demeaned.astype(float), k=k_final)
sigma_full = np.diag(sigma_vals_full)
all_user_predicted_ratings = np.dot(np.dot(U_full, sigma_full), Vt_full) + user_ratings_mean.reshape(-1, 1)

with open('models/predicted_ratings.pkl', 'wb') as f:
    pickle.dump(all_user_predicted_ratings, f)
print("   ✓ Predicted ratings matrix saved")

# ============================================================
# PART 6: Content-Based Filtering (Genre Matrix)
# ============================================================
print("\n[6/8] Creating content-based genre matrix...")

tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[\w-]+\b')
movies['genres_processed'] = movies['genres'].fillna('').str.replace('|', ' ')
genre_matrix = tfidf.fit_transform(movies['genres_processed'])

print(f"   - Genre matrix shape: {genre_matrix.shape}")

with open('models/genre_matrix.pkl', 'wb') as f:
    pickle.dump(genre_matrix, f)
print("   ✓ Saved genre_matrix.pkl")

# Precompute mapping movieId -> row in movies df
movieid_map = {m: i for i, m in enumerate(movies['movieId'])}
with open('models/movieidmap.pkl', 'wb') as f:
    pickle.dump(movieid_map, f)
print("   ✓ Saved movieidmap.pkl")

# ============================================================
# PART 7: Item-item similarities (rating-based and genre-based)
# ============================================================
print("\n[7/8] Computing item-item similarity matrices...")

# Rating-based similarity: normalize ratings by user mean (only where rated)
R_normalized = R.copy()
for i in range(num_users):
    mask = R[i, :] > 0
    if mask.any() and user_ratings_mean[i] > 0:
        R_normalized[i, mask] = R[i, mask] / user_ratings_mean[i]

# rating-based item similarity (may be dense)
item_sim_rating = cosine_similarity(R_normalized.T)
item_sim_rating_df = pd.DataFrame(item_sim_rating, index=movie_ids, columns=movie_ids)
with open('models/item_sim_matrix.pkl', 'wb') as f:
    pickle.dump(item_sim_rating_df, f)
print(f"   - Rating-based item similarity shape: {item_sim_rating_df.shape}")

# Genre-based similarity (sparse safe)
item_sim_genre = cosine_similarity(genre_matrix)
item_sim_genre_df = pd.DataFrame(item_sim_genre, index=movies['movieId'].tolist(), columns=movies['movieId'].tolist())
with open('models/item_sim_genre.pkl', 'wb') as f:
    pickle.dump(item_sim_genre_df, f)
print(f"   - Genre-based item similarity shape: {item_sim_genre_df.shape}")

# ============================================================
# PART 8: Movie statistics & popularity
# ============================================================
print("\n[8/8] Computing movie statistics and popularity scores...")

movie_stats = ratings.groupby('movieId').agg({'rating': ['mean', 'count']}).reset_index()
movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']

C = movie_stats['avg_rating'].mean()
m = movie_stats['num_ratings'].quantile(0.6)

def weighted_rating(x, m=m, C=C):
    v = x['num_ratings']
    Rm = x['avg_rating']
    return (v / (v + m) * Rm) + (m / (v + m) * C)

movie_stats['popularity_score'] = movie_stats.apply(weighted_rating, axis=1)

with open('models/movie_stats.pkl', 'wb') as f:
    pickle.dump(movie_stats, f)
print("   ✓ Saved movie_stats.pkl")

# Save moviemeta
with open('models/moviemeta.pkl', 'wb') as f:
    pickle.dump(movies, f)
print("   ✓ Saved moviemeta.pkl")

# Save final evaluation summary
with open('models/eval_summary.pkl', 'wb') as f:
    pickle.dump(eval_summary, f)
print("   ✓ Saved eval_summary.pkl (hyperparam CV results)")

print("\n" + "=" * 60)
print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nGenerated files in 'models/' directory:")
print("  - U.pkl, sigma.pkl, Vt.pkl (SVD components)")
print("  - predicted_ratings.pkl (Full prediction matrix)")
print("  - user_ratings_mean.pkl")
print("  - user_id_to_index.pkl, movie_id_to_index.pkl")
print("  - genre_matrix.pkl, item_sim_genre.pkl, item_sim_matrix.pkl")
print("  - moviemeta.pkl, movieidmap.pkl, movie_stats.pkl")
print("  - eval_summary.pkl")
print("\nNext: run app.py to serve recommendations.\n")
