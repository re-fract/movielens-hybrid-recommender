# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("Loading models...")

# Load all trained models (paths as saved by train_model.py)
with open('models/U.pkl', 'rb') as f:
    U = pickle.load(f)
with open('models/sigma.pkl', 'rb') as f:
    sigma = pickle.load(f)
with open('models/Vt.pkl', 'rb') as f:
    Vt = pickle.load(f)
with open('models/predicted_ratings.pkl', 'rb') as f:
    predicted_ratings = pickle.load(f)
with open('models/user_ratings_mean.pkl', 'rb') as f:
    user_ratings_mean = pickle.load(f)
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
with open('models/movieidmap.pkl', 'rb') as f:
    movieid_map = pickle.load(f)
with open('models/genre_matrix.pkl', 'rb') as f:
    genre_matrix = pickle.load(f)
with open('models/movie_stats.pkl', 'rb') as f:
    movie_stats = pickle.load(f)
# optional eval summary produced during training
try:
    with open('models/eval_summary.pkl', 'rb') as f:
        eval_summary = pickle.load(f)
except Exception:
    eval_summary = {}

user_ratings_df = pd.read_csv('data/ratings.csv')

# TMDB API Configuration (move key to env var in production)
TMDB_API_KEY = "a07341c35bdfc70aac5cfcc9ddaa8441"
TMDB_CACHE = {}

print("✓ All models loaded successfully!")
print(f"✓ SVD Matrix factorization: {U.shape[0]} users × {Vt.shape[1]} movies with {sigma.shape[0]} factors")

# -----------------------------
# Utilities
# -----------------------------
import re

def get_poster_url(title):
    """Robust movie poster fetcher with title cleanup, year extraction, caching & fallbacks."""
    # Extract year if present like "(1994)"
    year_match = re.search(r'\((\d{4})\)', title)
    year = int(year_match.group(1)) if year_match else None

    # Clean title (remove year and extra spaces)
    clean_title = re.sub(r'\s*\(\d{4}\)\s*', '', title).strip()

    cache_key = f"{clean_title}_{year}" if year else clean_title
    if cache_key in TMDB_CACHE:
        return TMDB_CACHE[cache_key]

    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "include_adult": False,
            "page": 1,
        }
        if year:
            params["year"] = year

        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])

        poster_url = None
        if data:
            # Try exact title match first
            for result in data:
                tmdb_title = result.get("title", "").lower()
                if clean_title.lower() in tmdb_title or tmdb_title.lower() in clean_title.lower():
                    if result.get("poster_path"):
                        poster_url = f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                        break

            # Fallback to first poster in results
            if not poster_url:
                for result in data:
                    if result.get("poster_path"):
                        poster_url = f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                        break

        # If TMDB fails, try OMDb as backup (optional, free tier)
        if not poster_url:
            omdb_url = f"http://www.omdbapi.com/?t={clean_title}&y={year or ''}&apikey=4a3b711b"
            omdb_resp = requests.get(omdb_url, timeout=5)
            omdb_data = omdb_resp.json()
            if omdb_data.get("Poster") and omdb_data["Poster"] != "N/A":
                poster_url = omdb_data["Poster"]

        # Final fallback placeholder
        if not poster_url:
            poster_url = "https://via.placeholder.com/500x750?text=Poster+Unavailable"

        TMDB_CACHE[cache_key] = poster_url
        return poster_url

    except Exception as e:
        print(f"⚠️ Poster fetch error for {title}: {e}")
        return "https://via.placeholder.com/500x750?text=Poster+Unavailable"



def get_diverse_sample(n=15):
    """Get diverse movie sample across different genres and popularity levels."""
    movies_with_stats = moviemeta.merge(movie_stats, on='movieId', how='left')
    popular_movies = movies_with_stats[movies_with_stats['num_ratings'] >= 50]
    if len(popular_movies) < n:
        popular_movies = movies_with_stats
    popular_movies = popular_movies.sort_values('num_ratings', ascending=False)
    tier_size = max(1, len(popular_movies) // 3)
    tier1 = popular_movies.iloc[:tier_size].sample(min(5, tier_size))
    tier2 = popular_movies.iloc[tier_size:2*tier_size].sample(min(5, tier_size))
    tier3 = popular_movies.iloc[2*tier_size:].sample(min(5, max(1, len(popular_movies) - 2*tier_size)))
    sample = pd.concat([tier1, tier2, tier3]).sample(frac=1).head(n)
    return sample[['movieId', 'title', 'genres']]

# -----------------------------
# Recommendation functions
# -----------------------------
def predict_svd_rating(user_id, movie_id):
    if user_id not in user_id_to_index or movie_id not in movie_id_to_index:
        return None
    user_idx = user_id_to_index[user_id]
    movie_idx = movie_id_to_index[movie_id]
    return float(predicted_ratings[user_idx, movie_idx])

def popularity_based_recommendations(n_recommendations=10):
    popular = moviemeta.merge(movie_stats, on='movieId')
    popular = popular.sort_values('popularity_score', ascending=False)
    results = []
    for _, movie in popular.head(n_recommendations).iterrows():
        poster_url = get_poster_url(movie['title'])
        results.append({
            'movieId': int(movie['movieId']),
            'title': movie['title'],
            'genres': movie['genres'],
            'score': float(movie['popularity_score'] / 5.0),
            'posterUrl': poster_url
        })
    return results

def hybrid_recommendations_svd(user_id, n_recommendations=10, alpha=0.6, beta=0.25, gamma=0.15):
    """
    Hybrid recommendation using:
    - SVD predictions (alpha)
    - Item-based CF (rating similarity) (beta)
    - Genre-based similarity (gamma)
    """
    if user_id not in user_id_to_index:
        return popularity_based_recommendations(n_recommendations)

    user_idx = user_id_to_index[user_id]
    user_rated = set(user_ratings_df[user_ratings_df['userId'] == user_id]['movieId'])
    if not user_rated:
        return popularity_based_recommendations(n_recommendations)

    scores = []
    for movie_id in moviemeta['movieId']:
        if movie_id in user_rated or movie_id not in movie_id_to_index:
            continue
        movie_idx = movie_id_to_index[movie_id]

        # SVD prediction (normalized 0-1)
        svd_pred = predicted_ratings[user_idx, movie_idx]
        svd_score = np.clip(svd_pred, 0, 5) / 5.0

        # Item-based CF score (rating similarity) — average similarity to user's rated items weighted by their ratings
        cf_score = 0.0
        sims = []
        for rated in user_rated:
            try:
                sim = item_sim_df.at[movie_id, rated]
            except Exception:
                sim = 0.0
            sims.append(sim)
        if sims:
            cf_score = np.mean(sims)

        # Genre-based content similarity (avg similarity to rated items)
        cbf_score = 0.0
        try:
            if movie_id in item_sim_genre_df.index:
                sims_g = [item_sim_genre_df.at[movie_id, r] for r in user_rated if r in item_sim_genre_df.columns]
                if sims_g:
                    cbf_score = float(np.mean(sims_g))
        except Exception:
            cbf_score = 0.0

        # Popularity score (normalize to 0-1)
        pop_score = 0.0
        movie_stat = movie_stats[movie_stats['movieId'] == movie_id]
        if not movie_stat.empty:
            pop_score = float(movie_stat.iloc[0]['popularity_score'] / 5.0)

        final_score = alpha * svd_score + beta * cf_score + gamma * cbf_score + (0.05 * pop_score)
        scores.append((movie_id, final_score, svd_pred))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    seen_titles = set()
    for mid, score, pred_rating in scores[:n_recommendations * 3]:
        movie_row = moviemeta[moviemeta['movieId'] == mid].iloc[0]
        title = movie_row['title']
        if title in seen_titles:
            continue
        seen_titles.add(title)
        poster_url = get_poster_url(title)
        results.append({
            'movieId': int(mid),
            'title': title,
            'genres': movie_row['genres'],
            'score': float(score),
            'predicted_rating': float(np.clip(pred_rating, 0, 5)),
            'posterUrl': poster_url
        })
        if len(results) >= n_recommendations:
            break

    if not results:
        return popularity_based_recommendations(n_recommendations)
    return results

def recommend_for_new_user_svd(user_selections, n_recommendations=10, alpha=0.6, beta=0.4):
    """
    For a brand-new user: combine item-based CF (using item_sim_df) and genre-based TF-IDF similarity.
    user_selections: list of {'movieId': int, 'rating': float}
    """
    if not user_selections:
        return popularity_based_recommendations(n_recommendations)

    selected_movie_ids = [sel['movieId'] for sel in user_selections]
    user_ratings_map = {sel['movieId']: sel.get('rating', 3.0) for sel in user_selections}

    scores = []
    for movie_id in moviemeta['movieId']:
        if movie_id in selected_movie_ids:
            continue
        # Genre-based score
        cbf_score = 0.0
        if movie_id in item_sim_genre_df.index:
            sims = []
            for sid in selected_movie_ids:
                if sid in item_sim_genre_df.columns:
                    sims.append(item_sim_genre_df.at[movie_id, sid])
            if sims:
                # weight each by the user's provided rating (normalized)
                weights = [user_ratings_map.get(sid, 3.0) / 5.0 for sid in selected_movie_ids if sid in item_sim_genre_df.columns]
                if weights:
                    cbf_score = float(np.average(sims, weights=weights))
                else:
                    cbf_score = float(np.mean(sims))

        # Item-based CF score
        cf_score = 0.0
        numer, denom = 0.0, 0.0
        for sid, r in user_ratings_map.items():
            if movie_id in item_sim_df.index and sid in item_sim_df.columns:
                sim = item_sim_df.at[movie_id, sid]
                numer += sim * r
                denom += abs(sim)
        cf_score = (numer / denom / 5.0) if denom != 0 else 0.0

        final_score = alpha * cf_score + beta * cbf_score
        scores.append((movie_id, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    seen_titles = set()
    for mid, score in scores[:n_recommendations * 3]:
        movie = moviemeta[moviemeta['movieId'] == mid].iloc[0]
        title = movie['title']
        if title in seen_titles:
            continue
        seen_titles.add(title)
        poster_url = get_poster_url(title)
        results.append({
            'movieId': int(mid),
            'title': title,
            'genres': movie['genres'],
            'score': float(score),
            'posterUrl': poster_url
        })
        if len(results) >= n_recommendations:
            break

    if not results:
        return popularity_based_recommendations(n_recommendations)
    return results

# -----------------------------
# API Endpoints
# -----------------------------
@app.route('/movies-sample', methods=['GET'])
def movies_sample():
    try:
        sample = get_diverse_sample(15)
        sample["posterUrl"] = sample["title"].apply(get_poster_url)
        return jsonify(sample.to_dict(orient="records"))
    except Exception as e:
        print(f"Error in movies-sample: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_id = int(data.get('userId'))
        n = int(data.get('numRecommendations', 10))
        alpha = float(data.get('alpha', 0.6))
        beta = float(data.get('beta', 0.25))
        gamma = float(data.get('gamma', 0.15))

        if user_id not in user_id_to_index:
            return jsonify({'error': f'User {user_id} not found in training data'}), 404

        recs = hybrid_recommendations_svd(user_id, n_recommendations=n, alpha=alpha, beta=beta, gamma=gamma)
        return jsonify({'recommendations': recs})
    except Exception as e:
        print(f"Error in recommend: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommend-newuser', methods=['POST'])
def recommend_new_user():
    try:
        data = request.json
        user_selections = data.get('userSelections', [])
        n = int(data.get('numRecommendations', 10))
        alpha = float(data.get('alpha', 0.6))
        beta = float(data.get('beta', 0.4))

        if not user_selections:
            return jsonify({'error': 'No user selections provided'}), 400

        recs = recommend_for_new_user_svd(user_selections, n_recommendations=n, alpha=alpha, beta=beta)
        return jsonify({'recommendations': recs})
    except Exception as e:
        print(f"Error in recommend-newuser: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/eval-summary', methods=['GET'])
def eval_summary_endpoint():
    try:
        return jsonify({'eval_summary': eval_summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'num_movies': int(len(moviemeta)),
        'num_ratings': int(len(user_ratings_df)),
        'svd_factors': int(sigma.shape[0]),
        'eval_summary_present': bool(eval_summary)
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Movie Recommendation System - Flask Server (Hybrid SVD)")
    print("=" * 60)
    print(f"Loaded {len(moviemeta):,} movies")
    print(f"Loaded {len(user_ratings_df):,} ratings")
    print(f"SVD with {sigma.shape[0]} latent factors")
    print("Server starting on http://127.0.0.1:5001")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
