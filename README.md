# MovieLens SVD Hybrid Recommender

A full-stack movie recommendation system using a hybrid recommendation engine that blends:
- SVD (SciPy svds) collaborative filtering
- TF-IDF genre-based content similarity
- Item-item cosine similarity

New users rate 15 movies for personalized cold-start recommendations.
Existing users can fetch top-N recommendations using a precomputed user×movie prediction matrix.

---

## Features

- Hybrid CF + Content + Item Similarity model
- Cold-start onboarding flow (15 diverse movies)
- Popularity-aware sampling to reduce bias
- TMDB poster fetching with caching
- Evaluated using RMSE, MAE, Precision@K, Recall@K, NDCG@K

---

## Tech Stack

Backend: Python, Flask, NumPy, SciPy, scikit-learn, Pandas, Requests  
Frontend: React, Material UI  
Data: MovieLens ratings.csv and movies.csv + TMDB

---

## Project Structure

backend/  
├── app.py  
├── train_model.py  
├── evaluate_model.py  
├── models/  
└── data/  

frontend/  
└── src/  

---

## Backend Setup

    cd backend
    pip install -r requirements.txt
    python train_model.py

Set TMDB API Key:

Mac/Linux:
    export TMDB_API_KEY=your_key_here

Windows:
    set TMDB_API_KEY=your_key_here

Run the Backend:
    python app.py

Backend runs at: http://localhost:5001

---

## Frontend Setup

    cd frontend
    npm install
    npm start

Ensure API_URL or proxy points to backend.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /movies-sample        | Get 15 movies for cold-start rating |
| POST   | /recommend            | Top-N recommendations for a userId |
| POST   | /recommend-newuser    | Cold-start recs based on user selections |

---

## Evaluation

Run evaluation:

    cd backend
    python evaluate_model.py

Outputs:
- RMSE / MAE
- Precision@10 / Recall@10 / NDCG@10

---

## Configuration

Hybrid weights parameters:
- alpha = SVD score weight
- beta = content similarity weight
- gamma = popularity weight

Latent factors and hyperparameters can be tuned in train_model.py

---

## Future Work

- User auth and persistent profiles
- Rich metadata embeddings (keywords, cast, tags)
- Auto weight tuning
- Docker deployment

---

## Credits

- MovieLens Dataset — GroupLens Research
- TMDB Poster API
