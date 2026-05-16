"""
Core recommendation logic for the HomLuv preference prototype.

Plain NumPy / pandas / scikit-learn. No deep learning here -- the heavy
image model runs once in prepare_data.py and only its output (number
vectors) is used below.

The idea, in one line: average the image vectors the user "liked" into a
single taste vector, then rank all homes by how close they are to it.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_artifacts(artifacts_dir="artifacts"):
    """Load the precomputed image vectors and the cleaned listings table."""
    embeddings = np.load(os.path.join(artifacts_dir, "embeddings.npy"))
    listings = pd.read_csv(os.path.join(artifacts_dir, "listings.csv"))
    return embeddings, listings


def cosine_similarity(vector, matrix):
    """How close one taste vector is to every home vector (1 = identical).

    Cosine similarity = dot product divided by the lengths. We compare
    direction, not size, so a bright photo and a dim one of the same
    style still score as similar.
    """
    vector_norm = vector / (np.linalg.norm(vector) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ vector_norm


def build_taste_profile(embeddings, liked_idx, disliked_idx=None):
    """Turn the user's liked/disliked homes into one 'taste' vector.

    Taste = average of liked vectors, minus a little of the disliked
    average so the profile leans away from styles they rejected.
    """
    liked_idx = list(liked_idx)
    taste = embeddings[liked_idx].mean(axis=0)
    if disliked_idx:
        disliked_idx = list(disliked_idx)
        taste = taste - 0.5 * embeddings[disliked_idx].mean(axis=0)
    return taste


def recommend(embeddings, listings, taste, exclude_idx=None, top_n=6):
    """Return the top-N homes most similar to the taste vector."""
    scores = cosine_similarity(taste, embeddings)
    result = listings.copy()
    result["match_score"] = scores
    if exclude_idx:
        result = result.drop(index=list(exclude_idx), errors="ignore")
    return result.sort_values("match_score", ascending=False).head(top_n)


def train_preference_model(embeddings, liked_idx, disliked_idx):
    """Small logistic regression on the user's own clicks (in-session).

    Returns None until the user has given at least one like AND one
    dislike -- a classifier needs both classes to learn anything.
    """
    if not liked_idx or not disliked_idx:
        return None
    idx = list(liked_idx) + list(disliked_idx)
    X = embeddings[idx]
    y = [1] * len(liked_idx) + [0] * len(disliked_idx)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def filter_by_budget(listings, max_price=None, zipcode=None):
    """Keep only homes inside the buyer's price range / location."""
    out = listings
    if max_price is not None:
        out = out[out["price"] <= max_price]
    if zipcode is not None:
        out = out[out["zipcode"] == zipcode]
    return out


def match_builder(recommended):
    """Pick the builder that appears most among the recommended homes."""
    if recommended.empty or "builder" not in recommended.columns:
        return None, 0
    counts = recommended["builder"].value_counts()
    return counts.index[0], int(counts.iloc[0])
