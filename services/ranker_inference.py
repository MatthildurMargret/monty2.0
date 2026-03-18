"""
Ranker Inference - Shared utilities for feature extraction and ranking.

Fixes the feature-extraction divergence between weekly_update.py (which ignored
the embedding dimension from metadata) and recommendations.py (which respected it).
The recommendations.py version is canonical: embeddings are padded/truncated to the
dimension recorded in metadata.
"""

import numpy as np
import pandas as pd


def flatten_embedding(emb, dim=1536):
    """Flatten an embedding to a fixed dimension, padding or truncating as needed.

    Args:
        emb: Embedding value — list, ndarray, None, or float NaN.
        dim: Target dimension (default 1536).

    Returns:
        np.ndarray of shape (dim,) and dtype float32.
    """
    if emb is None or (isinstance(emb, float) and np.isnan(emb)):
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(emb, dtype=np.float32).flatten()
    if len(arr) < dim:
        arr = np.pad(arr, (0, dim - len(arr)), 'constant')
    else:
        arr = arr[:dim]
    return arr


def build_feature_matrix(df_with_features, metadata):
    """Build a numpy feature matrix from a DataFrame with prepared features.

    Args:
        df_with_features: DataFrame containing text_embedding_group1,
            text_embedding_group2, experience_embedding, plus the columns
            listed in metadata['numeric_features'] and metadata['boolean_features'].
        metadata: Feature metadata dict (from ranker_only_feature_metadata.json).
            Expected keys: numeric_features, boolean_features, and optionally
            text_embedding_group1_dim, text_embedding_group2_dim,
            experience_embedding_dim.

    Returns:
        (X, valid_indices): numpy float32 array of shape (n_valid, n_features)
            and list of original DataFrame indices for valid rows.
        Returns (None, []) if no valid rows could be built.
    """
    emb1_dim = metadata.get('text_embedding_group1_dim', 1536)
    emb2_dim = metadata.get('text_embedding_group2_dim', 1536)
    exp_dim = metadata.get('experience_embedding_dim', 1536)

    X_all = []
    valid_indices = []

    for idx, row in df_with_features.iterrows():
        try:
            vec = []
            vec.extend(flatten_embedding(row.get('text_embedding_group1'), emb1_dim))
            vec.extend(flatten_embedding(row.get('text_embedding_group2'), emb2_dim))
            vec.extend(flatten_embedding(row.get('experience_embedding'), exp_dim))

            for f in metadata['numeric_features']:
                v = row.get(f, 0.0)
                vec.append(float(0.0 if pd.isna(v) else v))

            for f in metadata['boolean_features']:
                v = row.get(f, 0)
                vec.append(int(0 if pd.isna(v) else v))

            X_all.append(vec)
            valid_indices.append(idx)
        except Exception:
            continue

    if not X_all:
        return None, []

    return np.asarray(X_all, dtype=np.float32), valid_indices


def normalize_scores(raw_scores, rmin, rmax):
    """Normalize raw ranker scores to [0, 1].

    Uses min-max normalization when bounds are available; falls back to sigmoid.

    Args:
        raw_scores: np.ndarray of raw scores from the ranker.
        rmin: Minimum bound from training metadata (or None).
        rmax: Maximum bound from training metadata (or None).

    Returns:
        np.ndarray of normalized scores in [0, 1].
    """
    if rmin is not None and rmax is not None and rmax > rmin:
        scores = (raw_scores - rmin) / (rmax - rmin)
        return np.clip(scores, 0.0, 1.0)
    return 1.0 / (1.0 + np.exp(-raw_scores))


def rank_profiles(df, ranker_inference, prepare_features_fn=None):
    """Full ranking pipeline: prepare features, predict batch, normalize, sort.

    Args:
        df: DataFrame of profiles (should contain embedding columns if prepare_features_fn
            is None, or raw profile columns if prepare_features_fn is provided).
        ranker_inference: Dict with keys: ranker, metadata, ranker_min, ranker_max.
        prepare_features_fn: Optional callable(df, verbose=False) -> df_with_features.
            If None, df is used directly for feature extraction.

    Returns:
        DataFrame with 'ranker_score' column added and sorted descending.
        Returns original df unchanged on any failure so callers can apply their
        own fallback sorting.
    """
    if ranker_inference is None or df.empty:
        return df

    try:
        if prepare_features_fn is not None:
            df_feat = prepare_features_fn(df.copy(), verbose=False)
        else:
            df_feat = df.copy()

        metadata = ranker_inference['metadata']
        X, valid_indices = build_feature_matrix(df_feat, metadata)

        if X is None or len(valid_indices) == 0:
            raise RuntimeError("No valid feature rows for ranking")

        raw_scores = ranker_inference['ranker'].predict(X)
        scores = normalize_scores(
            raw_scores,
            ranker_inference['ranker_min'],
            ranker_inference['ranker_max'],
        )

        result = df.copy()
        result['ranker_score'] = 0.0
        result.loc[valid_indices, 'ranker_score'] = scores
        result.sort_values(by='ranker_score', ascending=False, inplace=True)
        return result

    except Exception as e:
        print(f"    ⚠️ Ranker failed: {e}")
        return df
