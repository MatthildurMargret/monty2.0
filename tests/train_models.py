"""
ML Training Pipeline

This module implements a complete ML pipeline for training and evaluating
classification and ranking models on the datasets produced by build_dataset.py.

It trains:
- XGBoostClassifier (3-class: Irrelevant, Relevant-but-Pass, IC-worthy)
- XGBoostRanker (pairwise ranking model)

Saves models in XGBoost native JSON format (not pickle) with separate
feature metadata files to avoid shape mismatches.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("xgboost is required. Install with: pip install xgboost>=2.0.0")

# Dataset paths
CLASSIFICATION_DATASET = 'data/datasets/classification_dataset.parquet'
RANKING_DATASET = 'data/datasets/ranking_dataset.parquet'

# Output directories
MODEL_DIR = 'data/models'
PREDICTIONS_DIR = 'data/predictions'

# Create output directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def flatten_embedding(embedding):
    """
    Flatten an embedding array/list into a numpy array.
    
    Args:
        embedding: Can be list, numpy array, or None
        
    Returns:
        numpy.ndarray: Flattened embedding vector
    """
    if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
        # Return zero vector if missing (1536 dims for text-embedding-3-small)
        return np.zeros(1536, dtype=np.float32)
    
    if isinstance(embedding, (list, np.ndarray)):
        arr = np.array(embedding, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        else:
            return arr.flatten()
    
    # Fallback: return zero vector
    return np.zeros(1536, dtype=np.float32)


def extract_classification_features(df):
    """
    Extract features from classification dataset.
    
    Args:
        df: DataFrame with classification data
        
    Returns:
        tuple: (X, y, feature_names, metadata)
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
            metadata: Dict with feature metadata
    """
    print("Extracting classification features...")
    
    n_samples = len(df)
    feature_rows = []
    feature_names = []
    
    # Embedding features (1536 dims each)
    text_emb_dim = 1536
    exp_emb_dim = 1536
    
    # Text embedding group 1 features (company/founder info)
    for i in range(text_emb_dim):
        feature_names.append(f'text_embedding_group1_{i}')
    
    # Text embedding group 2 features (product/market info)
    for i in range(text_emb_dim):
        feature_names.append(f'text_embedding_group2_{i}')
    
    # Experience embedding features
    for i in range(exp_emb_dim):
        feature_names.append(f'experience_embedding_{i}')
    
    # Numeric features (already scaled)
    numeric_features = [
        'years_of_experience_scaled',
        'years_building_scaled',
        'lastroundvaluation_scaled',
        'latestdealamount_scaled',
        'headcount_scaled'
    ]
    feature_names.extend(numeric_features)
    
    # Boolean features
    boolean_features = ['technical', 'repeat_founder']
    feature_names.extend(boolean_features)
    
    print(f"  Total features: {len(feature_names)}")
    print(f"  Processing {n_samples} samples...")
    
    # Extract features for each row
    for idx, row in df.iterrows():
        feature_vec = []
        
        # Flatten text embedding group 1 (company/founder info)
        text_emb1 = flatten_embedding(row.get('text_embedding_group1'))
        if len(text_emb1) != text_emb_dim:
            # Pad or truncate if dimension mismatch
            if len(text_emb1) < text_emb_dim:
                text_emb1 = np.pad(text_emb1, (0, text_emb_dim - len(text_emb1)), 'constant')
            else:
                text_emb1 = text_emb1[:text_emb_dim]
        feature_vec.extend(text_emb1.tolist())
        
        # Flatten text embedding group 2 (product/market info)
        text_emb2 = flatten_embedding(row.get('text_embedding_group2'))
        if len(text_emb2) != text_emb_dim:
            # Pad or truncate if dimension mismatch
            if len(text_emb2) < text_emb_dim:
                text_emb2 = np.pad(text_emb2, (0, text_emb_dim - len(text_emb2)), 'constant')
            else:
                text_emb2 = text_emb2[:text_emb_dim]
        feature_vec.extend(text_emb2.tolist())
        
        # Flatten experience embedding
        exp_emb = flatten_embedding(row.get('experience_embedding'))
        if len(exp_emb) != exp_emb_dim:
            if len(exp_emb) < exp_emb_dim:
                exp_emb = np.pad(exp_emb, (0, exp_emb_dim - len(exp_emb)), 'constant')
            else:
                exp_emb = exp_emb[:exp_emb_dim]
        feature_vec.extend(exp_emb.tolist())
        
        # Numeric features
        for feat in numeric_features:
            val = row.get(feat, 0.0)
            if pd.isna(val):
                val = 0.0
            feature_vec.append(float(val))
        
        # Boolean features
        for feat in boolean_features:
            val = row.get(feat, 0)
            if pd.isna(val):
                val = 0
            feature_vec.append(int(val))
        
        feature_rows.append(feature_vec)
    
    X = np.array(feature_rows, dtype=np.float32)
    y = df['label'].values.astype(int)
    
    # Create metadata
    metadata = {
        'n_features': len(feature_names),
        'n_samples': n_samples,
        'text_embedding_group1_dim': text_emb_dim,
        'text_embedding_group2_dim': text_emb_dim,
        'experience_embedding_dim': exp_emb_dim,
        'numeric_features': numeric_features,
        'boolean_features': boolean_features,
        'feature_names': feature_names
    }
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y, feature_names, metadata


def extract_ranking_features(df):
    """
    Extract features from ranking dataset.
    
    Args:
        df: DataFrame with ranking data
        
    Returns:
        tuple: (X, y, group_ids, feature_names, metadata)
            X: Feature matrix (n_samples, n_features)
            y: Relevance scores (n_samples,)
            group_ids: Group IDs for ranking (n_samples,)
            feature_names: List of feature names
            metadata: Dict with feature metadata
    """
    print("Extracting ranking features...")
    
    n_samples = len(df)
    feature_rows = []
    feature_names = []
    
    # Embedding features (1536 dims each)
    text_emb_dim = 1536
    exp_emb_dim = 1536
    
    # Text embedding group 1 features (company/founder info)
    for i in range(text_emb_dim):
        feature_names.append(f'text_embedding_group1_{i}')
    
    # Text embedding group 2 features (product/market info)
    for i in range(text_emb_dim):
        feature_names.append(f'text_embedding_group2_{i}')
    
    # Experience embedding features
    for i in range(exp_emb_dim):
        feature_names.append(f'experience_embedding_{i}')
    
    # Numeric features (already scaled)
    numeric_features = [
        'years_of_experience_scaled',
        'years_building_scaled',
        'lastroundvaluation_scaled',
        'latestdealamount_scaled',
        'headcount_scaled'
    ]
    feature_names.extend(numeric_features)
    
    # Boolean features
    boolean_features = ['technical', 'repeat_founder']
    feature_names.extend(boolean_features)
    
    print(f"  Total features: {len(feature_names)}")
    print(f"  Processing {n_samples} samples...")
    
    # Extract features for each row
    for idx, row in df.iterrows():
        feature_vec = []
        
        # Text embedding group 1 (company/founder info)
        text_emb1 = flatten_embedding(row.get('text_embedding_group1'))
        if len(text_emb1) != text_emb_dim:
            if len(text_emb1) < text_emb_dim:
                text_emb1 = np.pad(text_emb1, (0, text_emb_dim - len(text_emb1)), 'constant')
            else:
                text_emb1 = text_emb1[:text_emb_dim]
        feature_vec.extend(text_emb1.tolist())
        
        # Text embedding group 2 (product/market info)
        text_emb2 = flatten_embedding(row.get('text_embedding_group2'))
        if len(text_emb2) != text_emb_dim:
            if len(text_emb2) < text_emb_dim:
                text_emb2 = np.pad(text_emb2, (0, text_emb_dim - len(text_emb2)), 'constant')
            else:
                text_emb2 = text_emb2[:text_emb_dim]
        feature_vec.extend(text_emb2.tolist())
        
        # Flatten experience embedding
        exp_emb = flatten_embedding(row.get('experience_embedding'))
        if len(exp_emb) != exp_emb_dim:
            if len(exp_emb) < exp_emb_dim:
                exp_emb = np.pad(exp_emb, (0, exp_emb_dim - len(exp_emb)), 'constant')
            else:
                exp_emb = exp_emb[:exp_emb_dim]
        feature_vec.extend(exp_emb.tolist())
        
        # Numeric features
        for feat in numeric_features:
            val = row.get(feat, 0.0)
            if pd.isna(val):
                val = 0.0
            feature_vec.append(float(val))
        
        # Boolean features
        for feat in boolean_features:
            val = row.get(feat, 0)
            if pd.isna(val):
                val = 0
            feature_vec.append(int(val))
        
        feature_rows.append(feature_vec)
    
    X = np.array(feature_rows, dtype=np.float32)
    y = df['relevance_score'].values.astype(int)
    group_ids = df['group_id'].values
    
    # Create metadata
    metadata = {
        'n_features': len(feature_names),
        'n_samples': n_samples,
        'text_embedding_group1_dim': text_emb_dim,
        'text_embedding_group2_dim': text_emb_dim,
        'experience_embedding_dim': exp_emb_dim,
        'numeric_features': numeric_features,
        'boolean_features': boolean_features,
        'feature_names': feature_names
    }
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Relevance score distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Number of unique groups: {len(np.unique(group_ids))}")
    
    return X, y, group_ids, feature_names, metadata


def compute_ndcg_at_k(y_true_sorted, k=10):
    """
    Compute NDCG@K for a sorted list of relevance scores.
    
    Args:
        y_true_sorted: Sorted list of true relevance scores (descending order)
        k: Cutoff for NDCG
        
    Returns:
        float: NDCG@K score
    """
    k = min(k, len(y_true_sorted))
    if k == 0:
        return 0.0
    
    # Compute DCG@K
    dcg = 0.0
    for i in range(k):
        rel = y_true_sorted[i]
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    
    # Compute IDCG@K (ideal DCG)
    ideal_sorted = sorted(y_true_sorted, reverse=True)
    idcg = 0.0
    for i in range(k):
        rel = ideal_sorted[i]
        idcg += (2 ** rel - 1) / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_mrr(y_true_sorted):
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        y_true_sorted: Sorted list of true relevance scores (descending order)
        
    Returns:
        float: MRR score
    """
    # Find position of first relevant item (relevance > 0)
    for i, rel in enumerate(y_true_sorted):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(y_true_sorted, k=10):
    """
    Compute Precision@K.
    
    Args:
        y_true_sorted: Sorted list of true relevance scores (descending order)
        k: Cutoff for precision
        
    Returns:
        float: Precision@K score
    """
    k = min(k, len(y_true_sorted))
    if k == 0:
        return 0.0
    
    # Count relevant items (relevance > 0) in top K
    relevant_count = sum(1 for rel in y_true_sorted[:k] if rel > 0)
    return relevant_count / k


def evaluate_ranking_per_group(y_true, y_pred, group_ids, k=10):
    """
    Compute ranking metrics per group, then average.
    
    For each unique group_id:
    1. Extract rows for that group
    2. Sort by predicted score (descending)
    3. Compute NDCG@K, MRR, Precision@K
    4. Average across all groups
    
    Args:
        y_true: True relevance scores
        y_pred: Predicted scores
        group_ids: Group IDs for each sample
        k: Cutoff for NDCG and Precision
        
    Returns:
        tuple: (avg_metrics, metrics_per_group)
    """
    metrics_per_group = []
    
    for group_id in np.unique(group_ids):
        group_mask = group_ids == group_id
        group_true = y_true[group_mask]
        group_pred = y_pred[group_mask]
        
        # Sort by predicted score (descending)
        sort_idx = np.argsort(group_pred)[::-1]
        group_true_sorted = group_true[sort_idx]
        
        # Compute metrics for this group
        ndcg = compute_ndcg_at_k(group_true_sorted, k)
        mrr = compute_mrr(group_true_sorted)
        precision = compute_precision_at_k(group_true_sorted, k)
        
        metrics_per_group.append({
            'group_id': int(group_id),
            'ndcg@k': float(ndcg),
            'mrr': float(mrr),
            'precision@k': float(precision),
            'group_size': int(len(group_true))
        })
    
    # Average across groups
    avg_metrics = {
        'ndcg@k': float(np.mean([m['ndcg@k'] for m in metrics_per_group])),
        'mrr': float(np.mean([m['mrr'] for m in metrics_per_group])),
        'precision@k': float(np.mean([m['precision@k'] for m in metrics_per_group])),
        'n_groups': len(metrics_per_group)
    }
    
    return avg_metrics, metrics_per_group


def train_classifier(X, y, feature_names, metadata):
    """
    Train XGBoostClassifier with stratified validation split.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        metadata: Feature metadata
        
    Returns:
        tuple: (classifier, X_val, y_val, y_pred, y_proba, metrics)
    """
    print("\n" + "=" * 80)
    print("Training Classification Model")
    print("=" * 80)
    
    # Stratified train/validation split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Label distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Label distribution (val): {dict(zip(*np.unique(y_val, return_counts=True)))}")
    
    # Train XGBoostClassifier
    print("\nTraining XGBoostClassifier...")
    classifier = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        tree_method='hist'
    )
    
    classifier.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions
    y_pred = classifier.predict(X_val)
    y_proba = classifier.predict_proba(X_val)
    
    # Compute metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    print(f"\nValidation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return classifier, X_val, y_val, y_pred, y_proba, metrics


def train_ranker(X, y, group_ids, feature_names, metadata):
    """
    Train XGBoostRanker with group-based validation split.
    
    Args:
        X: Feature matrix
        y: Relevance scores
        group_ids: Group IDs for ranking
        feature_names: List of feature names
        metadata: Feature metadata
        
    Returns:
        tuple: (ranker, X_val, y_val, group_ids_val, y_pred, metrics)
    """
    print("\n" + "=" * 80)
    print("Training Ranking Model")
    print("=" * 80)
    
    # Group-based train/validation split (80/20)
    # Split by unique group_ids to keep groups together
    unique_groups = np.unique(group_ids)
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    n_train_groups = int(len(unique_groups) * 0.8)
    train_groups = set(unique_groups[:n_train_groups])
    val_groups = set(unique_groups[n_train_groups:])
    
    train_mask = np.array([gid in train_groups for gid in group_ids])
    val_mask = np.array([gid in val_groups for gid in group_ids])
    
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    group_ids_train = group_ids[train_mask]
    group_ids_val = group_ids[val_mask]
    
    print(f"Training set: {len(X_train)} samples in {len(train_groups)} groups")
    print(f"Validation set: {len(X_val)} samples in {len(val_groups)} groups")
    
    # Train XGBoostRanker
    print("\nTraining XGBoostRanker...")
    ranker = xgb.XGBRanker(
        objective='rank:pairwise',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )
    
    # For XGBoostRanker, data must be sorted by group_id
    # and group parameter should be a list of group sizes in order
    # Sort training data by group_id
    sort_idx = np.argsort(group_ids_train)
    X_train_sorted = X_train[sort_idx]
    y_train_sorted = y_train[sort_idx]
    group_ids_train_sorted = group_ids_train[sort_idx]
    
    # Compute group sizes (must be in same order as sorted groups)
    unique_train_groups_sorted = np.unique(group_ids_train_sorted)
    group_sizes = [np.sum(group_ids_train_sorted == gid) for gid in unique_train_groups_sorted]
    
    ranker.fit(
        X_train_sorted, y_train_sorted,
        group=group_sizes
    )
    
    # For XGBoostRanker, we need to provide group information during prediction
    # But for evaluation, we'll use the group_ids_val
    y_pred_val = ranker.predict(X_val)
    
    # Compute ranking metrics per group
    print("\nComputing ranking metrics per group...")
    avg_metrics, metrics_per_group = evaluate_ranking_per_group(
        y_val, y_pred_val, group_ids_val, k=10
    )
    
    print(f"\nValidation Metrics (averaged across groups):")
    print(f"  NDCG@10: {avg_metrics['ndcg@k']:.4f}")
    print(f"  MRR: {avg_metrics['mrr']:.4f}")
    print(f"  Precision@10: {avg_metrics['precision@k']:.4f}")
    print(f"  Number of groups: {avg_metrics['n_groups']}")
    
    # Store ranker score normalization parameters
    # CRITICAL: Compute min/max on TRAINING predictions, not validation
    # This ensures stable normalization for inference
    print("\nComputing ranker normalization parameters on training set...")
    y_pred_train = ranker.predict(X_train_sorted)
    ranker_min = float(np.min(y_pred_train))
    ranker_max = float(np.max(y_pred_train))
    print(f"  Ranker score range (training): [{ranker_min:.4f}, {ranker_max:.4f}]")
    
    metrics = {
        **avg_metrics,
        'ranker_score_min': ranker_min,
        'ranker_score_max': ranker_max,
        'metrics_per_group': metrics_per_group[:10]  # Store first 10 for reference
    }
    
    return ranker, X_val, y_val, group_ids_val, y_pred_val, metrics


def save_models(classifier, ranker, classifier_metadata, ranker_metadata,
                classifier_metrics, ranker_metrics, model_hyperparams):
    """
    Save models and metadata in XGBoost JSON format.
    
    Args:
        classifier: Trained XGBoostClassifier
        ranker: Trained XGBoostRanker
        classifier_metadata: Feature metadata for classifier
        ranker_metadata: Feature metadata for ranker
        classifier_metrics: Classification evaluation metrics
        ranker_metrics: Ranking evaluation metrics
        model_hyperparams: Model hyperparameters
    """
    print("\n" + "=" * 80)
    print("Saving Models and Metadata")
    print("=" * 80)
    
    # Save models in XGBoost JSON format
    classifier_path = os.path.join(MODEL_DIR, 'classifier_xgb.json')
    ranker_path = os.path.join(MODEL_DIR, 'ranker_xgb.json')
    
    print(f"Saving classifier to {classifier_path}...")
    classifier.save_model(classifier_path)
    
    print(f"Saving ranker to {ranker_path}...")
    ranker.save_model(ranker_path)
    
    # Save feature metadata separately
    classifier_feat_path = os.path.join(MODEL_DIR, 'classifier_feature_metadata.json')
    ranker_feat_path = os.path.join(MODEL_DIR, 'ranker_feature_metadata.json')
    
    print(f"Saving classifier feature metadata to {classifier_feat_path}...")
    with open(classifier_feat_path, 'w') as f:
        json.dump(classifier_metadata, f, indent=2)
    
    print(f"Saving ranker feature metadata to {ranker_feat_path}...")
    with open(ranker_feat_path, 'w') as f:
        json.dump(ranker_metadata, f, indent=2)
    
    # Save model metadata
    model_metadata = {
        'training_date': datetime.now().isoformat(),
        'hyperparameters': model_hyperparams,
        'classifier_metrics': classifier_metrics,
        'ranker_metrics': ranker_metrics,
        'ranker_normalization': {
            'min': ranker_metrics.get('ranker_score_min'),
            'max': ranker_metrics.get('ranker_score_max')
        }
    }
    
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.json')
    print(f"Saving model metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("✓ All models and metadata saved successfully!")


def generate_reports(classifier_metrics, ranker_metrics, y_val_clf, y_pred_clf,
                    y_proba_clf, y_val_rank, y_pred_rank, group_ids_val):
    """
    Generate evaluation reports and save predictions.
    
    Args:
        classifier_metrics: Classification metrics dict
        ranker_metrics: Ranking metrics dict
        y_val_clf: Validation labels for classification
        y_pred_clf: Predicted labels for classification
        y_proba_clf: Predicted probabilities for classification
        y_val_rank: Validation relevance scores for ranking
        y_pred_rank: Predicted scores for ranking
        group_ids_val: Group IDs for ranking validation set
    """
    print("\n" + "=" * 80)
    print("Generating Evaluation Reports")
    print("=" * 80)
    
    # Classification report
    clf_report_path = os.path.join(MODEL_DIR, 'classification_report.txt')
    with open(clf_report_path, 'w') as f:
        f.write("Classification Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Accuracy: {classifier_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {classifier_metrics['precision']:.4f}\n")
        f.write(f"Recall: {classifier_metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {classifier_metrics['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        cm = np.array(classifier_metrics['confusion_matrix'])
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val_clf, y_pred_clf, 
                                      target_names=['Irrelevant (0)', 'Relevant-but-Pass (1)', 'IC-worthy (2)']))
    
    print(f"Saved classification report to {clf_report_path}")
    
    # Ranking report
    rank_report_path = os.path.join(MODEL_DIR, 'ranking_report.txt')
    with open(rank_report_path, 'w') as f:
        f.write("Ranking Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"NDCG@10: {ranker_metrics['ndcg@k']:.4f}\n")
        f.write(f"MRR: {ranker_metrics['mrr']:.4f}\n")
        f.write(f"Precision@10: {ranker_metrics['precision@k']:.4f}\n")
        f.write(f"Number of groups evaluated: {ranker_metrics['n_groups']}\n\n")
        f.write("Ranker Score Normalization Parameters:\n")
        f.write(f"  Min: {ranker_metrics.get('ranker_score_min', 'N/A')}\n")
        f.write(f"  Max: {ranker_metrics.get('ranker_score_max', 'N/A')}\n")
    
    print(f"Saved ranking report to {rank_report_path}")
    
    # Save validation predictions
    clf_pred_path = os.path.join(PREDICTIONS_DIR, 'validation_classifier_predictions.parquet')
    clf_pred_df = pd.DataFrame({
        'true_label': y_val_clf,
        'predicted_label': y_pred_clf,
        'prob_class_0': y_proba_clf[:, 0],
        'prob_class_1': y_proba_clf[:, 1],
        'prob_class_2': y_proba_clf[:, 2]
    })
    clf_pred_df.to_parquet(clf_pred_path, index=False, engine='pyarrow')
    print(f"Saved classification predictions to {clf_pred_path}")
    
    rank_pred_path = os.path.join(PREDICTIONS_DIR, 'validation_ranker_predictions.parquet')
    rank_pred_df = pd.DataFrame({
        'group_id': group_ids_val,
        'true_relevance': y_val_rank,
        'predicted_score': y_pred_rank
    })
    rank_pred_df.to_parquet(rank_pred_path, index=False, engine='pyarrow')
    print(f"Saved ranking predictions to {rank_pred_path}")


def compute_combined_score(classifier_probs, ranker_score, gate_threshold=0.7,
                          ranker_min=None, ranker_max=None):
    """
    Compute combined recommendation score using stable formula.
    
    Formula:
    - Gate: If prob_class_0 > threshold, filter out (return 0.0)
    - Normalize ranker_score to [0, 1] using min-max normalization
    - Primary signal: normalized ranker score
    - Blend: multiply by (prob_class_2 + prob_class_1) to boost IC-worthy and relevant items
    
    Args:
        classifier_probs: Array of [prob_class_0, prob_class_1, prob_class_2]
        ranker_score: Raw ranker score (unbounded)
        gate_threshold: Threshold for filtering junk (default 0.7)
        ranker_min: Minimum ranker score for normalization (from training)
        ranker_max: Maximum ranker score for normalization (from training)
        
    Returns:
        float: Combined recommendation score
    """
    prob_class_0, prob_class_1, prob_class_2 = classifier_probs
    
    # Gate: filter out junk
    if prob_class_0 > gate_threshold:
        return 0.0
    
    # Normalize ranker score to [0, 1] range for stability
    if ranker_min is not None and ranker_max is not None:
        # Min-max normalization
        if ranker_max > ranker_min:
            normalized_ranker = (ranker_score - ranker_min) / (ranker_max - ranker_min)
            normalized_ranker = np.clip(normalized_ranker, 0.0, 1.0)
        else:
            # Edge case: all scores are the same
            normalized_ranker = 0.5
    else:
        # Fallback: sigmoid normalization if min/max not available
        normalized_ranker = 1.0 / (1.0 + np.exp(-ranker_score))
    
    # Primary signal: normalized ranker score
    # Blend: multiply by probability of being IC-worthy or relevant
    combined = normalized_ranker * (prob_class_2 + prob_class_1)
    return float(combined)


class InferenceModule:
    """
    Inference module for generating predictions using trained models.
    
    Loads models and metadata, provides methods for:
    - Classifier probabilities (0/1/2)
    - Ranker scores (taste scores)
    - Combined recommendation scores
    """
    
    def __init__(self, model_dir='data/models'):
        """
        Initialize inference module by loading models and metadata.
        
        Args:
            model_dir: Directory containing saved models and metadata
        """
        self.model_dir = model_dir
        
        # Load feature metadata
        clf_metadata_path = os.path.join(model_dir, 'classifier_feature_metadata.json')
        rank_metadata_path = os.path.join(model_dir, 'ranker_feature_metadata.json')
        model_metadata_path = os.path.join(model_dir, 'model_metadata.json')
        
        if not os.path.exists(clf_metadata_path):
            raise FileNotFoundError(f"Classifier metadata not found: {clf_metadata_path}")
        if not os.path.exists(rank_metadata_path):
            raise FileNotFoundError(f"Ranker metadata not found: {rank_metadata_path}")
        if not os.path.exists(model_metadata_path):
            raise FileNotFoundError(f"Model metadata not found: {model_metadata_path}")
        
        with open(clf_metadata_path, 'r') as f:
            self.clf_metadata = json.load(f)
        with open(rank_metadata_path, 'r') as f:
            self.rank_metadata = json.load(f)
        with open(model_metadata_path, 'r') as f:
            self.model_metadata = json.load(f)
        
        # Load models
        classifier_path = os.path.join(model_dir, 'classifier_xgb.json')
        ranker_path = os.path.join(model_dir, 'ranker_xgb.json')
        
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier model not found: {classifier_path}")
        if not os.path.exists(ranker_path):
            raise FileNotFoundError(f"Ranker model not found: {ranker_path}")
        
        self.classifier = xgb.XGBClassifier()
        self.classifier.load_model(classifier_path)
        
        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(ranker_path)
        
        # Get normalization parameters
        norm_params = self.model_metadata.get('ranker_normalization', {})
        self.ranker_min = norm_params.get('min')
        self.ranker_max = norm_params.get('max')
        
        print(f"Loaded inference module from {model_dir}")
        print(f"  Classifier features: {self.clf_metadata['n_features']}")
        print(f"  Ranker features: {self.rank_metadata['n_features']}")
        print(f"  Ranker normalization: min={self.ranker_min}, max={self.ranker_max}")
    
    def _extract_features_from_row(self, row, metadata):
        """
        Extract features from a single row (dict or Series) using metadata.
        
        Args:
            row: Row with company data
            metadata: Feature metadata dict
            
        Returns:
            numpy.ndarray: Feature vector
        """
        feature_vec = []
        
        # Text embedding group 1 (company/founder info)
        text_emb1 = flatten_embedding(row.get('text_embedding_group1'))
        text_emb_dim = metadata.get('text_embedding_group1_dim', metadata.get('text_embedding_dim', 1536))
        if len(text_emb1) != text_emb_dim:
            if len(text_emb1) < text_emb_dim:
                text_emb1 = np.pad(text_emb1, (0, text_emb_dim - len(text_emb1)), 'constant')
            else:
                text_emb1 = text_emb1[:text_emb_dim]
        feature_vec.extend(text_emb1.tolist())
        
        # Text embedding group 2 (product/market info)
        text_emb2 = flatten_embedding(row.get('text_embedding_group2'))
        text_emb_dim2 = metadata.get('text_embedding_group2_dim', metadata.get('text_embedding_dim', 1536))
        if len(text_emb2) != text_emb_dim2:
            if len(text_emb2) < text_emb_dim2:
                text_emb2 = np.pad(text_emb2, (0, text_emb_dim2 - len(text_emb2)), 'constant')
            else:
                text_emb2 = text_emb2[:text_emb_dim2]
        feature_vec.extend(text_emb2.tolist())
        
        # Experience embedding
        exp_emb = flatten_embedding(row.get('experience_embedding'))
        exp_emb_dim = metadata['experience_embedding_dim']
        if len(exp_emb) != exp_emb_dim:
            if len(exp_emb) < exp_emb_dim:
                exp_emb = np.pad(exp_emb, (0, exp_emb_dim - len(exp_emb)), 'constant')
            else:
                exp_emb = exp_emb[:exp_emb_dim]
        feature_vec.extend(exp_emb.tolist())
        
        # Numeric features
        for feat in metadata['numeric_features']:
            val = row.get(feat, 0.0)
            if pd.isna(val):
                val = 0.0
            feature_vec.append(float(val))
        
        # Boolean features
        for feat in metadata['boolean_features']:
            val = row.get(feat, 0)
            if pd.isna(val):
                val = 0
            feature_vec.append(int(val))
        
        return np.array(feature_vec, dtype=np.float32).reshape(1, -1)
    
    def predict_classifier(self, row):
        """
        Get classifier probabilities for a company.
        
        Args:
            row: Row with company data (dict or Series)
            
        Returns:
            dict: {
                'prob_class_0': probability of Irrelevant,
                'prob_class_1': probability of Relevant-but-Pass,
                'prob_class_2': probability of IC-worthy,
                'predicted_class': predicted class (0, 1, or 2)
            }
        """
        X = self._extract_features_from_row(row, self.clf_metadata)
        probs = self.classifier.predict_proba(X)[0]
        pred = self.classifier.predict(X)[0]
        
        return {
            'prob_class_0': float(probs[0]),
            'prob_class_1': float(probs[1]),
            'prob_class_2': float(probs[2]),
            'predicted_class': int(pred)
        }
    
    def predict_ranker(self, row):
        """
        Get ranker score (taste score) for a company.
        
        Args:
            row: Row with company data (dict or Series)
            
        Returns:
            dict: {
                'ranker_score': raw ranker score (unbounded)
            }
        """
        X = self._extract_features_from_row(row, self.rank_metadata)
        score = self.ranker.predict(X)[0]
        
        return {
            'ranker_score': float(score)
        }
    
    def predict_combined(self, row, gate_threshold=0.7):
        """
        Get combined recommendation score for a company.
        
        Args:
            row: Row with company data (dict or Series)
            gate_threshold: Threshold for filtering junk (default 0.7)
            
        Returns:
            dict: {
                'classifier_probs': [prob_class_0, prob_class_1, prob_class_2],
                'ranker_score': raw ranker score,
                'combined_score': combined recommendation score
            }
        """
        clf_result = self.predict_classifier(row)
        rank_result = self.predict_ranker(row)
        
        classifier_probs = np.array([
            clf_result['prob_class_0'],
            clf_result['prob_class_1'],
            clf_result['prob_class_2']
        ])
        
        combined_score = compute_combined_score(
            classifier_probs,
            rank_result['ranker_score'],
            gate_threshold=gate_threshold,
            ranker_min=self.ranker_min,
            ranker_max=self.ranker_max
        )
        
        return {
            'classifier_probs': classifier_probs.tolist(),
            'ranker_score': rank_result['ranker_score'],
            'combined_score': combined_score,
            'predicted_class': clf_result['predicted_class']
        }


def main():
    """
    Main function to orchestrate the ML training pipeline.
    """
    print("=" * 80)
    print("ML Training Pipeline")
    print("=" * 80)
    print()
    
    # Load datasets
    print("Loading datasets...")
    if not os.path.exists(CLASSIFICATION_DATASET):
        raise FileNotFoundError(f"Classification dataset not found: {CLASSIFICATION_DATASET}")
    if not os.path.exists(RANKING_DATASET):
        raise FileNotFoundError(f"Ranking dataset not found: {RANKING_DATASET}")
    
    classification_df = pd.read_parquet(CLASSIFICATION_DATASET, engine='pyarrow')
    ranking_df = pd.read_parquet(RANKING_DATASET, engine='pyarrow')
    
    print(f"Loaded classification dataset: {len(classification_df)} rows")
    print(f"Loaded ranking dataset: {len(ranking_df)} rows")
    print()
    
    # Extract features separately for each dataset
    X_clf, y_clf, clf_feature_names, clf_metadata = extract_classification_features(classification_df)
    X_rank, y_rank, group_ids_rank, rank_feature_names, rank_metadata = extract_ranking_features(ranking_df)
    
    # Train models
    classifier, X_val_clf, y_val_clf, y_pred_clf, y_proba_clf, clf_metrics = train_classifier(
        X_clf, y_clf, clf_feature_names, clf_metadata
    )
    
    ranker, X_val_rank, y_val_rank, group_ids_val, y_pred_rank, rank_metrics = train_ranker(
        X_rank, y_rank, group_ids_rank, rank_feature_names, rank_metadata
    )
    
    # Model hyperparameters (for metadata)
    model_hyperparams = {
        'classifier': {
            'objective': 'multi:softprob',
            'num_class': 3,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        },
        'ranker': {
            'objective': 'rank:pairwise',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    }
    
    # Save models and metadata
    save_models(
        classifier, ranker,
        clf_metadata, rank_metadata,
        clf_metrics, rank_metrics,
        model_hyperparams
    )
    
    # Generate reports
    generate_reports(
        clf_metrics, rank_metrics,
        y_val_clf, y_pred_clf, y_proba_clf,
        y_val_rank, y_pred_rank, group_ids_val
    )
    
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Predictions saved to: {PREDICTIONS_DIR}")
    print("\nFiles created:")
    print("  - classifier_xgb.json")
    print("  - ranker_xgb.json")
    print("  - classifier_feature_metadata.json")
    print("  - ranker_feature_metadata.json")
    print("  - model_metadata.json")
    print("  - classification_report.txt")
    print("  - ranking_report.txt")
    print("=" * 80)


if __name__ == "__main__":
    main()

