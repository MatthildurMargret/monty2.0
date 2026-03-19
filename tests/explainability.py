"""
Model Explainability and Analysis Module

Provides tools for understanding model predictions:
- Feature importance analysis
- SHAP value computation
- Class separation analysis
- Misclassification analysis
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import json


def get_feature_importance(classifier, ranker, feature_names):
    """
    Extract feature importance from trained models.
    
    Args:
        classifier: Trained XGBoostClassifier
        ranker: Trained XGBoostRanker
        feature_names: List of feature names
        
    Returns:
        dict: Feature importance for classifier and ranker
    """
    # Get feature importance from classifier
    clf_importance = classifier.get_booster().get_score(importance_type='gain')
    # Convert to array format matching feature_names
    clf_importance_array = np.array([clf_importance.get(f'f{i}', 0.0) for i in range(len(feature_names))])
    
    # Get feature importance from ranker
    rank_importance = ranker.get_booster().get_score(importance_type='gain')
    rank_importance_array = np.array([rank_importance.get(f'f{i}', 0.0) for i in range(len(feature_names))])
    
    # Normalize to percentages
    clf_total = clf_importance_array.sum()
    rank_total = rank_importance_array.sum()
    
    clf_importance_pct = (clf_importance_array / clf_total * 100) if clf_total > 0 else clf_importance_array
    rank_importance_pct = (rank_importance_array / rank_total * 100) if rank_total > 0 else rank_importance_array
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'classifier_importance': clf_importance_pct,
        'ranker_importance': rank_importance_pct,
        'combined_importance': (clf_importance_pct + rank_importance_pct) / 2
    })
    
    return importance_df.sort_values('combined_importance', ascending=False)


def analyze_class_separation(df, label_col='label', feature_cols=None):
    """
    Analyze how well classes are separated in feature space.
    
    Args:
        df: DataFrame with labels and features
        label_col: Name of label column
        feature_cols: List of feature columns to analyze
        
    Returns:
        dict: Separation metrics and statistics
    """
    if feature_cols is None:
        # Get numeric feature columns
        feature_cols = [col for col in df.columns if col.endswith('_scaled') or col in ['technical', 'repeat_founder']]
    
    # Filter to only labeled data
    labeled_df = df[df[label_col].notna()].copy()
    
    if len(labeled_df) == 0:
        return {}
    
    # Get class distributions
    class_0 = labeled_df[labeled_df[label_col] == 0]
    class_1 = labeled_df[labeled_df[label_col] == 1]
    class_2 = labeled_df[labeled_df[label_col] == 2]
    
    separation_stats = {
        'class_counts': {
            0: len(class_0),
            1: len(class_1),
            2: len(class_2)
        },
        'class_1_vs_2_separation': {}
    }
    
    # Analyze separation between class 1 and 2 (most important for IC-worthy prediction)
    if len(class_1) > 0 and len(class_2) > 0:
        for feature in feature_cols:
            if feature in labeled_df.columns:
                mean_1 = class_1[feature].mean()
                mean_2 = class_2[feature].mean()
                std_1 = class_1[feature].std()
                std_2 = class_2[feature].std()
                
                # Compute separation metric (difference in means relative to pooled std)
                pooled_std = np.sqrt((std_1**2 + std_2**2) / 2)
                if pooled_std > 0:
                    separation = abs(mean_2 - mean_1) / pooled_std
                else:
                    separation = 0
                
                separation_stats['class_1_vs_2_separation'][feature] = {
                    'mean_class_1': float(mean_1),
                    'mean_class_2': float(mean_2),
                    'separation_score': float(separation),  # Higher = better separation
                    'overlap': float(min(mean_1 + std_1, mean_2 + std_2) - max(mean_1 - std_1, mean_2 - std_2)) if pooled_std > 0 else 0
                }
    
    # Sort by separation score
    separation_stats['class_1_vs_2_separation'] = dict(
        sorted(separation_stats['class_1_vs_2_separation'].items(), 
               key=lambda x: x[1]['separation_score'], 
               reverse=True)
    )
    
    return separation_stats


def explain_prediction(row, classifier, ranker, feature_names, clf_metadata, rank_metadata):
    """
    Explain a single prediction by showing top contributing features.
    
    Args:
        row: Row with company data
        classifier: Trained classifier
        ranker: Trained ranker
        feature_names: List of feature names
        clf_metadata: Classifier metadata
        rank_metadata: Ranker metadata
        
    Returns:
        dict: Explanation with top features and their contributions
    """
    from train_models import flatten_embedding
    
    # Extract features manually (same logic as InferenceModule)
    def extract_features_from_row(row, metadata):
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
    
    # Extract features
    X_clf = extract_features_from_row(row, clf_metadata)
    X_rank = extract_features_from_row(row, rank_metadata)
    
    # Get predictions
    clf_probs = classifier.predict_proba(X_clf)[0]
    rank_score = ranker.predict(X_rank)[0]
    
    # Get feature importance for this specific prediction
    # Use SHAP-like approach: compute contribution of each feature
    clf_contributions = compute_feature_contributions(X_clf[0], classifier, feature_names, 'classifier')
    rank_contributions = compute_feature_contributions(X_rank[0], ranker, feature_names, 'ranker')
    
    # Combine contributions (weighted by prediction confidence)
    combined_contributions = {}
    for feat in feature_names:
        if feat in clf_contributions and feat in rank_contributions:
            # Weight classifier contribution by prob_class_2
            # Weight ranker contribution by normalized rank score
            clf_weight = clf_probs[2]  # IC-worthy probability
            rank_weight = 0.5  # Equal weight for ranker
            combined_contributions[feat] = (
                clf_contributions[feat] * clf_weight + 
                rank_contributions[feat] * rank_weight
            )
    
    # Get top contributing features
    top_features = sorted(combined_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    explanation = {
        'predicted_class': int(classifier.predict(X_clf)[0]),
        'classifier_probs': {
            'irrelevant': float(clf_probs[0]),
            'relevant_pass': float(clf_probs[1]),
            'ic_worthy': float(clf_probs[2])
        },
        'ranker_score': float(rank_score),
        'top_contributing_features': [
            {
                'feature': feat,
                'contribution': float(contrib),
                'classifier_contrib': float(clf_contributions.get(feat, 0)),
                'ranker_contrib': float(rank_contributions.get(feat, 0))
            }
            for feat, contrib in top_features
        ]
    }
    
    return explanation


def compute_feature_contributions(feature_vector, model, feature_names, model_type='classifier'):
    """
    Compute approximate feature contributions using permutation importance.
    
    Args:
        feature_vector: Feature vector (1D array)
        model: Trained model
        feature_names: List of feature names
        model_type: 'classifier' or 'ranker'
        
    Returns:
        dict: Feature contributions
    """
    # Baseline prediction
    X_baseline = feature_vector.reshape(1, -1)
    
    if model_type == 'classifier':
        baseline_pred = model.predict_proba(X_baseline)[0][2]  # IC-worthy probability
    else:
        baseline_pred = model.predict(X_baseline)[0]
    
    contributions = {}
    
    # For each feature, compute impact of setting it to 0
    for i, feat_name in enumerate(feature_names):
        X_perturbed = feature_vector.copy()
        X_perturbed[i] = 0.0  # Set feature to 0
        
        X_perturbed = X_perturbed.reshape(1, -1)
        
        if model_type == 'classifier':
            perturbed_pred = model.predict_proba(X_perturbed)[0][2]
        else:
            perturbed_pred = model.predict(X_perturbed)[0]
        
        # Contribution is the difference
        contributions[feat_name] = baseline_pred - perturbed_pred
    
    return contributions


def analyze_misclassifications(results_df, has_labels=True):
    """
    Analyze misclassifications to understand model weaknesses.
    
    Args:
        results_df: DataFrame with predictions and true labels
        has_labels: Whether true labels are available
        
    Returns:
        dict: Misclassification analysis
    """
    if not has_labels or 'true_label' not in results_df.columns:
        return {}
    
    analysis = {
        'total_samples': len(results_df),
        'correct_predictions': 0,
        'incorrect_predictions': 0,
        'confusion_matrix': {},
        'false_positives_class_2': [],  # Predicted 2, actually 0 or 1
        'false_negatives_class_2': []    # Actually 2, predicted 0 or 1
    }
    
    for idx, row in results_df.iterrows():
        pred = row['predicted_class']
        true = row['true_label']
        
        if pred == true:
            analysis['correct_predictions'] += 1
        else:
            analysis['incorrect_predictions'] += 1
            
            # Track false positives for class 2 (IC-worthy)
            if pred == 2 and true != 2:
                analysis['false_positives_class_2'].append({
                    'company_name': row.get('company_name', 'Unknown'),
                    'predicted_class': int(pred),
                    'true_label': int(true),
                    'ic_worthy_prob': row.get('prob_class_2', 0),
                    'combined_score': row.get('combined_score', 0)
                })
            
            # Track false negatives for class 2
            if true == 2 and pred != 2:
                analysis['false_negatives_class_2'].append({
                    'company_name': row.get('company_name', 'Unknown'),
                    'predicted_class': int(pred),
                    'true_label': int(true),
                    'ic_worthy_prob': row.get('prob_class_2', 0),
                    'combined_score': row.get('combined_score', 0)
                })
        
        # Build confusion matrix
        key = f"pred_{int(pred)}_true_{int(true)}"
        analysis['confusion_matrix'][key] = analysis['confusion_matrix'].get(key, 0) + 1
    
    analysis['accuracy'] = analysis['correct_predictions'] / analysis['total_samples'] if analysis['total_samples'] > 0 else 0
    
    return analysis


def print_explainability_report(importance_df, separation_stats, misclassification_analysis, 
                               top_misclassified=None):
    """
    Print a comprehensive explainability report.
    
    Args:
        importance_df: DataFrame with feature importance
        separation_stats: Class separation statistics
        misclassification_analysis: Misclassification analysis
        top_misclassified: List of top misclassified companies with explanations
    """
    print("\n" + "=" * 80)
    print("MODEL EXPLAINABILITY REPORT")
    print("=" * 80)
    
    # Feature Importance
    print("\n📊 TOP 20 MOST IMPORTANT FEATURES (Combined Classifier + Ranker)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<40} {'Classifier %':<15} {'Ranker %':<15} {'Combined %':<15}")
    print("-" * 80)
    for idx, row in importance_df.head(20).iterrows():
        print(f"{idx+1:<6} {row['feature']:<40} {row['classifier_importance']:<15.2f} "
              f"{row['ranker_importance']:<15.2f} {row['combined_importance']:<15.2f}")
    
    # Class Separation Analysis
    if separation_stats and 'class_1_vs_2_separation' in separation_stats:
        print("\n📈 CLASS SEPARATION ANALYSIS (Class 1 vs Class 2)")
        print("-" * 80)
        print("Features that best separate Relevant-but-Pass (1) from IC-worthy (2):")
        print(f"{'Feature':<40} {'Mean (Class 1)':<15} {'Mean (Class 2)':<15} {'Separation':<15}")
        print("-" * 80)
        
        top_separation = list(separation_stats['class_1_vs_2_separation'].items())[:15]
        for feat, stats in top_separation:
            print(f"{feat:<40} {stats['mean_class_1']:<15.4f} {stats['mean_class_2']:<15.4f} "
                  f"{stats['separation_score']:<15.4f}")
        
        print("\n💡 Interpretation:")
        print("  - Higher separation score = better distinction between classes")
        print("  - Negative overlap = classes are well separated")
        print("  - Positive overlap = classes overlap significantly")
    
    # Misclassification Analysis
    if misclassification_analysis:
        print("\n⚠️  MISCLASSIFICATION ANALYSIS")
        print("-" * 80)
        print(f"Total samples: {misclassification_analysis['total_samples']}")
        print(f"Correct predictions: {misclassification_analysis['correct_predictions']} "
              f"({misclassification_analysis['correct_predictions']/misclassification_analysis['total_samples']*100:.1f}%)")
        print(f"Incorrect predictions: {misclassification_analysis['incorrect_predictions']} "
              f"({misclassification_analysis['incorrect_predictions']/misclassification_analysis['total_samples']*100:.1f}%)")
        
        if misclassification_analysis['false_positives_class_2']:
            print(f"\n❌ FALSE POSITIVES (Predicted IC-worthy, but actually not): {len(misclassification_analysis['false_positives_class_2'])}")
            print("   These companies scored high but shouldn't be IC-worthy:")
            for i, fp in enumerate(misclassification_analysis['false_positives_class_2'][:10], 1):
                print(f"   {i}. {fp['company_name']}")
                print(f"      Predicted: {fp['predicted_class']}, True: {fp['true_label']}")
                print(f"      IC-worthy prob: {fp['ic_worthy_prob']:.3f}, Combined score: {fp['combined_score']:.4f}")
        
        if misclassification_analysis['false_negatives_class_2']:
            print(f"\n⚠️  FALSE NEGATIVES (Actually IC-worthy, but predicted otherwise): {len(misclassification_analysis['false_negatives_class_2'])}")
            for i, fn in enumerate(misclassification_analysis['false_negatives_class_2'][:10], 1):
                print(f"   {i}. {fn['company_name']}")
                print(f"      Predicted: {fn['predicted_class']}, True: {fn['true_label']}")
                print(f"      IC-worthy prob: {fn['ic_worthy_prob']:.3f}, Combined score: {fn['combined_score']:.4f}")
    
    # Top Misclassified with Explanations
    if top_misclassified:
        print("\n🔍 DETAILED EXPLANATIONS FOR TOP MISCLASSIFIED COMPANIES")
        print("-" * 80)
        for i, item in enumerate(top_misclassified[:5], 1):
            company = item['company']
            explanation = item['explanation']
            print(f"\n{i}. {company.get('company_name', 'Unknown')}")
            print(f"   Predicted: {explanation['predicted_class']}, "
                  f"IC-worthy prob: {explanation['classifier_probs']['ic_worthy']:.3f}")
            print(f"   Top contributing features:")
            for feat_info in explanation['top_contributing_features'][:5]:
                print(f"     - {feat_info['feature']}: {feat_info['contribution']:.4f} "
                      f"(clf: {feat_info['classifier_contrib']:.4f}, "
                      f"rank: {feat_info['ranker_contrib']:.4f})")
    
    print("\n" + "=" * 80)

