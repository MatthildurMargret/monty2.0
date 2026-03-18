"""
Model Loader - Shared utility for loading the ranker-only XGBoost model.

Extracted from the duplicate ~50-line loading blocks in weekly_update.py and
recommendations.py to ensure a single, consistent loading path.
"""

import json
import os


def load_ranker_model():
    """Load the ranker-only XGBoost model and its metadata.

    Returns:
        dict with keys: ranker, metadata, ranker_min, ranker_max
        or None on any failure (xgboost missing, files absent, load error).
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("❌ xgboost not available, will fall back to score-based sorting")
        return None

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    print(f"\nLoading ranker-only model from {model_dir}...")

    try:
        ranker_path = os.path.join(model_dir, 'ranker_only_xgb.json')
        ranker_metadata_path = os.path.join(model_dir, 'ranker_only_feature_metadata.json')
        ranker_model_metadata_path = os.path.join(model_dir, 'ranker_only_metadata.json')

        if not os.path.exists(ranker_path):
            print(f"❌ Ranker-only model not found at {ranker_path}")
            return None

        with open(ranker_metadata_path, 'r') as f:
            ranker_metadata = json.load(f)
        with open(ranker_model_metadata_path, 'r') as f:
            ranker_model_metadata = json.load(f)

        ranker = xgb.XGBRanker()
        ranker.load_model(ranker_path)

        norm = ranker_model_metadata.get('ranker_normalization', {})
        ranker_inference = {
            'ranker': ranker,
            'metadata': ranker_metadata,
            'ranker_min': norm.get('min'),
            'ranker_max': norm.get('max'),
        }

        print("✅ Ranker-only model loaded successfully")
        print(f"  Ranker normalization: min={norm.get('min')}, max={norm.get('max')}")
        return ranker_inference

    except Exception as e:
        print(f"❌ Error loading ranker-only model: {e}")
        return None
