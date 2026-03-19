"""
Test Models on Unseen Data

This script loads trained models and evaluates them on new data that wasn't
used during training. It can:
- Load test data from parquet files, database, or CSV
- Generate predictions using the InferenceModule
- Evaluate performance if labels are available
- Save predictions and evaluation results
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Add parent directory and tests directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tests_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, tests_dir)

# Import from train_models
from train_models import InferenceModule, evaluate_ranking_per_group, compute_combined_score

# Import from feature_preparation
from feature_preparation import (
    combine_text_fields, format_experiences_json, convert_boolean_field,
    get_text_fields, get_text_fields_group1, get_text_fields_group2,
    get_boolean_fields, prepare_numeric_features, load_scaler,
    convert_building_since_to_years
)

# Import explainability module
from explainability import (
    get_feature_importance, analyze_class_separation, explain_prediction,
    analyze_misclassifications, print_explainability_report
)

# Initialize OpenAI client for embeddings
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
openai_client = OpenAI(api_key=openai_api_key)

# Output directory
TEST_OUTPUT_DIR = 'data/test_predictions'
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# Scaler path (must match build_dataset.py)
SCALER_PATH = 'data/models/robust_scaler.pkl'


def load_test_data_from_parquet(filepath):
    """
    Load test data from a parquet file.
    
    The file should have the same structure as classification_dataset.parquet:
    - text_embedding, experience_embedding
    - Scaled numeric features
    - Boolean features
    - Optional: label (for evaluation)
    
    Args:
        filepath: Path to parquet file
        
    Returns:
        pandas.DataFrame: Test data
    """
    print(f"Loading test data from {filepath}...")
    df = pd.read_parquet(filepath, engine='pyarrow')
    print(f"Loaded {len(df)} rows")
    return df


def load_test_data_from_database(query=None, exclude_training_ids=None):
    """
    Load test data from the database.
    
    Filters for:
    - history = '' (empty) - this already excludes companies in Notion databases
    - tree_result = 'Strong recommend' (has a strong recommendation)
    
    Note: We don't need to check Notion databases because companies with history = ''
    are already not in the pipeline/tracked/passed databases.
    
    Args:
        query: Custom SQL query (if None, uses default)
        exclude_training_ids: List of IDs to exclude (from training set)
        
    Returns:
        pandas.DataFrame: Test data
    """
    from services.database import get_db_connection
    
    print("Loading test data from database...")
    print("  Filter: history = '' AND tree_result = 'Strong recommend'")
    
    if query is None:
        # Query for companies with history = '' and tree_result != ''
        query = """
            SELECT 
                id,
                profile_url,
                company_name,
                -- Text fields
                about,
                product,
                market,
                post_data,
                embeddednews,
                location,
                funding,
                company_tags,
                school_tags,
                tree_path,
                tree_thesis,
                -- Numeric fields
                years_of_experience,
                industry_expertise_score,
                building_since,
                company_tech_score,
                lastroundvaluation,
                latestdealamount,
                headcount,
                -- Boolean fields
                technical,
                repeat_founder,
                -- JSON field
                all_experiences,
                -- Labels (if available)
                tree_result,
                history
            FROM founders
            WHERE product != '' 
              AND market != ''
              AND history = ''
              AND tree_result IS NOT NULL
              AND tree_result = 'Track' 
        """
        
        if exclude_training_ids:
            id_list = ','.join([str(id) for id in exclude_training_ids])
            query += f" AND id NOT IN ({id_list})"
    
    conn = get_db_connection()
    if not conn:
        raise ConnectionError("Could not connect to database")
    
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        df = pd.DataFrame(rows, columns=column_names)
        print(f"  Loaded {len(df)} rows matching criteria")
        print("  Note: Companies with history = '' are already not in Notion databases")
        
        return df
        
    except Exception as e:
        print(f"Error loading data from database: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def embed_text_batch(texts, model="text-embedding-3-small", batch_size=100):
    """
    Embed a batch of texts using OpenAI API.
    Same as in build_dataset.py
    """
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        cleaned_batch = []
        for t in batch:
            if pd.isna(t) or t is None:
                cleaned_text = "empty"
            else:
                try:
                    cleaned_text = str(t).strip()
                    if not cleaned_text or cleaned_text.lower() in ['nan', 'none', '']:
                        cleaned_text = "empty"
                    if len(cleaned_text) > 32000:
                        cleaned_text = cleaned_text[:32000]
                except Exception:
                    cleaned_text = "empty"
            cleaned_batch.append(cleaned_text)
        
        try:
            response = openai_client.embeddings.create(
                model=model,
                input=cleaned_batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"    Error embedding batch {i//batch_size + 1}: {e}")
            embedding_dim = 1536
            embeddings.extend([[0.0] * embedding_dim] * len(cleaned_batch))
    
    return embeddings




def load_embeddings_from_db(profile_urls_or_ids, id_column='profile_url', verbose=True):
    """
    Load embeddings from database for given profile URLs or IDs.
    
    Args:
        profile_urls_or_ids: List of profile URLs or IDs to load embeddings for
        id_column: Column name to match on ('profile_url' or 'id')
        verbose: If True, print warnings
        
    Returns:
        dict: Dictionary mapping profile_url/id to embedding dict
    """
    from services.database import get_db_connection
    import json
    
    if not profile_urls_or_ids:
        return {}
    
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        
        # Build query to fetch embeddings
        if id_column == 'profile_url':
            placeholders = ','.join(['%s'] * len(profile_urls_or_ids))
            query = f"""
                SELECT profile_url, text_embedding_group1, text_embedding_group2, experience_embedding
                FROM founders
                WHERE profile_url IN ({placeholders})
            """
        else:  # id column
            placeholders = ','.join(['%s'] * len(profile_urls_or_ids))
            query = f"""
                SELECT id, text_embedding_group1, text_embedding_group2, experience_embedding
                FROM founders
                WHERE id IN ({placeholders})
            """
        
        cursor.execute(query, list(profile_urls_or_ids))
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        embeddings_dict = {}
        for row in rows:
            row_dict = dict(zip(column_names, row))
            key = row_dict.get(id_column)
            if key:
                # Parse JSON embeddings if they're stored as strings
                emb_dict = {}
                for emb_col in ['text_embedding_group1', 'text_embedding_group2', 'experience_embedding']:
                    val = row_dict.get(emb_col)
                    if val is not None:
                        if isinstance(val, str):
                            try:
                                emb_dict[emb_col] = json.loads(val)
                            except (json.JSONDecodeError, TypeError):
                                emb_dict[emb_col] = None
                        elif isinstance(val, (list, np.ndarray)):
                            emb_dict[emb_col] = list(val)
                        else:
                            emb_dict[emb_col] = val
                    else:
                        emb_dict[emb_col] = None
                embeddings_dict[key] = emb_dict
        
        return embeddings_dict
        
    except Exception as e:
        if verbose:
            print(f"  Warning: Error loading embeddings from database: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def save_embeddings_to_db(embeddings_data, id_column='profile_url', verbose=True):
    """
    Save embeddings to database.
    
    Args:
        embeddings_data: List of dicts with keys: id_column, text_embedding_group1, 
                        text_embedding_group2, experience_embedding
        id_column: Column name to match on ('profile_url' or 'id')
        verbose: If True, print warnings
        
    Returns:
        int: Number of profiles updated
    """
    from services.database import get_db_connection
    import json
    
    if not embeddings_data:
        return 0
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    updated_count = 0
    try:
        cursor = conn.cursor()
        
        for emb_data in embeddings_data:
            id_value = emb_data.get(id_column)
            if not id_value:
                continue
            
            # Prepare update data
            update_fields = []
            update_values = []
            
            for emb_col in ['text_embedding_group1', 'text_embedding_group2', 'experience_embedding']:
                emb_value = emb_data.get(emb_col)
                if emb_value is not None:
                    # Convert to JSON string for storage
                    if isinstance(emb_value, (list, np.ndarray)):
                        emb_json = json.dumps(list(emb_value))
                    elif isinstance(emb_value, str):
                        emb_json = emb_value  # Already a string
                    else:
                        emb_json = json.dumps(emb_value)
                    
                    update_fields.append(f'"{emb_col}" = %s')
                    update_values.append(emb_json)
            
            if not update_fields:
                continue
            
            # Build update query
            if id_column == 'profile_url':
                query = f"""
                    UPDATE founders
                    SET {', '.join(update_fields)}
                    WHERE profile_url = %s
                """
            else:  # id column
                query = f"""
                    UPDATE founders
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                """
            
            update_values.append(id_value)
            cursor.execute(query, update_values)
            if cursor.rowcount > 0:
                updated_count += 1
        
        conn.commit()
        return updated_count
        
    except Exception as e:
        conn.rollback()
        if verbose:
            print(f"  Warning: Error saving embeddings to database: {e}")
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def prepare_test_features(df, verbose=True, save_embeddings=True):
    """
    Prepare test data features from raw database data.
    
    Constructs all required features:
    - Text embeddings (from combined text fields) - loads from DB if available, creates if missing
    - Experience embeddings (from all_experiences JSON) - loads from DB if available, creates if missing
    - Scaled numeric features - computed on the fly (not saved to DB)
    - Boolean features - computed on the fly (not saved to DB)
    
    Note: Embeddings are now loaded from the database if available, and saved back after creation.
    This avoids recomputing embeddings on every run.
    
    Args:
        df: DataFrame with raw company data from database
        verbose: If True, print progress messages. If False, suppress output.
        save_embeddings: If True, save newly created embeddings to database (default: True)
        
    Returns:
        pandas.DataFrame: DataFrame with all features ready for inference
    """
    if verbose:
        print("Preparing test features from raw data...")
    df = df.copy()

    # --- CRITICAL: force embedding columns to object dtype ---
    for col in [
        'text_embedding_group1',
        'text_embedding_group2',
        'experience_embedding'
    ]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype(object)
    
    # Check if features are already present in the DataFrame
    has_text_emb1 = (
        'text_embedding_group1' in df.columns and
        df['text_embedding_group1'].notna().any()
    )

    has_text_emb2 = (
        'text_embedding_group2' in df.columns and
        df['text_embedding_group2'].notna().any()
    )

    has_exp_emb = (
        'experience_embedding' in df.columns and
        df['experience_embedding'].notna().any()
    )

    has_scaled = any(col.endswith('_scaled') for col in df.columns)

    if has_text_emb1 and has_text_emb2 and has_exp_emb and has_scaled:
        if verbose:
            print("  Features already prepared")
        return df

    # Try to load embeddings from database
    embeddings_to_save = []
    id_column = 'profile_url' if 'profile_url' in df.columns else 'id'
    
    if id_column in df.columns:
        if verbose:
            print("  Checking database for existing embeddings...")
        ids_to_check = df[id_column].dropna().unique().tolist()
        existing_embeddings = load_embeddings_from_db(ids_to_check, id_column=id_column, verbose=verbose)
        
        if existing_embeddings:
            if verbose:
                print(f"    Found embeddings for {len(existing_embeddings)} profiles in database")
            
            # Load embeddings into DataFrame
            for idx, row in df.iterrows():
                key = row.get(id_column)
                if key and key in existing_embeddings:
                    emb_data = existing_embeddings[key]
                    if emb_data.get('text_embedding_group1') is not None:
                        df.at[idx, 'text_embedding_group1'] = emb_data['text_embedding_group1']
                    if emb_data.get('text_embedding_group2') is not None:
                        df.at[idx, 'text_embedding_group2'] = emb_data['text_embedding_group2']
                    if emb_data.get('experience_embedding') is not None:
                        df.at[idx, 'experience_embedding'] = emb_data['experience_embedding']
    
    # Check again which embeddings are still missing
    has_text_emb1 = 'text_embedding_group1' in df.columns and df['text_embedding_group1'].notna().any()
    has_text_emb2 = 'text_embedding_group2' in df.columns and df['text_embedding_group2'].notna().any()
    has_exp_emb = 'experience_embedding' in df.columns and df['experience_embedding'].notna().any()
    
    # 1. Text Embeddings - Two separate groups
    if not has_text_emb1 or not has_text_emb2:
        if verbose:
            print("  Creating text embeddings (two groups)...")
        
        # Group 1: Company/Founder info
        if not has_text_emb1:
            if verbose:
                print("    Group 1: Company/Founder info...")
            text_fields_group1 = get_text_fields_group1()
            combined_texts_group1 = []
            rows_needing_emb1 = []
            for idx, row in df.iterrows():
                # Only create embedding if missing
                if pd.isna(row.get('text_embedding_group1')):
                    combined_text = combine_text_fields(row, text_fields_group1)
                    combined_texts_group1.append(combined_text)
                    rows_needing_emb1.append(idx)
                else:
                    combined_texts_group1.append(None)  # Placeholder
            
            if combined_texts_group1 and any(t is not None for t in combined_texts_group1):
                texts_to_embed = [t for t in combined_texts_group1 if t is not None]
                if verbose:
                    print(f"      Embedding {len(texts_to_embed)} group 1 texts...")
                text_embeddings_group1 = embed_text_batch(texts_to_embed)
                
                # Assign embeddings to rows that needed them
                emb_idx = 0
                for idx in rows_needing_emb1:
                    df.at[idx, 'text_embedding_group1'] = text_embeddings_group1[emb_idx]
                    # Track for saving to DB
                    if id_column in df.columns:
                        key = df.at[idx, id_column]
                        if key:
                            # Find or create entry in embeddings_to_save
                            emb_entry = next((e for e in embeddings_to_save if e.get(id_column) == key), None)
                            if not emb_entry:
                                emb_entry = {id_column: key}
                                embeddings_to_save.append(emb_entry)
                            emb_entry['text_embedding_group1'] = text_embeddings_group1[emb_idx]
                    emb_idx += 1
                
                if verbose:
                    print(f"      Created group 1 embeddings (dimension: {len(text_embeddings_group1[0]) if text_embeddings_group1 else 0})")
        else:
            if verbose:
                print("    Group 1 embeddings already present")
        
        # Group 2: Product/Market info
        if not has_text_emb2:
            if verbose:
                print("    Group 2: Product/Market info...")
            text_fields_group2 = get_text_fields_group2()
            combined_texts_group2 = []
            rows_needing_emb2 = []
            for idx, row in df.iterrows():
                # Only create embedding if missing
                if pd.isna(row.get('text_embedding_group2')):
                    combined_text = combine_text_fields(row, text_fields_group2)
                    combined_texts_group2.append(combined_text)
                    rows_needing_emb2.append(idx)
                else:
                    combined_texts_group2.append(None)  # Placeholder
            
            if combined_texts_group2 and any(t is not None for t in combined_texts_group2):
                texts_to_embed = [t for t in combined_texts_group2 if t is not None]
                if verbose:
                    print(f"      Embedding {len(texts_to_embed)} group 2 texts...")
                text_embeddings_group2 = embed_text_batch(texts_to_embed)
                
                # Assign embeddings to rows that needed them
                emb_idx = 0
                for idx in rows_needing_emb2:
                    df.at[idx, 'text_embedding_group2'] = text_embeddings_group2[emb_idx]
                    # Track for saving to DB
                    if id_column in df.columns:
                        key = df.at[idx, id_column]
                        if key:
                            # Find or create entry in embeddings_to_save
                            emb_entry = next((e for e in embeddings_to_save if e.get(id_column) == key), None)
                            if not emb_entry:
                                emb_entry = {id_column: key}
                                embeddings_to_save.append(emb_entry)
                            emb_entry['text_embedding_group2'] = text_embeddings_group2[emb_idx]
                    emb_idx += 1
                
                if verbose:
                    print(f"      Created group 2 embeddings (dimension: {len(text_embeddings_group2[0]) if text_embeddings_group2 else 0})")
        else:
            if verbose:
                print("    Group 2 embeddings already present")
    else:
        if verbose:
            print("  Text embeddings already present")
    
    # 2. Experience Embeddings
    if not has_exp_emb:
        if verbose:
            print("  Creating experience embeddings...")
        experience_texts = []
        rows_needing_exp_emb = []
        for idx, row in df.iterrows():
            # Only create embedding if missing
            if pd.isna(row.get('experience_embedding')):
                experiences = row.get('all_experiences', '')
                formatted = format_experiences_json(experiences)
                experience_texts.append(formatted)
                rows_needing_exp_emb.append(idx)
            else:
                experience_texts.append(None)  # Placeholder
        
        if experience_texts and any(t is not None for t in experience_texts):
            texts_to_embed = [t for t in experience_texts if t is not None]
            if verbose:
                print(f"    Embedding {len(texts_to_embed)} experience texts...")
            experience_embeddings = embed_text_batch(texts_to_embed)
            
            # Assign embeddings to rows that needed them
            emb_idx = 0
            for idx in rows_needing_exp_emb:
                df.at[idx, 'experience_embedding'] = experience_embeddings[emb_idx]
                # Track for saving to DB
                if id_column in df.columns:
                    key = df.at[idx, id_column]
                    if key:
                        # Find or create entry in embeddings_to_save
                        emb_entry = next((e for e in embeddings_to_save if e.get(id_column) == key), None)
                        if not emb_entry:
                            emb_entry = {id_column: key}
                            embeddings_to_save.append(emb_entry)
                        emb_entry['experience_embedding'] = experience_embeddings[emb_idx]
                emb_idx += 1
            
            if verbose:
                print(f"    Created experience embeddings (dimension: {len(experience_embeddings[0]) if experience_embeddings else 0})")
    else:
        if verbose:
            print("  Experience embeddings already present")
    
    # Save newly created embeddings to database
    if save_embeddings and embeddings_to_save:
        if verbose:
            print(f"  Saving {len(embeddings_to_save)} new embeddings to database...")
        saved_count = save_embeddings_to_db(embeddings_to_save, id_column=id_column, verbose=verbose)
        if verbose:
            print(f"    Saved embeddings for {saved_count} profiles")
    
    # 3. Numeric Standardization
    if not has_scaled:
        if verbose:
            print("  Standardizing numeric features...")
        # Load the scaler from training (CRITICAL: must use same scaler!)
        import contextlib
        import io
        try:
            if verbose:
                scaler = load_scaler(SCALER_PATH)
            else:
                # Suppress output from load_scaler
                with contextlib.redirect_stdout(io.StringIO()):
                    scaler = load_scaler(SCALER_PATH)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Scaler not found at {SCALER_PATH}. "
                "Please run build_dataset.py first to train and save the scaler."
            )
        
        # Use shared function with loaded scaler (fit_scaler=False for inference)
        # prepare_numeric_features has its own verbose output, so we need to suppress it
        if verbose:
            df, _ = prepare_numeric_features(df, scaler=scaler, fit_scaler=False)
        else:
            # Suppress output from prepare_numeric_features
            with contextlib.redirect_stdout(io.StringIO()):
                df, _ = prepare_numeric_features(df, scaler=scaler, fit_scaler=False)
    else:
        if verbose:
            print("  Scaled numeric features already present")
    
    # 5. Boolean Features
    if verbose:
        print("  Converting boolean features...")
    boolean_fields = get_boolean_fields()
    
    for field in boolean_fields:
        if field in df.columns:
            df[field] = df[field].apply(convert_boolean_field)
        else:
            df[field] = 0
    
    if verbose:
        print("  Feature preparation complete!")
    return df


def evaluate_classification(y_true, predictions):
    """
    Evaluate classification predictions.
    
    Args:
        y_true: True labels
        predictions: List of prediction dicts from InferenceModule
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = np.array([p['predicted_class'] for p in predictions])
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics, y_pred


def test_models(test_data_path=None, test_data_df=None, database_query=None,
                exclude_training_ids=None,
                model_dir='data/models', output_dir=TEST_OUTPUT_DIR, evaluate=True,
                limit=100):
    """
    Test trained models on unseen data.
    
    Args:
        test_data_path: Path to parquet file with test data
        test_data_df: DataFrame with test data (alternative to file path)
        database_query: SQL query to load test data from database
        exclude_training_ids: List of IDs to exclude from database query
        model_dir: Directory containing trained models
        output_dir: Directory to save predictions
        evaluate: Whether to evaluate performance (requires labels in test data)
        
    Returns:
        dict: Results dictionary with predictions and metrics
    """
    print("=" * 80)
    print("Testing Models on Unseen Data")
    print("=" * 80)
    print()
    
    # Load test data
    if test_data_df is not None:
        test_df = test_data_df.copy()
    elif test_data_path:
        test_df = load_test_data_from_parquet(test_data_path)
    elif database_query:
        test_df = load_test_data_from_database(database_query, exclude_training_ids)
    else:
        # Default: load from database with specific criteria
        test_df = load_test_data_from_database(None, exclude_training_ids)
    
    # Limit to first N companies
    if limit and len(test_df) > limit:
        print(f"\nLimiting to first {limit} companies (out of {len(test_df)} total)")
        test_df = test_df.head(limit).copy()
    
    # Prepare features (only for the limited set, not entire database)
    print(f"\nNote: Generating embeddings on the fly for {min(limit, len(test_df)) if limit else len(test_df)} companies (not entire database, not saved to DB)")
    test_df = prepare_test_features(test_df)
    
    # Check if labels are available for evaluation
    has_labels = 'label' in test_df.columns
    if evaluate and not has_labels:
        print("WARNING: No 'label' column found. Cannot evaluate classification performance.")
        print("Will generate predictions only.")
        evaluate = False
    
    # Load inference module
    print(f"\nLoading models from {model_dir}...")
    inference = InferenceModule(model_dir=model_dir)
    
    # Get feature names from metadata for explainability
    feature_names = inference.clf_metadata.get('feature_names', [])
    
    # Generate predictions
    print(f"\nGenerating predictions for {len(test_df)} samples...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        try:
            # Set gate_threshold to 1.0 to effectively disable the gate (probabilities are <= 1.0)
            result = inference.predict_combined(row, gate_threshold=1.0)
            predictions.append(result)
        except Exception as e:
            print(f"  Error predicting for row {idx}: {e}")
            predictions.append({
                'classifier_probs': [0.33, 0.33, 0.34],
                'ranker_score': 0.0,
                'combined_score': 0.0,
                'predicted_class': -1
            })
    
    # Create results DataFrame
    results_df = test_df[['id', 'company_name', 'profile_url']].copy() if 'id' in test_df.columns else test_df[['company_name']].copy()
    
    results_df['prob_class_0'] = [p['classifier_probs'][0] for p in predictions]
    results_df['prob_class_1'] = [p['classifier_probs'][1] for p in predictions]
    results_df['prob_class_2'] = [p['classifier_probs'][2] for p in predictions]
    results_df['predicted_class'] = [p['predicted_class'] for p in predictions]
    results_df['ranker_score'] = [p['ranker_score'] for p in predictions]
    results_df['combined_score'] = [p['combined_score'] for p in predictions]
    
    if has_labels:
        results_df['true_label'] = test_df['label'].values
    
    # Sort by combined score (descending)
    results_df = results_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    # Save predictions
    output_path = os.path.join(output_dir, 'test_predictions.parquet')
    results_df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved predictions to {output_path}")
    
    # ============================================================================
    # EXPLAINABILITY ANALYSIS
    # ============================================================================
    print("\n" + "=" * 80)
    print("Running Explainability Analysis...")
    print("=" * 80)
    
    # 1. Feature Importance Analysis
    print("\nComputing feature importance...")
    try:
        importance_df = get_feature_importance(
            inference.classifier,
            inference.ranker,
            feature_names
        )
    except Exception as e:
        print(f"  Warning: Could not compute feature importance: {e}")
        importance_df = None
    
    # 2. Class Separation Analysis
    print("Analyzing class separation...")
    try:
        # Prepare test_df with labels for separation analysis
        separation_df = test_df.copy()
        if 'label' in separation_df.columns:
            # Get numeric feature columns
            numeric_cols = [col for col in separation_df.columns 
                          if col.endswith('_scaled') or col in ['technical', 'repeat_founder']]
            separation_stats = analyze_class_separation(
                separation_df,
                label_col='label',
                feature_cols=numeric_cols
            )
        else:
            separation_stats = {}
    except Exception as e:
        print(f"  Warning: Could not analyze class separation: {e}")
        separation_stats = {}
    
    # 3. Misclassification Analysis
    print("Analyzing misclassifications...")
    misclassification_analysis = {}
    if has_labels:
        try:
            misclassification_analysis = analyze_misclassifications(results_df, has_labels=True)
        except Exception as e:
            print(f"  Warning: Could not analyze misclassifications: {e}")
    
    # 4. Detailed Explanations for High-Scoring False Positives
    print("Generating detailed explanations for high-scoring companies...")
    top_misclassified = []
    
    # Get high-scoring IC-worthy predictions that might be wrong
    high_scoring_ic = results_df[
        (results_df['predicted_class'] == 2) & 
        (results_df['combined_score'] > 0.5)
    ].head(10)
    
    for idx, row in high_scoring_ic.iterrows():
        try:
            # Get the original row from test_df
            original_row = test_df[test_df['id'] == row.get('id')].iloc[0] if 'id' in row else None
            if original_row is None:
                continue
            
            explanation = explain_prediction(
                original_row,
                inference.classifier,
                inference.ranker,
                feature_names,
                inference.clf_metadata,
                inference.rank_metadata
            )
            
            top_misclassified.append({
                'company': row.to_dict(),
                'explanation': explanation
            })
        except Exception as e:
            print(f"  Warning: Could not explain prediction for row {idx}: {e}")
            continue
    
    # Print explainability report
    if importance_df is not None or separation_stats or misclassification_analysis:
        print_explainability_report(
            importance_df,
            separation_stats,
            misclassification_analysis,
            top_misclassified
        )
    
    # Save explainability results
    explainability_path = os.path.join(output_dir, 'explainability_report.json')
    explainability_data = {
        'feature_importance': importance_df.to_dict('records') if importance_df is not None else None,
        'class_separation': separation_stats,
        'misclassification_analysis': misclassification_analysis,
        'top_explanations': [
            {
                'company_name': item['company'].get('company_name', 'Unknown'),
                'explanation': item['explanation']
            }
            for item in top_misclassified
        ]
    }
    
    with open(explainability_path, 'w') as f:
        json.dump(explainability_data, f, indent=2, default=str)
    print(f"\nSaved explainability report to {explainability_path}")
    
    # Evaluate if labels are available
    metrics = None
    if evaluate and has_labels:
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        
        y_true = test_df['label'].values
        clf_metrics, y_pred = evaluate_classification(y_true, predictions)
        
        print(f"\nClassification Metrics:")
        print(f"  Accuracy: {clf_metrics['accuracy']:.4f}")
        print(f"  Precision: {clf_metrics['precision']:.4f}")
        print(f"  Recall: {clf_metrics['recall']:.4f}")
        print(f"  F1-Score: {clf_metrics['f1_score']:.4f}")
        print(f"\nConfusion Matrix:")
        print(np.array(clf_metrics['confusion_matrix']))
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Irrelevant (0)', 'Relevant-but-Pass (1)', 'IC-worthy (2)']))
        
        # Save evaluation report
        report_path = os.path.join(output_dir, 'test_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write("Test Set Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Test samples: {len(test_df)}\n\n")
            f.write("Classification Metrics:\n")
            f.write(f"  Accuracy: {clf_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {clf_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {clf_metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {clf_metrics['f1_score']:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(np.array(clf_metrics['confusion_matrix'])) + "\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_true, y_pred,
                                          target_names=['Irrelevant (0)', 'Relevant-but-Pass (1)', 'IC-worthy (2)']))
        
        print(f"\nSaved evaluation report to {report_path}")
        metrics = clf_metrics
    
    # Filter to only show IC-worthy (2) companies
    ic_worthy_df = results_df[results_df['predicted_class'] == 2].copy()
    
    # Print only IC-worthy predictions
    print("\n" + "=" * 80)
    print(f"IC-worthy Companies (Predicted Class 2)")
    print(f"Showing {len(ic_worthy_df)} out of {len(results_df)} companies")
    print("=" * 80)
    
    class_names = {0: "(Irrelevant)", 1: "(Relevant-but-Pass)", 2: "(IC-worthy)"}
    
    for idx, row in ic_worthy_df.iterrows():
        print(f"\n{idx + 1}. {row['company_name']}")
        if 'profile_url' in row and pd.notna(row['profile_url']):
            print(f"   Profile URL: {row['profile_url']}")
        print(f"   Combined Score: {row['combined_score']:.4f}")
        print(f"   Ranker Score: {row['ranker_score']:.4f}")
        print(f"   Predicted Class: {row['predicted_class']} {class_names.get(row['predicted_class'], '')}")
        print(f"   Classifier Probabilities:")
        print(f"     - Irrelevant (0): {row['prob_class_0']:.3f}")
        print(f"     - Relevant-but-Pass (1): {row['prob_class_1']:.3f}")
        print(f"     - IC-worthy (2): {row['prob_class_2']:.3f}")
        if has_labels:
            print(f"   True Label: {row['true_label']} {class_names.get(row['true_label'], '')}")
            if row['predicted_class'] == row['true_label']:
                print(f"   ✓ Correct prediction!")
            else:
                print(f"   ✗ Prediction mismatch")
    
    # Print ratios
    print("\n" + "=" * 80)
    print("Prediction Ratios")
    print("=" * 80)
    
    total = len(results_df)
    irrelevant_count = len(results_df[results_df['predicted_class'] == 0])
    relevant_pass_count = len(results_df[results_df['predicted_class'] == 1])
    ic_worthy_count = len(results_df[results_df['predicted_class'] == 2])
    
    print(f"\nTotal companies processed: {total}")
    print(f"\nPredicted Class Distribution:")
    print(f"  Irrelevant (0): {irrelevant_count} ({irrelevant_count/total*100:.1f}%)")
    print(f"  Relevant-but-Pass (1): {relevant_pass_count} ({relevant_pass_count/total*100:.1f}%)")
    print(f"  IC-worthy (2): {ic_worthy_count} ({ic_worthy_count/total*100:.1f}%)")
    print(f"\nRelevant (1 + 2): {relevant_pass_count + ic_worthy_count} ({(relevant_pass_count + ic_worthy_count)/total*100:.1f}%)")
    
    if has_labels:
        print(f"\nTrue Label Distribution:")
        true_irrelevant = len(results_df[results_df['true_label'] == 0]) if 'true_label' in results_df.columns else 0
        true_relevant_pass = len(results_df[results_df['true_label'] == 1]) if 'true_label' in results_df.columns else 0
        true_ic_worthy = len(results_df[results_df['true_label'] == 2]) if 'true_label' in results_df.columns else 0
        
        print(f"  Irrelevant (0): {true_irrelevant} ({true_irrelevant/total*100:.1f}%)")
        print(f"  Relevant-but-Pass (1): {true_relevant_pass} ({true_relevant_pass/total*100:.1f}%)")
        print(f"  IC-worthy (2): {true_ic_worthy} ({true_ic_worthy/total*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
    
    return {
        'predictions': results_df,
        'metrics': metrics,
        'n_samples': len(test_df)
    }


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(description='Test trained models on unseen data')
    parser.add_argument('--test-data', type=str, help='Path to parquet file with test data')
    parser.add_argument('--from-database', action='store_true',
                       help='Load test data from database')
    parser.add_argument('--exclude-training-ids', type=str,
                       help='Path to parquet file with training IDs to exclude (e.g., classification_dataset.parquet)')
    parser.add_argument('--model-dir', type=str, default='data/models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default=TEST_OUTPUT_DIR,
                       help='Directory to save predictions')
    parser.add_argument('--no-evaluate', action='store_true',
                       help='Skip evaluation (useful when labels are not available)')
    parser.add_argument('--limit', type=int, default=100,
                       help='Limit number of companies to process (default: 100)')
    
    args = parser.parse_args()
    
    exclude_ids = None
    if args.exclude_training_ids:
        print(f"Loading training IDs to exclude from {args.exclude_training_ids}...")
        train_df = pd.read_parquet(args.exclude_training_ids, engine='pyarrow')
        exclude_ids = train_df['id'].tolist() if 'id' in train_df.columns else None
        print(f"Excluding {len(exclude_ids)} training IDs")
    
    if args.from_database:
        test_models(
            database_query=None,
            exclude_training_ids=exclude_ids,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            evaluate=not args.no_evaluate,
            limit=args.limit
        )
    elif args.test_data:
        test_models(
            test_data_path=args.test_data,
            exclude_training_ids=exclude_ids,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            evaluate=not args.no_evaluate,
            limit=args.limit
        )
    else:
        # Default: load from database with specific criteria
        print("No data source specified. Loading from database with default criteria:")
        print("  - history = ''")
        print("  - tree_result = 'Strong recommend'")
        test_models(
            database_query=None,
            exclude_training_ids=exclude_ids,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            evaluate=not args.no_evaluate,
            limit=args.limit
        )


if __name__ == "__main__":
    main()

