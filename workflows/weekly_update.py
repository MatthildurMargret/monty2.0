import json
import sys
import os
import numpy as np
import warnings
from services.tree_tools import find_pipeline_companies, find_nodes_by_name
from services.database import get_db_connection
import pandas as pd
from datetime import datetime, timedelta
from workflows.recommendations import mark_prev_recs
from services.path_mapper import get_all_matching_old_paths
from workflows.profile_recommendations import get_profile_recommendations, mark_profiles_as_recommended, format_profiles_for_weekly_update
from services.deal_processing import normalize_company_name


# Add tests directory to path to import model inference
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))

def get_recent_deals():
    deals_file = 'data/deal_data/early_deals.csv'
    
    # Handle missing file gracefully
    if not os.path.exists(deals_file):
        print(f"⚠️  Deals file not found at {deals_file}, returning empty DataFrame")
        return pd.DataFrame()
    
    try:
        deals = pd.read_csv(deals_file, quotechar='"')
    except Exception as e:
        print(f"⚠️  Error reading deals file: {e}, returning empty DataFrame")
        return pd.DataFrame()
    
    if deals.empty:
        return pd.DataFrame()
    
    deals["Investors"] = deals["Investors"].astype(str)
    deals["Investors"] = deals["Investors"].fillna("")
    deals["Vertical"] = deals["Vertical"].fillna("")
    deals["Funding Round"] = deals["Funding Round"].fillna("")
    deals["Category"] = deals["Category"].fillna("Other")
    
    # Filter by date first (past week) - more efficient to filter early
    current_date = datetime.now().strftime("%B %d, %Y")  # e.g., "November 16, 2024"
    past_week = datetime.now() - timedelta(days=6)
    deals["Date"] = pd.to_datetime(deals["Date"], errors="coerce")
    initial_count = len(deals)
    deals = deals[deals["Date"] > past_week].copy()
    after_date_count = len(deals)
    
    # Filter out Series A companies
    if "Funding Round" in deals.columns and len(deals) > 0:
        # Normalize funding round values for comparison (handle variations like "Series A", "SeriesA", etc.)
        deals["Funding Round Lower"] = deals["Funding Round"].astype(str).str.lower().str.strip()
        # Filter out Series A (case-insensitive matching, handles "series a", "seriesa", etc.)
        before_series_a_filter = len(deals)
        series_a_mask = deals["Funding Round Lower"].str.contains(r"series\s*a", na=False, regex=True)
        deals = deals[~series_a_mask].copy()
        deals = deals.drop(columns=["Funding Round Lower"])
        after_series_a_count = len(deals)
        filtered_series_a = before_series_a_filter - after_series_a_count
        if filtered_series_a > 0:
            print(f"  Filtered out {filtered_series_a} Series A deals")
    
    # Deduplicate by company name (normalized)
    if "Company" in deals.columns and len(deals) > 0:
        before_dedup_count = len(deals)
        # Create normalized company name column for deduplication
        deals["Company Normalized"] = deals["Company"].astype(str).apply(normalize_company_name)
        # Keep the first occurrence of each company
        deals = deals.drop_duplicates(subset=["Company Normalized"], keep="first")
        deals = deals.drop(columns=["Company Normalized"])
        after_dedup_count = len(deals)
        duplicates_removed = before_dedup_count - after_dedup_count
        if duplicates_removed > 0:
            print(f"  Removed {duplicates_removed} duplicate companies")
    
    print(f"  Deals: {initial_count} total → {after_date_count} in past week → {len(deals)} after filtering")
    
    return deals

def get_tracking():
    tracking_file = 'data/tracking/tracking_db.csv'
    
    # Handle missing file gracefully
    if not os.path.exists(tracking_file):
        print(f"⚠️  Tracking file not found at {tracking_file}, returning empty DataFrame")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(tracking_file)
    except Exception as e:
        print(f"⚠️  Error reading tracking file: {e}, returning empty DataFrame")
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    new_updates = df[df['most_recent_update'].notna() & (df['most_recent_update'] != '')]
    df.drop_duplicates(subset='company_name', keep='first', inplace=True)

    last_update = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    if 'last_checked' in new_updates.columns:
        # Try using last_checked instead if available
        new_updates['last_checked'] = pd.to_datetime(new_updates['last_checked'])
        new_updates = new_updates[new_updates['last_checked'] >= pd.Timestamp(last_update)]
    elif 'most_recent_update_date' in new_updates.columns:
        # Convert to datetime if it exists
        new_updates['most_recent_update_date'] = pd.to_datetime(new_updates['most_recent_update_date'])
        # Filter by date if the column exists
        new_updates = new_updates[new_updates['most_recent_update_date'] >= pd.Timestamp(last_update)]

    return new_updates

# Load the investment tree root node
def get_pipeline_stats(filter_date="2025-09-10"):
    """
    Analyze pipeline companies and return structured statistics.
    
    Returns:
        dict: {
            'total_companies': int,
            'filter_date': str,
            'dataframe': pd.DataFrame,
            'category_distribution': [{'category': str, 'count': int, 'percentage': float}, ...],
            'subcategories_by_category': {category: [{'subcategory': str, 'count': int, 'percentage': float}, ...], ...},
            'emerging_themes': {category: [{'theme': str, 'count': int}, ...], ...},
            'top_paths': [{'path': str, 'count': int}, ...]
        }
    """
    with open('data/taste_tree.json', 'r') as f:
        tree = json.load(f)

    # Find all pipeline companies across the tree
    pipeline_companies = find_pipeline_companies(tree, filter_date=filter_date)
    df = pd.DataFrame(pipeline_companies)
    df.drop_duplicates(subset='company_name', keep='first', inplace=True)

    # Extract hierarchical levels from paths
    df['category'] = df['path'].str.split(' > ').str[0]
    df['subcategory'] = df['path'].str.split(' > ').str[1]
    df['theme'] = df['path'].str.split(' > ').str[2]

    total_companies = len(df)
    
    # Analysis 1: Top-level category distribution
    category_counts = df['category'].value_counts()
    category_distribution = [
        {
            'category': cat,
            'count': int(count),
            'percentage': round((count / total_companies) * 100, 1)
        }
        for cat, count in category_counts.items()
    ]

    # Analysis 2: Dominant subcategories within each category
    subcategories_by_category = {}
    for category in category_counts.index:
        cat_df = df[df['category'] == category]
        subcategory_counts = cat_df['subcategory'].value_counts().head(5)
        subcategories_by_category[category] = [
            {
                'subcategory': subcat,
                'count': int(count),
                'percentage': round((count / len(cat_df)) * 100, 1)
            }
            for subcat, count in subcategory_counts.items()
            if pd.notna(subcat)
        ]

    # Analysis 3: Emerging themes (Level 3) within top categories
    emerging_themes = {}
    for category in category_counts.head(4).index:
        cat_df = df[df['category'] == category]
        theme_counts = cat_df['theme'].value_counts().head(5)
        if len(theme_counts) > 0 and theme_counts.iloc[0] > 1:
            emerging_themes[category] = [
                {
                    'theme': theme,
                    'count': int(count)
                }
                for theme, count in theme_counts.items()
                if pd.notna(theme) and count > 1
            ]

    # Analysis 4: Top paths
    path_counts = df['path'].value_counts().head(10)
    top_paths = [
        {
            'path': path,
            'count': int(count)
        }
        for path, count in path_counts.items()
    ]

    return {
        'total_companies': total_companies,
        'filter_date': filter_date,
        'dataframe': df,
        'category_distribution': category_distribution,
        'subcategories_by_category': subcategories_by_category,
        'emerging_themes': emerging_themes,
        'top_paths': top_paths
    }

def find_relevant_information(pipeline_dict):
    # Take the top subcategories and find more companies in those categories
    subcategories = pipeline_dict['subcategories_by_category']
    
    # Only process Healthcare, Fintech, and Commerce categories for the weekly update
    # Filter to only the categories that will be displayed
    allowed_categories = ['Healthcare', 'Fintech', 'Commerce']
    filtered_subcategories = {
        category: subs 
        for category, subs in subcategories.items() 
        if category in allowed_categories
    }
    
    # Print which categories are being processed
    if filtered_subcategories:
        print(f"\nProcessing categories: {', '.join(filtered_subcategories.keys())}")
        skipped = set(subcategories.keys()) - set(filtered_subcategories.keys())
        if skipped:
            print(f"Skipping categories (not displayed in weekly update): {', '.join(skipped)}")
    
    return filtered_subcategories

def get_founders_by_category_path(path):
    """
    Query all founders matching a category path from the database.
    
    Args:
        path: Category path to search for (e.g., "Payments" or "Fintech > Payments")
    
    Returns:
        pandas.DataFrame: DataFrame with all matching founders, or empty DataFrame if none found
    """
    conn = get_db_connection()
    if not conn:
        print(f"  ⚠️  Failed to connect to database for path: {path}")
        return pd.DataFrame()
    
    try:
        cursor = conn.cursor()
        # Query all founders matching the path (no hardcoded quality filters)
        cursor.execute(
            """
            SELECT * FROM founders 
            WHERE founder = true 
            AND history = '' 
            AND tree_path LIKE %s
            """, 
            ("%" + path + "%",)
        )
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        # Convert to DataFrame
        if rows:
            founders_df = pd.DataFrame([dict(zip(column_names, row)) for row in rows])
            return founders_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  ⚠️  Error querying founders for path {path}: {e}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_relevant_deals(subcategory, tree, filter_date=None):
    """
    Get recent deal activity for a given subcategory.
    
    Args:
        subcategory: The subcategory name to search for
        tree: The taste tree data structure
        filter_date: Optional date string (YYYY-MM-DD) to filter news by last_updated date
        
    Returns:
        dict: {
            'portfolio_companies': list of portfolio companies,
            'recent_news': str of recent funding news,
            'node_count': number of matching nodes,
            'filtered_node_count': number of nodes after date filtering
        }
    """
    from datetime import datetime
    
    # Find the nodes in the tree that contain this subcategory
    nodes = find_nodes_by_name(tree, subcategory)
    
    if not nodes:
        return {
            'portfolio_companies': [],
            'recent_news': '',
            'node_count': 0,
            'filtered_node_count': 0
        }
    
    # Apply date filter if provided
    filtered_nodes = nodes
    if filter_date:
        try:
            cutoff_date = datetime.fromisoformat(filter_date).date()
            filtered_nodes = []
            for node in nodes:
                last_updated = node.get('last_updated', '')
                if last_updated:
                    try:
                        node_date = datetime.fromisoformat(last_updated).date()
                        if node_date >= cutoff_date:
                            filtered_nodes.append(node)
                    except (ValueError, TypeError):
                        # If date parsing fails, include the node
                        filtered_nodes.append(node)
                else:
                    # If no last_updated date, include the node
                    filtered_nodes.append(node)
        except (ValueError, TypeError):
            # If filter_date is invalid, use all nodes
            filtered_nodes = nodes
    
    # Aggregate data from filtered nodes
    all_portfolio = []
    all_news = []
    
    for node in filtered_nodes:
        # Extract portfolio companies
        portfolio_count = node.get('portfolio_companies', 0)
        if portfolio_count > 0:
            # Note: The summary doesn't include full portfolio data, 
            # just the count. You'd need to fetch the full node if needed.
            all_portfolio.append({
                'node_name': node.get('name', ''),
                'path': node.get('path', ''),
                'count': portfolio_count,
                'last_updated': node.get('last_updated', '')
            })
        
        # Extract recent news
        recent_news = node.get('recent_news', '')
        if recent_news:
            all_news.append({
                'node_name': node.get('name', ''),
                'path': node.get('path', ''),
                'news': recent_news,
                'last_updated': node.get('last_updated', '')
            })
    
    # Combine all news into a single string
    combined_news = '\n\n'.join([
        f"[{item['path']}] (Updated: {item['last_updated']})\n{item['news']}" 
        for item in all_news
    ]) if all_news else ''
    
    return {
        'portfolio_companies': all_portfolio,
        'recent_news': combined_news,
        'node_count': len(nodes),
        'filtered_node_count': len(filtered_nodes)
    }

def get_recs():
    with open('data/taste_tree.json', 'r') as f:
        tree = json.load(f)

    pipeline_dict = get_pipeline_stats()
    subcategories = find_relevant_information(pipeline_dict)
    top_founders = []

    # Load ranker model for ranking founders
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    print(f"\nLoading ranker-only model from {model_dir}...")
    ranker_inference = None
    
    if xgb is None:
        print("❌ xgboost not available, will fall back to past_success_indication_score sorting")
    else:
        try:
            # Load ranker-only model and metadata
            ranker_path = os.path.join(model_dir, 'ranker_only_xgb.json')
            ranker_metadata_path = os.path.join(model_dir, 'ranker_only_feature_metadata.json')
            ranker_model_metadata_path = os.path.join(model_dir, 'ranker_only_metadata.json')
            
            if not os.path.exists(ranker_path):
                print(f"❌ Ranker-only model not found at {ranker_path}")
                print("Will fall back to past_success_indication_score sorting")
            else:
                # Load metadata
                with open(ranker_metadata_path, 'r') as f:
                    ranker_metadata = json.load(f)
                with open(ranker_model_metadata_path, 'r') as f:
                    ranker_model_metadata = json.load(f)
                
                # Load ranker model
                ranker = xgb.XGBRanker()
                ranker.load_model(ranker_path)
                
                # Get normalization parameters
                norm_params = ranker_model_metadata.get('ranker_normalization', {})
                ranker_min = norm_params.get('min')
                ranker_max = norm_params.get('max')
                
                ranker_inference = {
                    'ranker': ranker,
                    'metadata': ranker_metadata,
                    'ranker_min': ranker_min,
                    'ranker_max': ranker_max
                }
                
                print("✅ Ranker-only model loaded successfully")
                print(f"  Ranker normalization: min={ranker_min}, max={ranker_max}")
        except Exception as e:
            print(f"❌ Error loading ranker-only model: {e}")
            print("Will fall back to past_success_indication_score sorting")
            ranker_inference = None

    for category, subs in subcategories.items():
        print(f"Category: {category}")
        for sub in subs:
            print(f"  Subcategory: {sub['subcategory']} ({sub['count']} companies)")
            
            # Map to old taxonomy paths
            new_path = sub['subcategory']
            old_paths = get_all_matching_old_paths(new_path)
            mapped_paths = old_paths if old_paths else [new_path]
            
            # Query all founders matching this subcategory (no hardcoded filters)
            founders_list = []
            for path in mapped_paths:
                founders_df = get_founders_by_category_path(path)
                if not founders_df.empty:
                    founders_list.append(founders_df)
            
            # Concatenate all founders into one DataFrame
            if founders_list:
                # Suppress FutureWarning for DataFrame concatenation (harmless warning about empty/all-NA columns)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning, message='.*concatenation.*')
                    all_founders_df = pd.concat(founders_list, ignore_index=True)
            else:
                all_founders_df = pd.DataFrame()
            
            # Remove duplicates
            if not all_founders_df.empty:
                all_founders_df = all_founders_df.drop_duplicates(subset=['name', 'company_name'])
                all_founders_df = all_founders_df.reset_index(drop=True)
            
            # Use ranker model to select top founder
            if not all_founders_df.empty:
                print(f"    Found {len(all_founders_df)} founders, ranking with model...")
                
                # Use ranker model if available
                if ranker_inference is not None:
                    try:
                        from test_models import prepare_test_features
                        
                        # Prepare features for all founders
                        # Use copy to avoid modifying original dataframe
                        founders_with_features = prepare_test_features(all_founders_df.copy(), verbose=False)
                        
                        # Helper function to extract features from row (same as recommendations.py)
                        def extract_features_from_row(row, metadata):
                            """Extract features from a single row using metadata."""
                            def flatten_embedding(embedding):
                                """Flatten an embedding array/list into a numpy array."""
                                if embedding is None or (isinstance(embedding, float) and np.isnan(embedding)):
                                    return np.zeros(1536, dtype=np.float32)
                                if isinstance(embedding, (list, np.ndarray)):
                                    arr = np.array(embedding, dtype=np.float32)
                                    if arr.ndim == 1:
                                        return arr
                                    else:
                                        return arr.flatten()
                                return np.zeros(1536, dtype=np.float32)
                            
                            feature_vec = []
                            
                            # Text embedding group 1
                            text_emb1 = flatten_embedding(row.get('text_embedding_group1'))
                            text_emb_dim = metadata.get('text_embedding_group1_dim', 1536)
                            if len(text_emb1) != text_emb_dim:
                                if len(text_emb1) < text_emb_dim:
                                    text_emb1 = np.pad(text_emb1, (0, text_emb_dim - len(text_emb1)), 'constant')
                                else:
                                    text_emb1 = text_emb1[:text_emb_dim]
                            feature_vec.extend(text_emb1.tolist())
                            
                            # Text embedding group 2
                            text_emb2 = flatten_embedding(row.get('text_embedding_group2'))
                            text_emb_dim2 = metadata.get('text_embedding_group2_dim', 1536)
                            if len(text_emb2) != text_emb_dim2:
                                if len(text_emb2) < text_emb_dim2:
                                    text_emb2 = np.pad(text_emb2, (0, text_emb_dim2 - len(text_emb2)), 'constant')
                                else:
                                    text_emb2 = text_emb2[:text_emb_dim2]
                            feature_vec.extend(text_emb2.tolist())
                            
                            # Experience embedding
                            exp_emb = flatten_embedding(row.get('experience_embedding'))
                            exp_emb_dim = metadata.get('experience_embedding_dim', 1536)
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
                        
                        # Get ranker predictions for each founder
                        predictions = []
                        ranker = ranker_inference['ranker']
                        metadata = ranker_inference['metadata']
                        ranker_min = ranker_inference['ranker_min']
                        ranker_max = ranker_inference['ranker_max']
                        
                        for idx, row in founders_with_features.iterrows():
                            try:
                                # Extract features
                                X = extract_features_from_row(row, metadata)
                                
                                # Get raw ranker score
                                raw_score = ranker.predict(X)[0]
                                
                                # Normalize ranker score to [0, 1] range
                                if ranker_min is not None and ranker_max is not None and ranker_max > ranker_min:
                                    normalized_score = (raw_score - ranker_min) / (ranker_max - ranker_min)
                                    normalized_score = np.clip(normalized_score, 0.0, 1.0)
                                else:
                                    # Fallback: sigmoid normalization
                                    normalized_score = 1.0 / (1.0 + np.exp(-raw_score))
                                
                                predictions.append(float(normalized_score))
                            except Exception as e:
                                print(f"      Warning: Error predicting for {row.get('name', 'unknown')}: {e}")
                                predictions.append(0.0)  # Default to 0 if prediction fails
                        
                        # Add ranker_score to founders dataframe
                        all_founders_df['ranker_score'] = predictions
                        
                        # Sort by ranker_score (descending)
                        all_founders_df.sort_values(by='ranker_score', ascending=False, inplace=True)
                        
                        # Select top founder
                        top_founder = all_founders_df.iloc[0].to_dict()
                        print(f"    Selected top founder (ranker_score: {top_founder.get('ranker_score', 'N/A'):.4f})")
                        sub['top_founder'] = top_founder
                        
                    except Exception as e:
                        print(f"    ⚠️  Error using ranker model: {e}")
                        print("    Falling back to past_success_indication_score sorting")
                        # Fallback: Sort by past_success_indication_score
                        all_founders_df.sort_values(
                            by='past_success_indication_score',
                            ascending=False,
                            inplace=True
                        )
                        top_founder = all_founders_df.iloc[0].to_dict()
                        sub['top_founder'] = top_founder
                else:
                    # Fallback: Sort by past_success_indication_score
                    all_founders_df.sort_values(
                        by='past_success_indication_score',
                        ascending=False,
                        inplace=True
                    )
                    top_founder = all_founders_df.iloc[0].to_dict()
                    sub['top_founder'] = top_founder
            else:
                sub['top_founder'] = None

            top_founders.append(sub.get('top_founder'))
            
            # Get relevant deal activity for this subcategory (with same date filter)
            deal_data = get_relevant_deals(sub['subcategory'], tree, filter_date="2025-10-01")
            sub['deal_activity'] = deal_data

            # Also attach 'interest' metadata from the corresponding subcategory node
            interest_text = ""
            try:
                nodes = find_nodes_by_name(tree, sub['subcategory'])
                for node in nodes:
                    meta = node.get('meta', {}) if isinstance(node, dict) else {}
                    interest_val = meta.get('interest', '')
                    if isinstance(interest_val, str) and interest_val.strip():
                        interest_text = interest_val.strip()
                        break
            except Exception:
                # If anything fails, leave interest_text as empty
                pass
            sub['interest'] = interest_text

    return subcategories, pipeline_dict


# Collect founders and companies that will be shown in the newsletter's recommendations
def collect_recommended_entities(recs_dict):
    names = []
    companies = []
    for category, subs in recs_dict.items():
        if not subs:
            continue
        for sub in subs:
            founders = sub.get("founders", [])
            founder = founders[0] if founders else sub.get("top_founder")
            if not founder:
                continue
            name = founder.get("name") or founder.get("Name")
            company = founder.get("company_name") or founder.get("Company")
            if name:
                names.append(name)
            if company:
                companies.append(company)
    # Deduplicate
    return list(dict.fromkeys(names)), list(dict.fromkeys(companies))

def mark_as_recommended(recs):
    try:
        rec_names, rec_companies = collect_recommended_entities(recs)
        if rec_names or rec_companies:
            mark_prev_recs(names=rec_names, companies=rec_companies)
    except Exception as e:
        print(f"Warning: could not mark previous recommendations: {e}")

def get_profile_recs():
    """Get profile recommendations for the weekly update.
    
    Returns:
        tuple: (profiles_list, profile_ids_to_mark)
    """
    try:
        profiles = get_profile_recommendations(limit=3)
        
        if not profiles:
            return [], []
        
        profile_ids = [p['id'] for p in profiles]
        
        return profiles, profile_ids
    except Exception as e:
        print(f"Warning: could not get profile recommendations: {e}")
        return [], []

# Load the tree for deal lookups
def main():
    from services.weekly_formatting import generate_html
    from services.google_client import send_email
    
    print("Gathering recent deals...")
    recent_deals = get_recent_deals()
    print(f"Found {len(recent_deals)} recent deals")
    
    print("\nGathering tracking updates...")
    tracking = get_tracking()
    print(f"Found {len(tracking)} tracking updates")
    
    print("\nGenerating recommendations...")
    recs, pipeline_dict = get_recs()
    print(f"Generated recommendations for {len(recs)} categories")
    
    print("\nGetting profile recommendations...")
    profile_recs, profile_ids = get_profile_recs()
    if profile_recs:
        print(f"Found {len(profile_ids)} tracking profile recommendations")
    else:
        print("No new tracking profile recommendations available")
        return
    
    print("\nGenerating HTML email...")
    greeting_text = "Happy Friday! Hope everyone is staying warm and cozy as we head into the holiday season ⛄🌟🍷"
    greeting_text += "There were a ton of deals done this week, highlighted below, and I'm very excited about the early stage companies I've sourced for you. "
    greeting_text += "I'm always developing my algorithms to make better sourcing recommendations for you all, this week I've found some really interesting companies aligned with what we have in the pipeline."
    greeting_text += "\n\nWishing you all a great weekend!\n\n - Monty"
    html_output = generate_html(recent_deals, tracking, recs, pipeline_dict, greeting_text, profile_recs=profile_recs)
    
    # Save to file
    output_path = 'data/weekly_update_output.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"\n✅ Weekly update generated successfully!")
    print(f"📄 Saved to: {output_path}")

    # Send email
    #send_email(html_output)

    # Mark recommended founders/companies to avoid repeats
    #mark_as_recommended(recs)
    
    # Mark recommended profiles to avoid repeats
    #if profile_ids:
    #    mark_profiles_as_recommended(profile_ids)
    #    print(f"Marked {len(profile_ids)} profiles as recommended")
    
    return html_output


if __name__ == "__main__":
    main()

