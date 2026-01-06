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
    """Get tracking updates from Supabase (with CSV fallback).
    
    Returns:
        pandas.DataFrame: DataFrame with tracking updates from the last 5 days
    """
    # Try to import tracking functions
    try:
        from workflows.tracking import load_tracking_from_supabase
    except ImportError:
        print("⚠️  Could not import tracking functions, falling back to CSV")
        load_tracking_from_supabase = None
    
    # Try Supabase first
    if load_tracking_from_supabase:
        try:
            df = load_tracking_from_supabase()
            if not df.empty:
                # Filter for updates from last 5 days
                last_update = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
                
                # Filter for companies with updates
                new_updates = df[df['most_recent_update'].notna() & (df['most_recent_update'] != '')]
                
                if 'last_checked' in new_updates.columns:
                    # Filter by last_checked date
                    new_updates['last_checked'] = pd.to_datetime(new_updates['last_checked'], errors='coerce')
                    new_updates = new_updates[new_updates['last_checked'] >= pd.Timestamp(last_update)]
                elif 'update_date' in new_updates.columns:
                    # Fallback to update_date
                    new_updates['update_date'] = pd.to_datetime(new_updates['update_date'], errors='coerce')
                    new_updates = new_updates[new_updates['update_date'] >= pd.Timestamp(last_update)]
                
                # Remove duplicates
                new_updates = new_updates.drop_duplicates(subset='company_name', keep='first')
                
                return new_updates
        except Exception as e:
            print(f"⚠️  Error loading from Supabase: {e}, falling back to CSV")
    
    # Fallback to CSV
    tracking_file = 'data/tracking/tracking_db.csv'
    
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
        new_updates['last_checked'] = pd.to_datetime(new_updates['last_checked'], errors='coerce')
        new_updates = new_updates[new_updates['last_checked'] >= pd.Timestamp(last_update)]
    elif 'most_recent_update_date' in new_updates.columns:
        # Convert to datetime if it exists
        new_updates['most_recent_update_date'] = pd.to_datetime(new_updates['most_recent_update_date'], errors='coerce')
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

    # --------------------------------------------------
    # Load ranker model
    # --------------------------------------------------
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
            ranker_path = os.path.join(model_dir, 'ranker_only_xgb.json')
            ranker_metadata_path = os.path.join(model_dir, 'ranker_only_feature_metadata.json')
            ranker_model_metadata_path = os.path.join(model_dir, 'ranker_only_metadata.json')

            if not os.path.exists(ranker_path):
                print(f"❌ Ranker-only model not found at {ranker_path}")
            else:
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
                    'ranker_max': norm.get('max')
                }

                print("✅ Ranker-only model loaded successfully")
                print(f"  Ranker normalization: min={norm.get('min')}, max={norm.get('max')}")

        except Exception as e:
            print(f"❌ Error loading ranker-only model: {e}")
            ranker_inference = None

    # --------------------------------------------------
    # Iterate categories / subcategories
    # --------------------------------------------------
    for category, subs in subcategories.items():
        print(f"Category: {category}")

        for sub in subs:
            print(f"  Subcategory: {sub['subcategory']} ({sub['count']} companies)")

            # Map taxonomy
            new_path = sub['subcategory']
            old_paths = get_all_matching_old_paths(new_path)
            mapped_paths = old_paths if old_paths else [new_path]

            # Load founders
            founders_list = []
            for path in mapped_paths:
                df = get_founders_by_category_path(path)
                if not df.empty:
                    founders_list.append(df)

            if not founders_list:
                sub['top_founder'] = None
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                all_founders_df = pd.concat(founders_list, ignore_index=True)

            all_founders_df = (
                all_founders_df
                .drop_duplicates(subset=['name', 'company_name'])
                .reset_index(drop=True)
            )

            print(f"    Found {len(all_founders_df)} founders")

            # --------------------------------------------------
            # Rank with model (or fallback)
            # --------------------------------------------------
            if ranker_inference is not None:
                try:
                    from test_models import prepare_test_features

                    founders_with_features = prepare_test_features(
                        all_founders_df.copy(),
                        verbose=False
                    )

                    # --- Build feature matrix ---
                    X_all = []
                    valid_indices = []

                    def flatten_embedding(emb):
                        if emb is None or (isinstance(emb, float) and np.isnan(emb)):
                            return np.zeros(1536, dtype=np.float32)
                        arr = np.asarray(emb, dtype=np.float32)
                        return arr.flatten()[:1536]

                    metadata = ranker_inference['metadata']

                    for idx, row in founders_with_features.iterrows():
                        try:
                            vec = []

                            vec.extend(flatten_embedding(row.get('text_embedding_group1')))
                            vec.extend(flatten_embedding(row.get('text_embedding_group2')))
                            vec.extend(flatten_embedding(row.get('experience_embedding')))

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
                        raise RuntimeError("No valid feature rows for ranking")

                    X_all = np.asarray(X_all, dtype=np.float32)

                    # --- Predict ---
                    raw_scores = ranker_inference['ranker'].predict(X_all)

                    # --- Normalize ---
                    rmin = ranker_inference['ranker_min']
                    rmax = ranker_inference['ranker_max']

                    if rmin is not None and rmax is not None and rmax > rmin:
                        scores = (raw_scores - rmin) / (rmax - rmin)
                        scores = np.clip(scores, 0.0, 1.0)
                    else:
                        scores = 1.0 / (1.0 + np.exp(-raw_scores))

                    founders_with_features['ranker_score'] = 0.0
                    founders_with_features.loc[valid_indices, 'ranker_score'] = scores

                    founders_with_features.sort_values(
                        by='ranker_score',
                        ascending=False,
                        inplace=True
                    )

                    top_founder = founders_with_features.iloc[0].to_dict()
                    sub['top_founder'] = top_founder

                    print(
                        f"    Selected top founder "
                        f"(ranker_score={top_founder.get('ranker_score', 0):.4f})"
                    )

                except Exception as e:
                    print(f"    ⚠️ Ranker failed: {e}")
                    all_founders_df.sort_values(
                        by='past_success_indication_score',
                        ascending=False,
                        inplace=True
                    )
                    sub['top_founder'] = all_founders_df.iloc[0].to_dict()

            else:
                all_founders_df.sort_values(
                    by='past_success_indication_score',
                    ascending=False,
                    inplace=True
                )
                sub['top_founder'] = all_founders_df.iloc[0].to_dict()

            # --------------------------------------------------
            # Deal activity + interest
            # --------------------------------------------------
            sub['deal_activity'] = get_relevant_deals(
                sub['subcategory'],
                tree,
                filter_date="2025-10-01"
            )

            interest_text = ""
            try:
                nodes = find_nodes_by_name(tree, sub['subcategory'])
                for node in nodes:
                    meta = node.get('meta', {})
                    val = meta.get('interest', '')
                    if isinstance(val, str) and val.strip():
                        interest_text = val.strip()
                        break
            except Exception:
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
    greeting_text = "Happy Friday everyone! ⛄🌟🍷 Hopefully everyone is looking forward to the holidays and all the good food. I'll hold down the fort on sourcing while you all get a well deserved break."
    greeting_text += " There were a ton of deals done this week, highlighted below, and I'm very excited about the early stage companies I've sourced for you. "
    greeting_text += "\n\nWishing you all happy holidays and a great weekend!\n\n - Monty"
    html_output = generate_html(recent_deals, tracking, recs, pipeline_dict, greeting_text, profile_recs=profile_recs)
    
    # Save to file
    output_path = 'data/weekly_update_output.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"\n✅ Weekly update generated successfully!")
    print(f"📄 Saved to: {output_path}")

    # Send email
    send_email(html_output)

    # Mark recommended founders/companies to avoid repeats
    mark_as_recommended(recs)
    
    # Mark recommended profiles to avoid repeats
    if profile_ids:
        mark_profiles_as_recommended(profile_ids)
        print(f"Marked {len(profile_ids)} profiles as recommended")
    
    return html_output


if __name__ == "__main__":
    main()

