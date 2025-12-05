"""
Recommendations System for Monty
=================================

This module handles personalized founder/company recommendations sent to team members via Slack.

Main Functions:
--------------
1. send_extra_recs() - Main function to send recommendations to all users
   - Fetches new recommended profiles from the database
   - Matches them to each user's areas of interest (from taste tree in Supabase/JSON)
   - Sends personalized Slack DMs with top 3 recommendations per user
   - Marks sent recommendations to avoid duplicates

2. find_new_recs(username) - Preview recommendations for a specific user (testing)

3. mark_prev_recs(names, companies) - Mark profiles as 'recommended' in database

4. filter_categories_by_tree_path(categories, user) - Filter categories by tree_path for specific users

Category Filtering:
------------------
You can configure per-user category filters using the `user_category_filters` dictionary.
This allows you to restrict which categories a user receives based on keywords in the tree_path.

Example:
    user_category_filters = {
        "Connie": ["Commerce"],  # Only send Connie categories with "Commerce" in tree_path
        "Todd": ["Fintech"],     # Only send Todd categories with "Fintech" in tree_path
    }

To disable filtering for a user, either:
    - Don't include them in the dictionary, or
    - Set their value to None or an empty list

Usage:
------
To send recommendations to all users:
    from workflows.recommendations import send_extra_recs
    send_extra_recs()

To preview recommendations for a specific user:
    from workflows.recommendations import find_new_recs
    find_new_recs("Matthildur")

Requirements:
------------
- SLACK_BOT_TOKEN environment variable must be set
- Database must have 'founders' table with tree_result and history columns
- Taste tree (in Supabase or data/taste_tree.json) must have 'montage_lead' metadata for user assignments
- For Supabase: SUPABASE_URL and SUPABASE_KEY environment variables must be set
"""

import pandas as pd
import time
import os
import warnings
from psycopg2 import sql

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Slack user ID mapping
user_map = {
    "Todd": "U03KZ1KQF",
    "Matt": "U9V59N8R1",
    "Daphne": "U03L4HK4S",
    "Connie": "U03JA4UARF1",
    "Nia": "U04V74ZMZ7F",
    "Matthildur": "U07MTGUFMSB"
}

# User category filter configuration
# Maps username to a list of strings that must appear in the tree_path
# If None or empty list, no filtering is applied (user gets all their assigned categories)
# Example: {"Connie": ["Commerce"]} means Connie only gets categories with "Commerce" in the tree_path
user_category_filters = {
    "Connie": ["Commerce"],  # Uncomment to enable filtering for Connie
    # Add more users as needed:
    #"Todd": ["Fintech"],
    # "Daphne": ["Healthcare"],
}

def filter_categories_by_tree_path(categories, user):
    """Filter categories for a user based on tree_path filters.
    
    Args:
        categories: List of category paths (e.g., ["Commerce > E-commerce", "Fintech > Payments"])
        user: Username to check filters for
    
    Returns:
        Filtered list of categories that match the user's tree_path filter (if enabled)
    """
    # Check if user has a filter configured
    if user not in user_category_filters or not user_category_filters[user]:
        # No filter configured, return all categories
        return categories
    
    # Get the filter strings for this user
    filter_strings = user_category_filters[user]
    if not filter_strings:
        return categories
    
    # Filter categories that contain any of the filter strings in their path
    filtered = []
    for category in categories:
        # Check if any filter string appears in the category path
        if any(filter_str in category for filter_str in filter_strings):
            filtered.append(category)
    
    if filtered:
        print(f"  Applied tree_path filter for {user}: {filter_strings}")
        print(f"  Filtered from {len(categories)} to {len(filtered)} categories")
    else:
        print(f"  ⚠️  Warning: Filter for {user} ({filter_strings}) removed all categories!")
    
    return filtered


def find_new_recs(username, test=True):
    """Find new recommendations for a specific user (for testing/preview purposes).
    
    Args:
        username: Name of the user to find recommendations for
        test: If True, only prints recommendations. If False, sends Slack message.
    """
    from services.tree import get_nodes_and_names
    from services.database import get_db_connection
    from services.path_mapper import get_all_matching_old_paths
    
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to the database")
        return None
    try:
        new_profiles = pd.read_sql(f"""
        SELECT *
        FROM founders
        WHERE founder = true
        AND history = ''
        AND (tree_result = 'Strong recommend' OR tree_result = 'Recommend')
        """, conn)
        new_profiles = new_profiles.drop_duplicates(subset=['name'])
        new_profiles = new_profiles.reset_index(drop=True)        
    except Exception as e:
        print(f"Error fetching new profiles: {e}")
        return None
    finally:
        if conn:
            conn.close()

    columns_to_share = ['name', 'company_name', 'company_website', 'profile_url', 'tree_thesis', 'product', 'market', 'tree_path', 'past_success_indication_score', 'tree_result']
    recs = new_profiles[columns_to_share].copy()
    recs = recs.drop_duplicates(subset=['company_name'])
    recs = recs.drop_duplicates(subset=['name'])
    recs = recs.drop_duplicates(subset=['profile_url'])
    recs = recs.reset_index(drop=True)

    # Load user interests from Supabase (with fallback to JSON)
    print("Loading user interests from taste tree...")
    nodes_and_names = get_nodes_and_names(use_supabase=True)

    recs['top_category'] = recs['tree_path'].apply(lambda x: x.split(' > ')[0] if pd.notna(x) else '')

    for user, categories in nodes_and_names.items():
        if user != username:
            continue
        print(f"\nProfiles for {user}")
        print("-" * 60)
        
        # Apply category filter if configured for this user
        categories = filter_categories_by_tree_path(categories, user)
        
        if not categories:
            print(f"No categories remaining after filtering for {user}")
            return
        
        profiles = pd.DataFrame()
        for category in categories:
            # Get all old paths that map to this new category
            old_paths = get_all_matching_old_paths(category)
            
            # Match profiles using both new and old paths
            # First try new path (for any newly categorized profiles)
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
            
            # Then match using old paths (for existing database entries)
            for old_path in old_paths:
                old_matching = recs[recs['tree_path'].str.contains(old_path, na=False, regex=False)]
                matching = pd.concat([matching, old_matching])
            
            # Remove duplicates and add to profiles
            matching = matching.drop_duplicates(subset=['name', 'company_name'])
            profiles = pd.concat([profiles, matching])
        
        if len(profiles) == 0:
            print(f"No matching profiles found for {user}")
            return
        
        profiles['category'] = profiles['tree_path'].apply(lambda x: x.split(' > ')[-2] if len(x.split(' > ')) > 1 else x)

        # Get top profile from each category
        profiles = profiles.groupby('category').head(1)

        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)
        profiles.sort_values(by=['tree_result', 'past_success_indication_score'], ascending=[False, False], inplace=True)
        
        # Take top 3
        profiles = profiles.head(3)

        print(f"Found {len(profiles)} recommendations:\n")
        
        # Build category string for greeting
        categories_mentioned = profiles['tree_path'].unique()
        category_string = ""
        for category in categories_mentioned:
            last_node = category.split(' > ')[-2] if len(category.split(' > ')) > 1 else category.split(' > ')[0]
            category_string += f"{last_node}, "
        category_string = category_string[:-2]  # Remove trailing comma
        
        # Build Slack message
        greeting_text = f"Hey {user}! I came across these profiles in {category_string} that I wanted to share with you."
        message_lines = [greeting_text]
        
        for _, row in profiles.iterrows():
            # Format each recommendation
            line = (
                f"• <{fix_profile_url(row['profile_url'])}|*{row['name']}*> at {fix_company_url(row['company_website'], row['company_name'])}\n"
                f"    {row['product']}"
            )
            message_lines.append(line)
            
            # Print to console
            profile_link = fix_profile_url(row['profile_url'])
            print(f"  - {row['name']} at {row['company_name']} ({row['tree_result']})")
            print(f"    Profile: {profile_link}")
            print(f"    Category: {row['tree_path']}")
            print()
        
        message = "\n\n".join(message_lines)
        
        # Send message if not in test mode
        user_id = user_map.get(user)
        if user_id:
            if test:
                print("=" * 60)
                print("TEST MODE - Message preview:")
                print("=" * 60)
                print(message)
                print("=" * 60)
                print("Won't send Slack message (test=True)")
            else:
                print("=" * 60)
                print(f"Sending Slack message to {user} ({user_id})")
                send_slack_dm(user_id, message)
                print("✅ Message sent!")
        else:
            print(f"⚠️  No Slack user ID found for {user}")


def fix_profile_url(url):
    """Ensure LinkedIn profile URL is properly formatted."""
    if not url:
        return "#"
    if not url.startswith('http'):
        return f"https://{url}"
    return url


def fix_company_url(url, company_name):
    """Format company URL for Slack display."""
    if not url or url == '' or pd.isna(url):
        return company_name
    if not url.startswith('http'):
        url = f"https://{url}"
    return f"<{url}|*{company_name}*>"


def send_slack_dm(user_id, message):
    """Send a direct message to a Slack user.
    
    Args:
        user_id: Slack user ID (e.g., 'U07MTGUFMSB')
        message: Message text to send
    """
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        print("❌ SLACK_BOT_TOKEN not found in environment")
        return False
    
    client = WebClient(token=slack_token)
    
    try:
        response = client.chat_postMessage(
            channel=user_id,
            text=message
        )
        return True
    except SlackApiError as e:
        print(f"❌ Error sending Slack message: {e.response['error']}")
        return False


def mark_prev_recs(names=None, companies=None):
    """Mark profiles as 'recommended' in the database to avoid sending duplicates.
    
    Args:
        names: List of founder names to mark as recommended
        companies: List of company names to mark as recommended
    """
    from services.database import get_db_connection
    
    if not names and not companies:
        print("No names or companies provided to mark")
        return
    
    all_names = set(names) if names else set()
    companies_set = set(companies) if companies else set()
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to the database")
        return
        
    try:
        cur = conn.cursor()
        for table_name in ["founders", "stealth_founders"]:
            query = sql.SQL(
                "UPDATE {} SET history = 'recommended' WHERE name = ANY(%s) OR company_name = ANY(%s);"
            ).format(sql.Identifier(table_name))
            cur.execute(query, (list(all_names), list(companies_set)))
            print(f"Updated {cur.rowcount} records in {table_name}.")
        conn.commit()
        print(f"✅ Successfully marked profiles as 'recommended'.")
    except Exception as e:
        print(f"❌ Error updating records: {e}")
        conn.rollback()
    finally:
        if 'cur' in locals():
            cur.close()
        if conn:
            conn.close()


def send_extra_recs(test=True):
    """Send personalized recommendations to all Slack users based on their interests.
    
    This function:
    1. Fetches new recommended profiles from the database
    2. Matches them to each user's areas of interest (from the taste tree)
    3. Uses ML model to rank profiles by combined_score
    4. Sends personalized Slack DMs with top 3 recommendations per user
    5. Marks sent recommendations to avoid duplicates
    """
    import sys
    import os
    # Add tests directory to path to import model inference
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests'))
    
    import json
    import numpy as np
    from services.tree import get_nodes_and_names
    from services.database import get_db_connection
    from services.path_mapper import get_all_matching_old_paths
    from services.notion import import_pipeline, normalize_string
    from test_models import prepare_test_features
    from psycopg2 import sql
    
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    
    # Notion database IDs
    PIPELINE_ID = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
    TRACKED_ID = "912974853b494f98a5652fcbff3ad795"
    PASSED_ID = "bc5f875961234aa6aa4b293cf1915ac2"
    
    # Load ranker-only model for ranking
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
    print(f"\nLoading ranker-only model from {model_dir}...")
    ranker_inference = None
    
    if xgb is None:
        print("❌ xgboost not available, falling back to tree_result sorting")
    else:
        try:
            # Load ranker-only model and metadata
            ranker_path = os.path.join(model_dir, 'ranker_only_xgb.json')
            ranker_metadata_path = os.path.join(model_dir, 'ranker_only_feature_metadata.json')
            ranker_model_metadata_path = os.path.join(model_dir, 'ranker_only_metadata.json')
            
            if not os.path.exists(ranker_path):
                print(f"❌ Ranker-only model not found at {ranker_path}")
                print("Falling back to tree_result and past_success_indication_score sorting")
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
            print("Falling back to tree_result and past_success_indication_score sorting")
            ranker_inference = None
    
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to the database")
        return None
    
    try:
        # Fetch all new profiles that are recommended but haven't been sent yet
        new_profiles = pd.read_sql("""
        SELECT *
        FROM founders
        WHERE founder = true
        AND history = ''
        AND (tree_result = 'Strong recommend' OR tree_result = 'Recommend')
        """, conn)
        new_profiles = new_profiles.drop_duplicates(subset=['name'])
        new_profiles = new_profiles.reset_index(drop=True)
        
        print(f"Found {len(new_profiles)} new recommended profiles")
        
    except Exception as e:
        print(f"❌ Error fetching new profiles: {e}")
        return None
    finally:
        if conn:
            conn.close()
    
    # Keep full profiles for feature preparation, but also create a filtered version for matching
    # Remove duplicates from full dataset
    new_profiles = new_profiles.drop_duplicates(subset=['company_name'])
    new_profiles = new_profiles.drop_duplicates(subset=['name'])
    new_profiles = new_profiles.drop_duplicates(subset=['profile_url'])
    new_profiles = new_profiles.reset_index(drop=True)
    
    print(f"After deduplication: {len(new_profiles)} unique profiles")
    
    # Load companies from Notion pipeline/tracked/passed databases
    print("\nLoading companies from Notion databases...")
    pipeline_companies = set()
    tracked_companies = set()
    passed_companies = set()
    
    try:
        pipeline_df = import_pipeline(PIPELINE_ID)
        for company_name in pipeline_df['company_name'].dropna():
            normalized = normalize_string(str(company_name))
            if normalized:
                pipeline_companies.add(normalized)
        print(f"  Pipeline: {len(pipeline_companies)} companies")
    except Exception as e:
        print(f"  ⚠️  Error loading pipeline: {e}")
    
    try:
        tracked_df = import_pipeline(TRACKED_ID)
        for company_name in tracked_df['company_name'].dropna():
            normalized = normalize_string(str(company_name))
            if normalized:
                tracked_companies.add(normalized)
        print(f"  Tracked: {len(tracked_companies)} companies")
    except Exception as e:
        print(f"  ⚠️  Error loading tracked: {e}")
    
    try:
        passed_df = import_pipeline(PASSED_ID)
        for company_name in passed_df['company_name'].dropna():
            normalized = normalize_string(str(company_name))
            if normalized:
                passed_companies.add(normalized)
        print(f"  Passed: {len(passed_companies)} companies")
    except Exception as e:
        print(f"  ⚠️  Error loading passed: {e}")
    
    # Check which companies are already in Notion and update database
    print("\nChecking for companies already in Notion databases...")
    companies_to_update = {
        'pipeline': [],
        'tracked': [],
        'pass': []
    }
    
    for idx, row in new_profiles.iterrows():
        company_name = row.get('company_name', '')
        if pd.notna(company_name) and company_name:
            try:
                normalized = normalize_string(str(company_name))
                if normalized in pipeline_companies:
                    companies_to_update['pipeline'].append((row.get('name'), company_name))
                elif normalized in tracked_companies:
                    companies_to_update['tracked'].append((row.get('name'), company_name))
                elif normalized in passed_companies:
                    companies_to_update['pass'].append((row.get('name'), company_name))
            except (ValueError, TypeError) as e:
                continue
    
    # Update database with history values
    if any(companies_to_update.values()):
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                for history_value, company_list in companies_to_update.items():
                    if company_list:
                        names = [item[0] for item in company_list if item[0]]
                        companies = [item[1] for item in company_list if item[1]]
                        if names or companies:
                            query = sql.SQL(
                                "UPDATE founders SET history = %s WHERE (name = ANY(%s) OR company_name = ANY(%s)) AND history = '';"
                            )
                            cur.execute(query, (history_value, list(set(names)), list(set(companies))))
                            print(f"  Updated {cur.rowcount} companies with history = '{history_value}'")
                conn.commit()
            except Exception as e:
                print(f"  ⚠️  Error updating database: {e}")
                conn.rollback()
            finally:
                if 'cur' in locals():
                    cur.close()
                conn.close()
    
    # Filter out companies that are in Notion databases
    print("\nFiltering out companies already in Notion...")
    initial_count = len(new_profiles)
    notion_companies_all = pipeline_companies | tracked_companies | passed_companies
    
    mask_not_in_notion = pd.Series([True] * len(new_profiles), index=new_profiles.index)
    for idx, row in new_profiles.iterrows():
        company_name = row.get('company_name', '')
        if pd.notna(company_name) and company_name:
            try:
                normalized = normalize_string(str(company_name))
                if normalized in notion_companies_all:
                    mask_not_in_notion[idx] = False
            except (ValueError, TypeError):
                continue
    
    new_profiles = new_profiles[mask_not_in_notion].copy()
    filtered_count = initial_count - len(new_profiles)
    print(f"  Filtered out {filtered_count} companies already in Notion (remaining: {len(new_profiles)})")
    
    if len(new_profiles) == 0:
        print("No companies left after filtering. Exiting.")
        return
    
    # Select relevant columns for matching and display
    columns_to_share = ['name', 'company_name', 'company_website', 'profile_url', 
                       'tree_thesis', 'product', 'market', 'tree_path', 
                       'past_success_indication_score', 'tree_result']
    recs = new_profiles[columns_to_share].copy()
    
    # Get user interest mappings from the taste tree (from Supabase with fallback to JSON)
    print("Loading user interests from taste tree...")
    nodes_and_names = get_nodes_and_names(use_supabase=True)
    
    # Add top category for filtering
    recs['top_category'] = recs['tree_path'].apply(lambda x: x.split(' > ')[0] if pd.notna(x) else '')
    
    all_names = []
    companies = []
    
    # Process each user
    for user, categories in nodes_and_names.items():
        print(f"\nProcessing recommendations for {user}")
        
        # Apply category filter if configured for this user
        categories = filter_categories_by_tree_path(categories, user)
        
        if not categories:
            print(f"  ⚠️  No categories remaining after filtering for {user}")
            continue
        
        # Filter profiles matching this user's categories
        profiles = pd.DataFrame()
        for category in categories:
            # Get all old paths that map to this new category
            old_paths = get_all_matching_old_paths(category)
            
            # Match profiles using both new and old paths
            # First try new path (for any newly categorized profiles)
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
            
            # Then match using old paths (for existing database entries)
            for old_path in old_paths:
                old_matching = recs[recs['tree_path'].str.contains(old_path, na=False, regex=False)]
                matching = pd.concat([matching, old_matching])
            
            # Remove duplicates before adding to profiles
            matching = matching.drop_duplicates(subset=['name', 'company_name'])
            profiles = pd.concat([profiles, matching])
        
        if len(profiles) == 0:
            print(f"  ⚠️  No matching profiles found for {user}")
            continue
        
        # Extract category for grouping
        profiles['category'] = profiles['tree_path'].apply(
            lambda x: x.split(' > ')[-2] if len(x.split(' > ')) > 1 else x
        )
        
        # Get top profile from each category (before ML ranking)
        profiles = profiles.groupby('category').head(1)
        
        # Remove duplicates
        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)
        
        # Use ranker-only model to rank profiles if available
        if ranker_inference is not None and len(profiles) > 0:
            print(f"  Ranking {len(profiles)} profiles using ranker-only model...")
            try:
                # Get full profile data from new_profiles for feature preparation
                # Use name and company_name as keys to match
                profile_keys = profiles[['name', 'company_name']].copy()
                profiles_full = profile_keys.merge(
                    new_profiles, 
                    on=['name', 'company_name'], 
                    how='left'
                )
                
                # Prepare features for ML model (suppress verbose output)
                profiles_with_features = prepare_test_features(profiles_full, verbose=False)
                
                # Helper function to extract features from row (same as InferenceModule)
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
                
                # Get ranker predictions for each profile
                predictions = []
                ranker = ranker_inference['ranker']
                metadata = ranker_inference['metadata']
                ranker_min = ranker_inference['ranker_min']
                ranker_max = ranker_inference['ranker_max']
                
                for idx, row in profiles_with_features.iterrows():
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
                        print(f"    Warning: Error predicting for {row.get('name', 'unknown')}: {e}")
                        predictions.append(0.0)  # Default to 0 if prediction fails
                
                # Add ranker_score to profiles
                profiles['ranker_score'] = predictions
                
                # Sort by ranker_score (descending)
                profiles.sort_values(by='ranker_score', ascending=False, inplace=True)
                print(f"  Ranker ranking complete. Top score: {profiles['ranker_score'].iloc[0]:.4f}")
            except Exception as e:
                print(f"  ⚠️  Error using ranker model: {e}")
                print("  Falling back to tree_result and past_success_indication_score sorting")
                profiles.sort_values(
                    by=['tree_result', 'past_success_indication_score'], 
                    ascending=[False, False], 
                    inplace=True
                )
        else:
            # Fallback: Sort by recommendation strength and success score
            profiles.sort_values(
                by=['tree_result', 'past_success_indication_score'], 
                ascending=[False, False], 
                inplace=True
            )
        
        # Take top 3 recommendations
        profiles = profiles.head(3)
        
        if len(profiles) == 0:
            print(f"No profiles left after filtering for {user}")
            continue
        
        # Track sent recommendations
        all_names.extend(profiles['name'].tolist())
        companies.extend(profiles['company_name'].tolist())
        
        # Build category string for greeting
        categories_mentioned = profiles['tree_path'].unique()
        category_string = ""
        for category in categories_mentioned:
            last_node = category.split(' > ')[-2] if len(category.split(' > ')) > 1 else category.split(' > ')[0]
            category_string += f"{last_node}, "
        category_string = category_string[:-2].lower()  # Remove trailing comma
        
        # Build Slack message
        greeting_text = f"Hey {user}! I came across these profiles in {category_string} that I wanted to share with you."
        message_lines = [greeting_text]
        
        print(greeting_text)
        
        for _, row in profiles.iterrows():
            # Format each recommendation
            line = (
                f"• <{fix_profile_url(row['profile_url'])}|*{row['name']}*> at {fix_company_url(row['company_website'], row['company_name'])}\n"
                f"    {row['product']}"
            )
            message_lines.append(line)
            # Print with profile URL for easy access
            profile_link = fix_profile_url(row['profile_url'])
            score_info = f"ranker_score: {row['ranker_score']:.4f}" if 'ranker_score' in row else f"tree_result: {row['tree_result']}"
            print(f"    - {row['name']} at {row['company_name']} ({score_info})")
            print(f"      Profile: {profile_link}")
        
        message = "\n\n".join(message_lines)
        
        # Remove profiles from the pool so they're not sent to other users
        recs = recs[~recs['company_name'].isin(profiles['company_name'])]
        
        # Send the message
        user_id = user_map.get(user)
        if user_id:
            if test:
                print("Won't send anything yet")
            else:
                print(f"Sending to Slack user {user_id}")
                send_slack_dm(user_id, message)
                time.sleep(2)  # Rate limiting
        else:
            print(f"No Slack user ID found for {user}")
    
    # Mark all sent recommendations in the database
    if all_names or companies:
        if test:
            print("Won't mark anything as recommended")
        else:
            print(f"Marking {len(set(all_names))} profiles as recommended")
            mark_prev_recs(all_names, companies)
    
    print("Recommendation sending complete!")


def main():
    """Main function to send recommendations."""
    
    print("=" * 60)
    print("Monty Recommendations System")
    print("=" * 60)
    
    # Option 1: Preview recommendations for a specific user (testing)
    ##print("\n📋 Preview Mode - Testing recommendations for Matthildur:")
    #print("-" * 60)
    #find_new_recs("Matthildur", test=False)
    
    # Option 2: Send recommendations to all users
    # Uncomment the lines below when ready to send to everyone
    print("\n📧 Sending recommendations to all users:")
    print("-" * 60)
    send_extra_recs(test=True)
    
    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()