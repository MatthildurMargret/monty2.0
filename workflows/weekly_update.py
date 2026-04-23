import json
import sys
import os
import numpy as np
import warnings

# Ensure project root is on sys.path when run as a subprocess
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from services.tree_tools import find_pipeline_companies, find_nodes_by_name
from services.database import get_db_connection
import pandas as pd
from datetime import datetime, timedelta
from workflows.recommendations import mark_prev_recs, is_company_less_than_2_years_old
from services.path_mapper import get_all_matching_old_paths
from workflows.profile_recommendations import get_profile_recommendations, mark_profiles_as_recommended, format_profiles_for_weekly_update
from services.deal_processing import normalize_company_name
from services.model_loader import load_ranker_model
from services.ranker_inference import rank_profiles


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
    past_week = datetime.now() - timedelta(days=7)
    deals["Date"] = pd.to_datetime(deals["Date"], errors="coerce")
    initial_count = len(deals)
    
    # Check for invalid dates
    invalid_dates = deals["Date"].isna().sum()
    if invalid_dates > 0:
        print(f"  ⚠️  Warning: {invalid_dates} deals have invalid/missing dates (will be excluded)")
    
    # Use >= to include deals from exactly 7 days ago
    deals = deals[deals["Date"] >= past_week].copy()
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
        pass
    
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
        pass
    
    print(f"  Deals: {initial_count} total → {after_date_count} in past week → {len(deals)} after filtering")
    
    return deals

def _filter_tracking_updates(df, days=5):
    """Filter a tracking DataFrame to only include recent updates.

    Filters rows that have a non-empty most_recent_update and whose date
    (last_checked, update_date, or most_recent_update_date) falls within the
    last `days` days. Deduplicates by company_name at the end.

    Args:
        df: Full tracking DataFrame.
        days: Look-back window in days (default 5).

    Returns:
        Filtered and deduplicated DataFrame.
    """
    last_update = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    new_updates = df[df['most_recent_update'].notna() & (df['most_recent_update'] != '')]

    if 'last_checked' in new_updates.columns:
        new_updates = new_updates.copy()
        new_updates['last_checked'] = pd.to_datetime(new_updates['last_checked'], errors='coerce')
        new_updates = new_updates[new_updates['last_checked'] >= pd.Timestamp(last_update)]
    elif 'update_date' in new_updates.columns:
        new_updates = new_updates.copy()
        new_updates['update_date'] = pd.to_datetime(new_updates['update_date'], errors='coerce')
        new_updates = new_updates[new_updates['update_date'] >= pd.Timestamp(last_update)]
    elif 'most_recent_update_date' in new_updates.columns:
        new_updates = new_updates.copy()
        new_updates['most_recent_update_date'] = pd.to_datetime(new_updates['most_recent_update_date'], errors='coerce')
        new_updates = new_updates[new_updates['most_recent_update_date'] >= pd.Timestamp(last_update)]
    else:
        # No date column available: return empty rather than show stale data
        new_updates = new_updates.iloc[0:0]

    return new_updates.drop_duplicates(subset='company_name', keep='first')


def _filter_irrelevant_tracking_updates(df):
    """Use LLM to verify each tracking update is (1) actually about the tracked
    company and (2) a substantive business signal worth surfacing in the newsletter.
    Entries that fail either check are dropped silently (with a console note).
    On any error the entry is kept, so this is a best-effort filter.
    """
    if df.empty:
        return df

    try:
        from services.openai_api import ask_monty
    except ImportError:
        return df

    keep = []
    for _, row in df.iterrows():
        company = str(row.get('company_name') or '').strip()
        update  = str(row.get('most_recent_update') or '').strip()

        if not company or not update:
            keep.append(True)
            continue

        prompt = (
            "You are filtering a VC firm's weekly portfolio news feed. "
            "Given a tracked company name and a news update summary, return a JSON object with:\n"
            '- "relevant": true if the update is clearly about this specific company. '
            'Relevant signals include: funding, product launch, key hire, partnership, '
            'acquisition, regulatory milestone, or job postings FROM this specific company '
            '(hiring is a signal of growth). false only if the article is about a completely '
            'different company that happens to share a similar name, or is a generic industry '
            'article with no mention of this company.\n'
            '- "reason": one short sentence\n'
            "Return only the JSON object, no other text."
        )
        data = f"Company: {company}\n\nUpdate:\n{update[:1500]}"

        try:
            response = ask_monty(prompt, data, max_tokens=80)
            parsed = json.loads(response.strip())
            if parsed.get('relevant') is False:
                keep.append(False)
            else:
                keep.append(True)
        except Exception as e:
            print(f"  ⚠️  Could not verify tracking update for {company}: {e} — keeping")
            keep.append(True)

    return df[keep].reset_index(drop=True)


def get_tracking():
    """Get tracking updates from Supabase (with CSV fallback).

    Only returns updates from the last 5 days, filtered by last_checked
    (when we last checked the company) or update_date (when the update was found).
    This ensures the weekly update only displays recent tracking items.

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
                filtered = _filter_tracking_updates(df)
                print(f"  Running LLM relevance check on {len(filtered)} tracking entries...")
                return _filter_irrelevant_tracking_updates(filtered)
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

    filtered = _filter_tracking_updates(df)
    print(f"  Running LLM relevance check on {len(filtered)} tracking entries...")
    return _filter_irrelevant_tracking_updates(filtered)

# Load the investment tree root node
def get_pipeline_stats(filter_date=None):
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
    if filter_date is None:
        filter_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

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
    allowed_categories = ['Healthcare', 'Fintech', 'Commerce', 'AI']
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

def get_founders_by_category_path(path, conn=None):
    """
    Query all founders matching a category path from the database.

    Args:
        path: Category path to search for (e.g., "Payments" or "Fintech > Payments")
        conn: Optional existing database connection. If provided it is used as-is
              and will NOT be closed by this function. If None, a new connection
              is opened and closed before returning.

    Returns:
        pandas.DataFrame: DataFrame with all matching founders, or empty DataFrame if none found
    """
    _managed = conn is None
    if _managed:
        conn = get_db_connection()
        if not conn:
            print(f"  ⚠️  Failed to connect to database for path: {path}")
            return pd.DataFrame()

    cursor = None
    try:
        cursor = conn.cursor()
        # Query all founders matching the path (no hardcoded quality filters)
        cursor.execute(
            """
            SELECT * FROM founders
            WHERE founder = true
            AND history = ''
            AND tree_path LIKE %s
            AND (pedigree_passes IS DISTINCT FROM false)
            AND (latestdealtype IS NULL OR latestdealtype NOT ILIKE 'Series%%')
            AND (company_name IS NULL OR company_name NOT LIKE '%%(YC %%')
            AND (
                building_since IS NULL OR building_since = ''
                OR (building_since ~ '^\\d{4}' AND SUBSTRING(building_since FROM 1 FOR 4)::integer >= DATE_PART('year', CURRENT_DATE) - 3)
            )
            """,
            ("%" + path + "%",)
        )
        rows = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        if rows:
            return pd.DataFrame([dict(zip(column_names, row)) for row in rows])
        return pd.DataFrame()

    except Exception as e:
        print(f"  ⚠️  Error querying founders for path {path}: {e}")
        if not _managed:
            # Shared connection died — retry with a fresh one
            try:
                fresh_conn = get_db_connection()
                if fresh_conn:
                    fresh_cursor = fresh_conn.cursor()
                    fresh_cursor.execute(
                        """
                        SELECT * FROM founders
                        WHERE founder = true
                        AND history = ''
                        AND tree_path LIKE %s
                        AND (pedigree_passes IS DISTINCT FROM false)
                        """,
                        ("%" + path + "%",)
                    )
                    rows = fresh_cursor.fetchall()
                    column_names = [desc[0] for desc in fresh_cursor.description]
                    fresh_cursor.close()
                    fresh_conn.close()
                    if rows:
                        return pd.DataFrame([dict(zip(column_names, row)) for row in rows])
            except Exception as retry_err:
                print(f"  ⚠️  Retry also failed for path {path}: {retry_err}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if _managed and conn:
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

def get_all_recent_deals():
    """Load all deals (not just early stage) from the past week."""
    deals_file = 'data/deal_data/all_deals.csv'
    if not os.path.exists(deals_file):
        return pd.DataFrame()
    try:
        deals = pd.read_csv(deals_file, quotechar='"')
    except Exception as e:
        print(f"⚠️  Error reading all_deals file: {e}")
        return pd.DataFrame()
    if deals.empty:
        return pd.DataFrame()
    deals["Date"] = pd.to_datetime(deals["Date"], errors="coerce")
    past_week = datetime.now() - timedelta(days=7)
    deals = deals[deals["Date"] >= past_week].copy()
    deals["Investors"] = deals["Investors"].fillna("").astype(str)
    deals["Category"] = deals["Category"].fillna("Other")
    deals["Vertical"] = deals["Vertical"].fillna("")
    return deals


def fetch_pipeline_companies():
    """Fetch pipeline companies from Notion filtered by active priorities."""
    ACTIVE_PRIORITIES = {"qualifying", "medium", "high", "low", "track"}
    try:
        from services.notion import import_pipeline
        PIPELINE_ID = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
        df = import_pipeline(PIPELINE_ID)
        if df.empty:
            return pd.DataFrame()
        df = df[df["priority"].str.lower().isin(ACTIVE_PRIORITIES)].copy()
        return df
    except Exception as e:
        print(f"⚠️  Could not fetch Notion pipeline: {e}")
        return pd.DataFrame()


def _pick_top_founder_with_pedigree(ranked_df, full_founders_df):
    """Iterate ranked founders top-down, run pedigree check, return first that passes.

    Returns:
        tuple: (founder_dict or None, list of (profile_url, passes, reason))
    """
    from workflows.aviato_processing import check_founder_pedigree
    pedigree_results = []
    for _, row in ranked_df.iterrows():
        profile_url = row.get('profile_url')
        # Use full row data for pedigree (ranked df may have fewer columns)
        if profile_url and not full_founders_df.empty and 'profile_url' in full_founders_df.columns:
            match = full_founders_df[full_founders_df['profile_url'] == profile_url]
            profile_data = match.iloc[0].to_dict() if not match.empty else row.to_dict()
        else:
            profile_data = row.to_dict()

        passes, reason = check_founder_pedigree(profile_data)
        pedigree_results.append((profile_url, passes, reason))
        if passes:
            return row.to_dict(), pedigree_results

    # None passed — skip this category entirely.
    print("    ⚠️  No founder passed pedigree check — skipping category")
    return None, pedigree_results


def _save_weekly_pedigree_results(pedigree_results):
    """Persist pedigree check results to the founders table."""
    if not pedigree_results:
        return
    from services.database import get_db_connection
    conn = get_db_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        for url, passes, reason in pedigree_results:
            if not url:
                continue
            cur.execute(
                "UPDATE founders SET pedigree_passes = %s, pedigree_reason = %s WHERE profile_url = %s",
                (passes, reason, url)
            )
            if not passes:
                cur.execute(
                    "UPDATE founders SET history = 'pedigree_fail' WHERE profile_url = %s AND history = ''",
                    (url,)
                )
        conn.commit()
        passed = sum(1 for _, p, _ in pedigree_results if p)
        failed = len(pedigree_results) - passed
        pass
    except Exception as e:
        print(f"    ⚠️  Error saving pedigree results: {e}")
        conn.rollback()
    finally:
        if 'cur' in locals():
            cur.close()
        conn.close()


# --------------------------------------------------
# Sourcing constants
# --------------------------------------------------
_SOURCING_STATUS_PRIORITY = {
    "Very high":      5,
    "High":           4,
    "High/moderate":  3,
    "Moderate/high":  2,
    "Medium":         1,
    "medium":         1,
    # "New" intentionally omitted — excluded from sourcing
}
_SOURCING_PIPELINE_PRIORITY = 0  # pipeline-only paths checked last
_SOURCING_CATEGORIES = ["Fintech", "Healthcare", "Commerce", "AI"]
_SOURCING_TARGET_PER_CATEGORY = 5
_SOURCING_PEDIGREE_BUDGET_PER_CATEGORY = 30


def _sourcing_build_path_priority_map(tree):
    """Return {node_name: priority} for every taste-tree node with a relevant investment_status."""
    mapping = {}

    def walk(node):
        for name, child in node.items():
            if not isinstance(child, dict):
                continue
            status = (child.get("meta", {}).get("investment_status") or "").strip()
            priority = _SOURCING_STATUS_PRIORITY.get(status, 0)
            if priority > 0:
                if mapping.get(name, -1) < priority:
                    mapping[name] = priority
            if "children" in child:
                walk(child["children"])

    walk(tree)
    return mapping


def _sourcing_build_combined_patterns(pipeline_dict, priority_map):
    """Union of pipeline subcategory names and interest-status node names."""
    allowed = {"Healthcare", "Fintech", "Commerce", "AI"}
    combined = dict(priority_map)  # interest-status entries

    for category, subs in pipeline_dict["subcategories_by_category"].items():
        if category not in allowed:
            continue
        for sub in subs:
            name = sub.get("subcategory")
            if name and pd.notna(name):
                # Pipeline paths only upgrade to PIPELINE_PRIORITY if not already higher
                if combined.get(name, -1) < _SOURCING_PIPELINE_PRIORITY:
                    combined[name] = _SOURCING_PIPELINE_PRIORITY

    return combined  # {node_name: priority}


def _sourcing_fetch_candidates(conn):
    """Fetch all basic-filter candidates from the DB (no tree_path constraint)."""
    return pd.read_sql(
        """
        SELECT *
        FROM founders
        WHERE founder = true
        AND history = ''
        AND (pedigree_passes IS DISTINCT FROM false)
        AND (latestdealtype IS NULL OR latestdealtype NOT ILIKE 'Series%%')
        AND (company_name IS NULL OR company_name NOT LIKE '%%(YC %%')
        AND (
            building_since IS NULL OR building_since = ''
            OR (
                building_since ~ '^\\d{4}'
                AND SUBSTRING(building_since FROM 1 FOR 4)::integer
                    >= DATE_PART('year', CURRENT_DATE) - 3
            )
        )
        """,
        conn,
    )


def _sourcing_filter_and_score(df, combined_patterns):
    """Keep only rows whose tree_path contains a known pattern; attach priority score."""
    if df.empty or "tree_path" not in df.columns:
        return df

    def best_priority(tree_path):
        if not tree_path or (isinstance(tree_path, float) and pd.isna(tree_path)):
            return -1
        tp = str(tree_path)
        best = -1
        for pattern, priority in combined_patterns.items():
            if pattern in tp and priority > best:
                best = priority
        return best

    df = df.copy()
    df["_interest_priority"] = df["tree_path"].apply(best_priority)
    before = len(df)
    df = df[df["_interest_priority"] >= 0].reset_index(drop=True)
    print(f"  Path filter: {before} → {len(df)} founders matching pipeline or High/Medium paths")
    return df


def _sourcing_quality_filters(df):
    """Dedup + location + tenure + title + age filters."""
    from workflows.recommendations import (
        is_us_location,
        founder_tenure_at_company_less_than_2_years,
        is_founder_or_cofounder_at_current_company,
        is_company_less_than_2_years_old,
    )
    from workflows.aviato_processing import estimate_founder_age

    df = df.drop_duplicates(subset=["name"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["company_name"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["profile_url"]).reset_index(drop=True)

    if "building_since" in df.columns:
        df = df[df.apply(
            lambda r: is_company_less_than_2_years_old(r.get("building_since"), max_years=3, row=r), axis=1
        )].reset_index(drop=True)

    if "location" in df.columns:
        df = df[df["location"].apply(is_us_location)].reset_index(drop=True)

    if "all_experiences" in df.columns:
        df = df[df.apply(
            lambda r: founder_tenure_at_company_less_than_2_years(r, max_years=2), axis=1
        )].reset_index(drop=True)

    df = df[df.apply(is_founder_or_cofounder_at_current_company, axis=1)].reset_index(drop=True)

    df = df[df.apply(
        lambda r: (estimate_founder_age(r.to_dict()) or 0) <= 45, axis=1
    )].reset_index(drop=True)

    print(f"  After quality filters: {len(df)} candidates")
    return df


def _sourcing_collect_by_category(df):
    """For each high-level category, sweep its candidates sorted by interest priority
    and pedigree-check until 5 pass (or budget exhausted).

    Returns (flat list of all passing founders, pedigree_log).
    """
    from workflows.aviato_processing import check_founder_pedigree

    if "_interest_priority" in df.columns:
        df = df.sort_values("_interest_priority", ascending=False).reset_index(drop=True)

    all_passing = []
    all_pedigree_log = []

    for category in _SOURCING_CATEGORIES:
        cat_mask = df["tree_path"].apply(
            lambda tp: str(tp).split(">")[0].strip() == category
            if tp and not (isinstance(tp, float)) else False
        )
        cat_df = df[cat_mask].reset_index(drop=True)

        if cat_df.empty:
            print(f"  {category}: no candidates after path filter")
            continue

        print(f"  {category}: {len(cat_df)} candidates — checking pedigree")
        passing = []
        checked = 0

        for _, row in cat_df.iterrows():
            if len(passing) >= _SOURCING_TARGET_PER_CATEGORY:
                break
            if checked >= _SOURCING_PEDIGREE_BUDGET_PER_CATEGORY:
                print(f"    ⚠️  {category}: pedigree budget ({_SOURCING_PEDIGREE_BUDGET_PER_CATEGORY}) reached")
                break

            profile_data = row.to_dict()
            profile_url = profile_data.get("profile_url", "")
            name = profile_data.get("name", "unknown")

            passes, reason = check_founder_pedigree(profile_data)
            checked += 1
            all_pedigree_log.append((profile_url, passes, reason))

            status = "✅" if passes else "❌"
            print(f"    [{checked:>2}] {status}  {name:<30}  — {reason[:70]}")

            if passes:
                profile_data["pedigree_reason"] = reason
                passing.append(profile_data)

        print(f"  {category}: {len(passing)}/{_SOURCING_TARGET_PER_CATEGORY} founders passed ({checked} checked)")
        all_passing.extend(passing)

    return all_passing, all_pedigree_log


def _sourcing_build_recs(passing_founders, tree, pipeline_dict):
    """Organise a flat list of passing founders into the recs dict the HTML generator expects.

    Each founder becomes its own subcategory entry (one card per founder).
    Category and subcategory are inferred from tree_path.
    """
    allowed_categories = {"Healthcare", "Fintech", "Commerce", "AI"}
    filter_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    recs = {}

    for founder in passing_founders:
        tree_path = founder.get("tree_path") or ""
        segments = [s.strip() for s in str(tree_path).split(">")]
        category = segments[0] if segments else "Other"
        subcategory = segments[1] if len(segments) > 1 else (segments[0] if segments else "General")

        # Keep only allowed categories for the newsletter layout
        if category not in allowed_categories:
            category = "Other"

        # Look up interest text
        interest_text = ""
        try:
            nodes = find_nodes_by_name(tree, subcategory)
            for node in nodes:
                val = node.get("meta", {}).get("interest", "")
                if isinstance(val, str) and val.strip():
                    interest_text = val.strip()
                    break
        except Exception:
            pass

        # Look up deal activity
        deal_activity = get_relevant_deals(subcategory, tree, filter_date=filter_date)

        sub_entry = {
            "subcategory": subcategory,
            "count": 0,
            "top_founder": founder,
            "deal_activity": deal_activity,
            "interest": interest_text,
        }

        recs.setdefault(category, []).append(sub_entry)

    return recs


def get_recs():
    with open('data/taste_tree.json', 'r') as f:
        tree = json.load(f)

    pipeline_dict = get_pipeline_stats()

    print("  Building path patterns (pipeline + High/Medium interest)...")
    priority_map = _sourcing_build_path_priority_map(tree)
    combined_patterns = _sourcing_build_combined_patterns(pipeline_dict, priority_map)
    print(f"  Combined patterns: {len(combined_patterns)} node names")

    db_conn = get_db_connection()
    if not db_conn:
        print("❌ Failed to connect to database for sourcing")
        return {}, pipeline_dict

    try:
        print("  Fetching candidates from DB...")
        raw_df = _sourcing_fetch_candidates(db_conn)
        print(f"  Raw candidates: {len(raw_df)}")
    finally:
        db_conn.close()

    scored_df = _sourcing_filter_and_score(raw_df, combined_patterns)
    filtered_df = _sourcing_quality_filters(scored_df)

    if filtered_df.empty:
        print("  No candidates after filtering — sourcing section will be empty")
        return {}, pipeline_dict

    passing_founders, pedigree_log = _sourcing_collect_by_category(filtered_df)
    _save_weekly_pedigree_results(pedigree_log)

    recs = _sourcing_build_recs(passing_founders, tree, pipeline_dict)
    total = sum(len(v) for v in recs.values())
    print(f"  Sourcing complete: {total} founders across {len(recs)} categories")

    return recs, pipeline_dict


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
        from workflows.aviato_processing import map_aviato_to_schema
        from services.openai_api import generate_talent_description
        
        profiles = get_profile_recommendations(limit=3)
        
        if not profiles:
            return [], []
        
        # Generate personalized descriptions for each profile
        for profile in profiles:
            try:
                # Map enriched profile to get all_experiences
                mapped_profile = map_aviato_to_schema(profile)
                all_experiences = mapped_profile.get("all_experiences", [])
                
                # Get person's name for context
                person_name = profile.get("fullName") or mapped_profile.get("name", "this person")
                
                # Generate personalized description if we have experiences
                if all_experiences and len(all_experiences) > 0:
                    personalized_desc = generate_talent_description(
                        all_experiences=all_experiences,
                        person_name=person_name
                    )
                    if personalized_desc:
                        profile["personalized_description"] = personalized_desc
                    else:
                        profile["personalized_description"] = None
                else:
                    profile["personalized_description"] = None
            except Exception as e:
                print(f"Warning: could not generate personalized description for profile: {e}")
                profile["personalized_description"] = None
        
        profile_ids = [p['id'] for p in profiles]
        
        return profiles, profile_ids
    except Exception as e:
        print(f"Warning: could not get profile recommendations: {e}")
        return [], []

def generate_greeting_with_openai(fallback=None):
    """
    Call OpenAI to generate a brief Friday greeting. No newsletter context needed:
    just happy Friday / weekend vibes, a quick mention of what's in the issue, and something funny.
    Returns fallback text if the API call fails.
    """
    if fallback is None:
        fallback = (
            "Happy Friday everyone! This week you've got an overview of deals, tracking news, "
            "sourcing results, and some talent recommendations—plus one joke that may or may not land. "
            "Have a great weekend!\n\n - Monty"
        )
    try:
        from services.openai_api import ask_monty
    except ImportError:
        return fallback

    prompt = (
        "You write the opening greeting for a weekly Friday newsletter from Monty (Montage Ventures). "
        "Do NOT reference any specific companies, deals, or categories—keep it generic. "
        "Mention that the newsletter includes: an overview of deals, tracking news, talent recommendations, "
        "and sourcing results. Add something light and funny (a short joke, pun, or playful line). "
        "IMPORTANT: Vary the humor each time—come up with something fresh and creative. "
        "Avoid overused jokes (e.g. the mosquito one, 'TGIF', 'case of the Mondays'). "
        "Keep it to 3-5 short sentences. End with a weekend sign-off and ' - Monty'. "
        "Output only the greeting, no preamble or quotes."
    )
    # Vary the data slightly each run so the model gets different context (helps avoid repeating the same joke)
    data = (
        f"Sections in this newsletter: deals overview, tracking news, talent recommendations, sourcing results. "
        f"Write a friendly Friday intro."
        f"(Generated: {datetime.now().strftime('%A %B %d')})"
    )
    try:
        greeting = ask_monty(prompt, data, max_tokens=250)
        if greeting and greeting.strip():
            return greeting.strip()
    except Exception as e:
        print(f"  ⚠️  OpenAI greeting generation failed: {e}, using fallback")
    return fallback


# Load the tree for deal lookups
def main(no_send=False):
    from services.weekly_formatting import generate_html
    from services.google_client import send_email
    
    print("Gathering recent deals...")
    recent_deals = get_recent_deals()
    print(f"Found {len(recent_deals)} recent deals")

    print("\nLoading all recent deals for pipeline crossref + investor activity...")
    all_recent_deals = get_all_recent_deals()
    print(f"Found {len(all_recent_deals)} total deals this week")

    print("\nFetching Notion pipeline companies...")
    pipeline_companies = fetch_pipeline_companies()
    print(f"Found {len(pipeline_companies)} active pipeline companies")
    
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
        profile_recs = []
    
    print("\nGenerating greeting (OpenAI)...")
    default_greeting = (
        "Happy Friday everyone! This week you've got an overview of deals, tracking news, "
        "sourcing results, and some talent recommendations—plus one joke that may or may not land. But you're super excited about the sourcing results this week! "
        "Have a great weekend!\n\n - Monty"
    )
    greeting_text = generate_greeting_with_openai(fallback=default_greeting)
    print("\nGenerating HTML email...")
    html_output = generate_html(recent_deals, tracking, recs, pipeline_dict, greeting_text, profile_recs=profile_recs,
                               all_deals_df=all_recent_deals, pipeline_companies_df=pipeline_companies)
    
    # Save to file
    output_path = 'data/weekly_update_output.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_output)
    
    print(f"\n✅ Weekly update generated successfully!")
    print(f"📄 Saved to: {output_path}")

    # Send email (skipped in dry-run mode)
    if no_send:
        print("\n⏭  --no-send: skipping email send and marking steps")
    else:
        send_email(html_output)

        # Mark recommended founders/companies to avoid repeats
        mark_as_recommended(recs)

        # Mark recommended profiles to avoid repeats
        if profile_ids:
            mark_profiles_as_recommended(profile_ids)
            print(f"Marked {len(profile_ids)} profiles as recommended")
    
    return html_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-send", action="store_true",
                        help="Generate HTML but skip sending the email and marking recommendations")
    args = parser.parse_args()
    main(no_send=args.no_send)

