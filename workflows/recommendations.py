"""
Recommendations System for Monty
=================================

This module handles personalized founder/company recommendations sent to team members via email.

Main Functions:
--------------
1. send_extra_recs() - Main function to send recommendations to all users
   - Fetches new recommended profiles from the database
   - Matches them to each user's areas of interest (from taste tree in Supabase/JSON)
   - Sends personalized emails with top 3 recommendations per user
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
- Gmail API credentials (GOOGLE_CREDENTIALS_BASE64) for sending email
- Database must have 'founders' table with tree_result and history columns
- Taste tree (in Supabase or data/taste_tree.json) must have 'montage_lead' metadata for user assignments
- For Supabase: SUPABASE_URL and SUPABASE_KEY environment variables must be set
"""

import json
import sys
import os
import pandas as pd
import time
import warnings
import re
import html as html_module
from datetime import datetime
from psycopg2 import sql

# Add tests directory to path so prepare_test_features is importable for ranking
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'tests'))

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=UserWarning)
try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass  # Removed in pandas 2.0+

# Notion database IDs used for pipeline/tracked/passed filtering
PIPELINE_ID = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
TRACKED_ID = "912974853b494f98a5652fcbff3ad795"
PASSED_ID = "bc5f875961234aa6aa4b293cf1915ac2"

# Email address mapping for recommendation delivery
email_map = {
    "Todd": "todd@montageventures.com",
    "Matt": "matt@montageventures.com",
    "Daphne": "daphne@montageventures.com",
    "Connie": "connie@montageventures.com",
    "Nia": "nia@montageventures.com",
    "Matthildur": "matthildur@montageventures.com",
}

# When test=True, one email is sent to this address only (for verification)
TEST_EMAIL_RECIPIENT = "matthildur@montageventures.com"

# Vertical history per user — used for Exa discovery search.
# Each user maps to a list of verticals in chronological order.
# The LAST item is the current vertical that will be used for the next send.
# To update: append a new vertical to the user's list.
USER_VERTICALS = {
    "Todd": [
        "banking and credit for consumers",
        "AI native engineering services",
        "AI for payments, fraud, and financial infra"
    ],
    "Matt": [
        "financial infrastructure",
        "banking and credit",
        "AI-native trading / investment"
    ],
    "Daphne": [
        "AI drug discovery",
        "Global retail enablement infrastructure",
        "AI native ecommerce infra"
    ],
    "Connie": [
        "AI for accounting",
        "agentic commerce infrastructure",
        "AI for tax & accounting / compliance"
    ],
    "Nia": [
        "AI for mathematical reasoning",
        "Neuroscience infrastructure (modeling and simulating brain)",
        "Computational neuroscience / CNS drug discovery & development"
    ],
    "Matthildur": [
        "AI for engineering physics simulation",
        "Robotics and physical AI"
    ],
}


def _current_vertical(user: str) -> str | None:
    """Return the current (last) vertical for a user, or None if not configured."""
    verticals = USER_VERTICALS.get(user)
    if not verticals:
        return None
    return verticals[-1] if isinstance(verticals, list) else verticals

# User category filter configuration
# Maps username to a list of strings that must appear in the tree_path
# If None or empty list, no filtering is applied (user gets all their assigned categories)
# Multiple filter strings are supported - categories matching ANY of the strings will be included
# Filter strings can be ANY substring in the tree_path - they don't need to be nodes assigned to the user
# However, filtering only applies to categories already assigned to the user in the taste tree
# Example: {"Connie": ["Commerce"]} means Connie only gets categories with "Commerce" in the tree_path
# Example: {"Todd": ["Fintech", "Payments"]} means Todd gets categories with "Fintech" OR "Payments" in the tree_path
user_category_filters = {
    # "Connie": ["Commerce"],  # Removed: Connie now gets all her assigned categories
    #"Nia": ["Foundation & Frontier Models", "Fintech"]
    # "Todd": ["Fintech", "Payments"],  # Multiple filter values - matches categories with either string
    # "Daphne": ["Healthcare", "Biotech"],  # Multiple filter values example
    # Add more users as needed
}

def is_company_less_than_2_years_old(building_since, max_years=2):
    """
    Check if a company has been building for less than the specified number of years.
    
    Args:
        building_since: Date string in various formats (YYYY-MM-DD, "Month YYYY", YYYY, etc.) or None
        max_years: Maximum years to consider (default: 2)
    
    Returns:
        bool: True if company is less than max_years old, False otherwise (or if date is invalid/missing)
    """
    if pd.isna(building_since) or building_since is None or building_since == '':
        # If no date available, include it (don't filter out)
        return True
    
    try:
        building_since_str = str(building_since).strip()
        date_obj = None
        
        # Try to parse as ISO date (YYYY-MM-DD)
        if re.match(r'^\d{4}-\d{2}-\d{2}', building_since_str):
            date_obj = datetime.strptime(building_since_str[:10], '%Y-%m-%d')
        # Try to parse as "Month YYYY" format (e.g., "Dec 2019", "December 2019")
        elif re.match(r'^[A-Za-z]+\s+\d{4}', building_since_str):
            # Try abbreviated month first, then full month name
            for fmt in ['%b %Y', '%B %Y']:
                try:
                    date_obj = datetime.strptime(building_since_str, fmt)
                    break
                except ValueError:
                    continue
        # Try to parse as just year (YYYY)
        elif re.match(r'^\d{4}$', building_since_str):
            date_obj = datetime.strptime(building_since_str, '%Y')
        # Try other common formats
        else:
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    date_obj = datetime.strptime(building_since_str, fmt)
                    break
                except ValueError:
                    continue
        
        if date_obj is None:
            # Could not parse date, include it (don't filter out)
            return True
        
        # Calculate years since that date
        now = datetime.now()
        delta = now - date_obj
        years = delta.days / 365.25
        
        # Return True if less than max_years old
        return years < max_years
        
    except (ValueError, TypeError, AttributeError):
        # If parsing fails, include it (don't filter out)
        return True


def founder_tenure_at_company_less_than_2_years(row, max_years=2):
    """
    Check if the founder has been at their current company for less than max_years.
    Uses all_experiences JSON to find the company and parse start_date.
    
    Args:
        row: DataFrame row or dict with company_name, all_experiences, and optionally company_index
        max_years: Maximum years at company (default: 2)
    
    Returns:
        bool: True if tenure < max_years (or if we cannot determine), False if 2+ years
    """
    from services.deal_processing import normalize_company_name
    
    company_name = row.get('company_name')
    if pd.isna(company_name) or not company_name:
        return True  # Cannot determine, include
    
    all_exp = row.get('all_experiences')
    if all_exp is None or (isinstance(all_exp, float) and pd.isna(all_exp)):
        return True  # No experiences, include
    
    if isinstance(all_exp, str):
        try:
            all_exp = json.loads(all_exp) if all_exp.strip() else []
        except (json.JSONDecodeError, TypeError):
            return True
    
    if not isinstance(all_exp, list) or len(all_exp) == 0:
        return True
    
    target_normalized = normalize_company_name(str(company_name))
    if not target_normalized:
        return True
    
    # Find matching experience: use company_index if available, else search by company name
    exp = None
    company_index = row.get('company_index')
    if company_index is not None and not pd.isna(company_index):
        try:
            idx = int(company_index)
            if 0 <= idx < len(all_exp):
                e = all_exp[idx]
                if isinstance(e, dict):
                    exp_name = e.get('company_name') or e.get('company', '')
                    if normalize_company_name(str(exp_name)) == target_normalized:
                        exp = e
        except (ValueError, TypeError):
            pass
    
    if exp is None:
        for e in all_exp:
            if not isinstance(e, dict):
                continue
            exp_name = e.get('company_name') or e.get('company', '')
            if normalize_company_name(str(exp_name)) == target_normalized:
                exp = e
                break
    
    if exp is None:
        return True  # Could not find matching experience, include
    
    start_date = exp.get('start_date')
    if not start_date or (isinstance(start_date, float) and pd.isna(start_date)):
        return True  # No start date, include
    
    try:
        start_str = str(start_date).strip()
        date_obj = None
        if re.match(r'^\d{4}$', start_str):
            date_obj = datetime.strptime(start_str, '%Y')
        elif re.match(r'^\d{4}-\d{2}', start_str):
            if len(start_str) >= 7:
                date_obj = datetime.strptime(start_str[:7], '%Y-%m')
            else:
                date_obj = datetime.strptime(start_str[:10], '%Y-%m-%d')
        elif re.match(r'^[A-Za-z]+\s+\d{4}', start_str):
            for fmt in ['%b %Y', '%B %Y']:
                try:
                    date_obj = datetime.strptime(start_str, fmt)
                    break
                except ValueError:
                    continue
        
        if date_obj is None:
            return True
        
        now = datetime.now()
        delta = now - date_obj
        years = delta.days / 365.25
        return years < max_years
    except (ValueError, TypeError, AttributeError):
        return True


# Matches "Founding Engineer", "Founding Advisor", "Founding Team Member", etc. —
# roles that use "founding" as an adjective for a non-founder position.
_NON_FOUNDER_TITLE_RE = re.compile(
    r'\bfounding\s+(engineer|advisor|adviser|team\s*member|member|developer|designer|'
    r'scientist|researcher|analyst|product\s*manager|pm|intern|associate)\b',
    re.IGNORECASE
)


def is_founder_or_cofounder_at_current_company(row):
    """Check that the person's current role is genuinely a founder/co-founder title.

    Only returns False when a title is available and clearly indicates a non-founder
    "founding" role (e.g. "Founding Engineer", "Founding Advisor"). Returns True when
    the title is absent (benefit of the doubt) or when it's consistent with being a
    founder (e.g. "Co-Founder", "Founder & CEO", "CEO").

    Args:
        row: DataFrame row or dict-like with 'title' and/or 'all_experiences'.

    Returns:
        bool: False only if a clearly non-founder founding title is found.
    """
    # Prefer the direct title field on the DB row
    title = row.get('title') or ''

    # Fall back to the most-recent entry in all_experiences
    if not title:
        all_exp = row.get('all_experiences')
        if all_exp is not None and not (isinstance(all_exp, float) and pd.isna(all_exp)):
            if isinstance(all_exp, str):
                try:
                    all_exp = json.loads(all_exp) if str(all_exp).strip() else []
                except (json.JSONDecodeError, TypeError):
                    all_exp = []
            if isinstance(all_exp, list) and all_exp:
                first = all_exp[0]
                if isinstance(first, dict):
                    title = (
                        first.get('position') or
                        first.get('title') or
                        first.get('role') or ''
                    )

    if not title:
        return True  # No title data — give benefit of the doubt

    if _NON_FOUNDER_TITLE_RE.search(str(title)):
        return False  # Confirmed non-founder "founding" role

    return True


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


def _select_top_with_pedigree(profiles_ranked, new_profiles, target=3):
    """Iterate ranked profiles top-down, running pedigree check on each.
    Returns (selected_df, pedigree_results) where pedigree_results is a list
    of (profile_url, passes, reason) for every profile checked."""
    from workflows.aviato_processing import check_founder_pedigree
    selected_rows = []
    pedigree_results = []
    for _, row in profiles_ranked.iterrows():
        if len(selected_rows) >= target:
            break
        profile_url = row.get('profile_url')
        full_match = new_profiles[new_profiles['profile_url'] == profile_url]
        profile_data = full_match.iloc[0].to_dict() if not full_match.empty else row.to_dict()
        passes, reason = check_founder_pedigree(profile_data)
        pedigree_results.append((profile_url, passes, reason))
        if passes:
            selected_rows.append(row)

    result = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame(columns=profiles_ranked.columns)
    return result, pedigree_results


def _save_pedigree_results(pedigree_results):
    """Persist pedigree check results to DB (both passes and failures).
    Failures also get history = 'pedigree_fail' to exclude from future queries."""
    if not pedigree_results:
        return
    from services.database import get_db_connection
    conn = get_db_connection()
    if not conn:
        return
    try:
        cur = conn.cursor()
        failed = 0
        passed = 0
        for url, passes, reason in pedigree_results:
            cur.execute(
                "UPDATE founders SET pedigree_passes = %s, pedigree_reason = %s WHERE profile_url = %s",
                (passes, reason, url)
            )
            if not passes:
                cur.execute(
                    "UPDATE founders SET history = 'pedigree_fail' WHERE profile_url = %s AND history = ''",
                    (url,)
                )
                failed += 1
            else:
                passed += 1
        conn.commit()
        pass
    except Exception as e:
        print(f"  ⚠️  Error saving pedigree results: {e}")
        conn.rollback()
    finally:
        if 'cur' in locals():
            cur.close()
        conn.close()


def find_new_recs(username, test=True, return_html_only=False):
    """Find new recommendations for a specific user (for testing/preview purposes).

    Args:
        username: Name of the user to find recommendations for
        test: If True, prints recommendations without sending email. If False, sends to the user's email.
        return_html_only: If True, build and return HTML body without sending or printing; used by preview script.

    Returns:
        When return_html_only=True, returns the HTML body string or None if no recommendations.
    """
    from services.tree import get_nodes_and_names
    from services.path_mapper import get_all_matching_old_paths
    from services.model_loader import load_ranker_model
    from services.ranker_inference import rank_profiles

    new_profiles = _fetch_and_filter_founders()
    if new_profiles is None:
        return None

    columns_to_share = [
        'name', 'company_name', 'company_website', 'profile_url',
        'tree_thesis', 'product', 'market', 'tree_path',
        'past_success_indication_score', 'tree_result',
    ]
    recs = new_profiles[columns_to_share].copy()
    recs = recs.drop_duplicates(subset=['company_name'])
    recs = recs.drop_duplicates(subset=['name'])
    recs = recs.drop_duplicates(subset=['profile_url'])
    recs = recs.reset_index(drop=True)

    # Load ranker for consistent ranking with send_extra_recs
    ranker_inference = load_ranker_model()
    prepare_test_features = None
    if ranker_inference is not None:
        try:
            from test_models import prepare_test_features
        except ImportError:
            print("⚠️  Could not import prepare_test_features, skipping ranker")
            ranker_inference = None

    # Load user interests from Supabase (with fallback to JSON)
    if not return_html_only:
        print("Loading user interests from taste tree...")
    nodes_and_names = get_nodes_and_names(use_supabase=True)

    recs['top_category'] = recs['tree_path'].apply(lambda x: x.split(' > ')[0] if pd.notna(x) else '')

    for user, categories in nodes_and_names.items():
        if user != username:
            continue
        if not return_html_only:
            print(f"\nProfiles for {user}")
            print("-" * 60)

        # Apply category filter if configured for this user
        categories = filter_categories_by_tree_path(categories, user)

        if not categories:
            print(f"No categories remaining after filtering for {user}")
            return None

        # Build profile pool, tracking the depth and the matched user-category per profile.
        # deeper match = more specific alignment with what this user cares about.
        profiles_chunks = []
        for category in categories:
            old_paths = get_all_matching_old_paths(category)
            cat_depth = category.count(' > ') + 1

            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
            for old_path in old_paths:
                old_matching = recs[recs['tree_path'].str.contains(old_path, na=False, regex=False)]
                matching = pd.concat([matching, old_matching])

            matching = matching.drop_duplicates(subset=['name', 'company_name'])
            if not matching.empty:
                matching = matching.copy()
                matching['match_depth'] = cat_depth
                matching['matched_category'] = category  # user interest category that triggered this match
                profiles_chunks.append(matching)

        if not profiles_chunks:
            print(f"No matching profiles found for {user}")
            return None

        profiles = pd.concat(profiles_chunks)
        # Keep each profile's deepest match (sort desc before dedup preserves the best row)
        profiles = profiles.sort_values('match_depth', ascending=False)
        profiles['category'] = profiles['tree_path'].apply(
            lambda x: x.split(' > ')[-2] if len(x.split(' > ')) > 1 else x
        )
        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)

        if len(profiles) == 0:
            print(f"No matching profiles found for {user}")
            return None

        # Rank with model (or fall back to heuristic sort)
        if ranker_inference is not None:
            profiles_full = profiles[['name', 'company_name']].merge(
                new_profiles, on=['name', 'company_name'], how='left'
            )
            profiles_ranked = rank_profiles(profiles_full, ranker_inference, prepare_test_features)
            if 'ranker_score' in profiles_ranked.columns:
                score_map = dict(zip(
                    zip(profiles_ranked['name'], profiles_ranked['company_name']),
                    profiles_ranked['ranker_score']
                ))
                profiles['ranker_score'] = profiles.apply(
                    lambda r: score_map.get((r['name'], r['company_name']), 0.0), axis=1
                )
                profiles.sort_values(by=['ranker_score', 'match_depth'], ascending=[False, False], inplace=True)
            else:
                profiles.sort_values(
                    by=['match_depth', 'past_success_indication_score'],
                    ascending=[False, False], inplace=True
                )
        else:
            profiles.sort_values(
                by=['match_depth', 'past_success_indication_score'],
                ascending=[False, False], inplace=True
            )

        # Take top 3 (with iterative pedigree check)
        profiles, pedigree_results = _select_top_with_pedigree(profiles, new_profiles, target=3)
        _save_pedigree_results(pedigree_results)

        if not return_html_only:
            print(f"Found {len(profiles)} recommendations:\n")

        # Build category string from the matched user-interest categories (not the profile paths).
        # Use the last (most specific) node of each matched category, deduplicated.
        seen_cats: set = set()
        category_parts = []
        for cat in profiles['matched_category']:
            label = cat.split(' > ')[-1]
            if label not in seen_cats:
                seen_cats.add(label)
                category_parts.append(label)
        category_string = ", ".join(category_parts)

        # Build greeting and message (console preview uses Slack-style text)
        greeting_text = f"Hey {user}! I came across these profiles in {category_string} that I wanted to share with you."
        message_lines = [greeting_text]

        for _, row in profiles.iterrows():
            line = (
                f"• <{fix_profile_url(row['profile_url'])}|*{row['name']}*> at {fix_company_url(row['company_website'], row['company_name'])}\n"
                f"    {row['product']}"
            )
            message_lines.append(line)

            profile_link = fix_profile_url(row['profile_url'])
            if not return_html_only:
                print(f"  - {row['name']} at {row['company_name']} ({row['tree_result']})")
                print(f"    Profile: {profile_link}")
                print(f"    Category: {row['tree_path']}")
                print()

        message = "\n\n".join(message_lines)
        html_body = build_recommendation_email_html(profiles, greeting_text)

        if return_html_only:
            return html_body

        if test:
            print("=" * 60)
            print("TEST MODE - Message preview:")
            print("=" * 60)
            print(message)
            print("=" * 60)
            print("TEST MODE - skipping email send.")
        else:
            to_email = email_map.get(user)
            if to_email:
                print("=" * 60)
                print(f"Sending recommendation email to {user} ({to_email})")
                send_recommendation_email(to_email, "Monty recommendations for you", html_body)
                print("✅ Message sent!")
            else:
                print(f"⚠️  No email found for {user}")


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


def _company_url_for_html(url, company_name):
    """Return (url, display_text) for company link in HTML. If no URL, return (None, company_name)."""
    if not url or url == '' or pd.isna(url):
        return None, company_name
    if not url.startswith('http'):
        url = f"https://{url}"
    return url, company_name


# ---------------------------------------------------------------------------
# Exa vertical discovery — same algorithm as tests/test_exa_vertical_search.py
# ---------------------------------------------------------------------------

def _exa_pre_filter(r: dict) -> bool:
    """Keep only profiles whose Exa title or highlight contains 'founder'/'co-founder'."""
    combined = (r["title"] + " " + " ".join(r["highlights"])).lower()
    return "founder" in combined or "co-founder" in combined


def _llm_relevance_filter(r: dict, vertical: str) -> tuple[bool, str]:
    """Use GPT-4o-mini to check if this Exa profile is actually relevant to the vertical.

    Runs before Aviato enrichment so we only use the Exa title + highlights.
    Returns (keep: bool, reason: str).
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        text = f"Title: {r['title']}\n\nHighlights:\n" + "\n".join(r["highlights"])
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a venture capital analyst screening founder profiles. "
                        "Given a founder's LinkedIn title and highlights, decide if: "
                        "(1) they appear to be founding or co-founding an early-stage startup (not just an employee or executive at an established company), AND "
                        "(2) that startup is relevant or adjacent to the specified vertical. "
                        "Be lenient on relevance — if there is a reasonable connection, lean YES. "
                        "Only answer NO if the profile is clearly not a startup founder, or clearly unrelated to the vertical. "
                        "Reply with YES or NO followed by a short reason on the same line. "
                        "Example: YES — founding an AI simulation startup for engineering workflows"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Vertical: {vertical}\n\n{text}",
                },
            ],
            max_tokens=60,
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip()
        keep = answer.upper().startswith("YES")
        return keep, answer
    except Exception as e:
        # If LLM call fails, let the profile through rather than silently dropping it
        return True, f"LLM filter skipped ({e})"

def _exa_post_filter(mapped: dict) -> tuple[bool, str]:
    """Reject non-US or stale founder roles after Aviato enrichment."""
    if not mapped.get("founder"):
        return False, "not a founder"
    location = mapped.get("location") or ""
    _allowed = ("united states", "united kingdom", ", uk", "uk,")
    if location and not any(k in location.lower() for k in _allowed):
        return False, f"non-US/UK location: {location}"
    cutoff_year = datetime.now().year - 3
    idx = mapped.get("company_index", 0)
    all_exp = mapped.get("all_experiences") or []
    if idx < len(all_exp):
        start = (all_exp[idx].get("start_date") or "")[:4]
        try:
            if int(start) < cutoff_year:
                return False, f"founder role started {start}"
        except (ValueError, TypeError):
            pass
    return True, "ok"

def get_exa_profiles_for_vertical(vertical: str, num_results: int = 10) -> list:
    """Run Exa search for a vertical and return enriched founder dicts (max 3).

    Uses the same flow as tests/test_exa_vertical_search.py:
      Exa search → pre-filter → Aviato /person/enrich → map_aviato_to_schema
      → post-filter → monty_enrich_profile
    """
    try:
        from exa_py import Exa
    except ImportError:
        print("  ⚠️  exa_py not installed — skipping Exa search")
        return []

    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        print("  ⚠️  EXA_API_KEY not set — skipping Exa search")
        return []

    from workflows.aviato_processing import get_linkedin_id, enrich_profile, map_aviato_to_schema, monty_enrich_profile, check_founder_pedigree
    from services.database import clean_linkedin_url

    query = f"Founders of {vertical} startups that were founded within the last 2 years in the US"
    exa = Exa(api_key)
    response = None
    for _attempt in range(3):
        try:
            response = exa.search(
                query,
                category="people",
                type="deep",
                num_results=num_results,
                contents={"highlights": {"max_characters": 2000}},
            )
            break
        except Exception as e:
            if _attempt < 2:
                time.sleep(5)
            else:
                print(f"  ⚠️  Exa search failed after 3 attempts: {e}")
                return []

    results = [
        {
            "url":        getattr(r, "url", None) or "",
            "title":      getattr(r, "title", None) or "",
            "highlights": getattr(r, "highlights", []) or [],
        }
        for r in response.results
        if "linkedin.com/in/" in (getattr(r, "url", None) or "")
    ]

    enriched = []
    for r in results:
        if not _exa_pre_filter(r):
            continue
        keep, _ = _llm_relevance_filter(r, vertical)
        if not keep:
            continue
        url_clean   = clean_linkedin_url(r["url"])
        linkedin_id = get_linkedin_id(r["url"])
        raw = enrich_profile(linkedin_id, url_clean)
        if raw is None:
            continue
        mapped = map_aviato_to_schema(raw)
        mapped["profile_url"] = url_clean
        mapped["source"]      = f"exa_vertical:{vertical}"
        keep, reason = _exa_post_filter(mapped)
        if not keep:
            continue
        mapped = monty_enrich_profile(mapped)
        passes, reason = check_founder_pedigree(mapped)
        if not passes:
            pass
            continue
        mapped["access_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mapped = _run_exa_pipeline(mapped)
        enriched.append(mapped)

    return enriched


def _run_exa_pipeline(profile: dict) -> dict:
    """Run the full AI enrichment pipeline on an Exa-sourced profile dict.

    Called after monty_enrich_profile() to add product/market, verticals,
    company/school tags, technical flag, repeat_founder, and tree_path.
    Any step that fails is silently skipped so the profile is still returned.
    """
    from services.profile_analysis import extract_info_from_website, extract_info_from_description_only, is_technical_founder
    from services.ai_parsing import get_past_notable_company, get_past_notable_education, generate_verticals
    from services.tree import tree_analysis

    # tree_analysis expects location_1; map_aviato_to_schema only sets location
    profile.setdefault("location_1", profile.get("location") or "")
    profile.setdefault("product", "")
    profile.setdefault("market", "")
    profile.setdefault("funding", "")

    # 1. Format funding string (tree_analysis uses it for the funding_filter check)
    try:
        amount = float(profile.get("fundingamount") or 0)
        deal_type = profile.get("latestdealtype") or ""
        if amount and deal_type:
            if amount >= 1_000_000_000:
                fmt = f"{amount / 1_000_000_000:.1f}B"
            elif amount >= 1_000_000:
                fmt = f"{amount / 1_000_000:.1f}M"
            elif amount >= 1_000:
                fmt = f"{amount / 1_000:.1f}K"
            else:
                fmt = str(int(amount))
            profile["funding"] = f"US$ {fmt}, {deal_type}"
    except (ValueError, TypeError):
        pass

    # 2. Product and market from website / description
    try:
        website = profile.get("company_website") or ""
        description = profile.get("description_1") or ""
        if website and website != "Not available":
            info = extract_info_from_website(website, description)
        elif description and description != "Not available":
            info = extract_info_from_description_only(description)
        else:
            info = {}
        profile["product"] = (info.get("product_description") or "")[:500]
        profile["market"]  = (info.get("market_description") or "")[:500]
    except Exception:
        pass

    # 3. AI-generated verticals
    try:
        profile["verticals"] = generate_verticals(profile, use_json=True)
    except Exception:
        pass

    # 4. Company / school tags
    try:
        profile["company_tags"] = get_past_notable_company(profile, use_json=True)
    except Exception:
        pass
    try:
        profile["school_tags"] = get_past_notable_education(profile)
    except Exception:
        pass

    # 5. Technical / repeat-founder flags
    try:
        profile["technical"] = is_technical_founder(profile)
    except Exception:
        pass
    try:
        all_exp = profile.get("all_experiences") or []
        if isinstance(all_exp, str):
            import json as _json
            all_exp = _json.loads(all_exp)
        founder_count = sum(
            1 for e in all_exp
            if isinstance(e, dict) and any(
                kw in (e.get("position") or "").lower()
                for kw in ("founder", "co-founder", "ceo")
            )
        )
        profile["repeat_founder"] = founder_count > 1
    except Exception:
        pass

    # 6. Tree analysis — places profile in the investment thesis tree
    try:
        profile = tree_analysis(profile)
    except Exception:
        pass

    return profile


def build_recommendation_email_html(profiles, greeting_text, exa_profiles=None, exa_vertical=None, exa_intro=None):
    """Build HTML body for a recommendation email.

    Args:
        profiles: DataFrame with classic recommendations
        greeting_text: Opening greeting line
        exa_profiles: Optional list of Aviato-enriched dicts from Exa discovery
        exa_vertical: The vertical string used for the Exa search (for the intro sentence)

    Returns:
        str: HTML fragment safe to embed in an email body
    """
    parts = []
    if greeting_text:
        parts += [
            '<p style="margin-bottom: 1em;">',
            html_module.escape(greeting_text),
            '</p>',
        ]
    parts.append('<ul style="margin: 0; padding-left: 1.5em;">')
    for _, row in (profiles.iterrows() if profiles is not None and len(profiles) > 0 else iter([])):
        profile_url = fix_profile_url(row['profile_url'])
        company_url, company_name = _company_url_for_html(row.get('company_website'), row.get('company_name', ''))
        name_esc = html_module.escape(str(row.get('name', '')))
        company_esc = html_module.escape(str(company_name))
        product_esc = html_module.escape(str(row.get('product', '')))
        name_link = f'<a href="{html_module.escape(profile_url)}"><strong>{name_esc}</strong></a>'
        if company_url:
            company_link = f'<a href="{html_module.escape(company_url)}"><strong>{company_esc}</strong></a>'
        else:
            company_link = f'<strong>{company_esc}</strong>'
        parts.append(
            f'<li style="margin-bottom: 0.75em;">{name_link} at {company_link}<br/>'
            f'<span style="color: #555;">{product_esc}</span></li>'
        )
    parts.append('</ul>')

    # Exa section — only appended when profiles were found
    if exa_profiles:
        if exa_intro:
            intro = html_module.escape(exa_intro)
        else:
            vertical_label = html_module.escape(exa_vertical or "this vertical")
            intro = (
                f"Additionally, since I know you're looking into {vertical_label}, "
                f"I just found these profiles and wanted to share in case it's interesting."
            )
        parts.append(f'<p style="margin-top: 2em; margin-bottom: 0.75em;">{intro}</p>')
        parts.append('<ul style="margin: 0; padding-left: 1.5em;">')
        for p in exa_profiles:
            profile_url = fix_profile_url(p.get("profile_url") or p.get("linkedin_url") or "")
            company_url, company_name = _company_url_for_html(
                p.get("company_website"), p.get("company_name") or ""
            )
            name_esc    = html_module.escape(str(p.get("name") or ""))
            company_esc = html_module.escape(str(company_name))
            desc_esc    = html_module.escape(str(p.get("product") or p.get("description_1") or "")[:300])
            name_link   = f'<a href="{html_module.escape(profile_url)}"><strong>{name_esc}</strong></a>'
            if company_url:
                company_link = f'<a href="{html_module.escape(company_url)}"><strong>{company_esc}</strong></a>'
            else:
                company_link = f'<strong>{company_esc}</strong>'
            parts.append(
                f'<li style="margin-bottom: 0.75em;">{name_link} at {company_link}<br/>'
                f'<span style="color: #555;">{desc_esc}</span></li>'
            )
        parts.append('</ul>')

    return '\n'.join(parts)


def send_recommendation_email(to_email, subject, html_body):
    """Send a recommendation email using the Gmail API.
    
    Args:
        to_email: Recipient email address (str or list)
        subject: Email subject line
        html_body: HTML body of the email
    
    Returns:
        bool: True if sent successfully
    """
    from services.google_client import send_html_email
    result = send_html_email(to_email, subject, html_body)
    if result:
        print(f"✅ Recommendation email sent to {to_email}")
        return True
    print(f"❌ Failed to send recommendation email to {to_email}")
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
        date_str = datetime.now().strftime("%Y-%m-%d")
        history_value = f"recommended train {date_str}"
        for table_name in ["founders", "stealth_founders"]:
            query = sql.SQL(
                "UPDATE {} SET history = %s WHERE name = ANY(%s) OR company_name = ANY(%s);"
            ).format(sql.Identifier(table_name))
            cur.execute(query, (history_value, list(all_names), list(companies_set)))
            print(f"Updated {cur.rowcount} records in {table_name}.")
        conn.commit()
        print(f"✅ Successfully marked profiles as '{history_value}'.")
    except Exception as e:
        print(f"❌ Error updating records: {e}")
        conn.rollback()
    finally:
        if 'cur' in locals():
            cur.close()
        if conn:
            conn.close()


def _fetch_and_filter_founders():
    """Fetch all active founders from the database and apply standard quality filters.

    Shared by find_new_recs() and send_extra_recs() to avoid duplicating the
    fetch + filter + dedup logic.

    Filters applied (in order):
      1. building_since < 2 years ago
      2. location contains "United States"
      3. company_name does not contain "(YC "
      4. founder tenure at current company < 2 years

    Returns:
        pd.DataFrame of filtered, deduplicated founders (all columns), or None on DB error.
    """
    from services.database import get_db_connection

    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to the database")
        return None
    try:
        new_profiles = pd.read_sql("""
        SELECT *
        FROM founders
        WHERE founder = true
        AND history = ''
        AND (pedigree_passes IS DISTINCT FROM false)
        AND (latestdealtype IS NULL OR latestdealtype NOT ILIKE 'Series%')
        AND (
            building_since IS NULL OR building_since = ''
            OR (building_since ~ '^\\d{4}' AND SUBSTRING(building_since FROM 1 FOR 4)::integer >= DATE_PART('year', CURRENT_DATE) - 2)
        )
        """, conn)
        new_profiles = new_profiles.drop_duplicates(subset=['name'])
        new_profiles = new_profiles.reset_index(drop=True)
        print(f"Found {len(new_profiles)} new recommended profiles")
    except Exception as e:
        print(f"❌ Error fetching new profiles: {e}")
        return None
    finally:
        conn.close()

    # Filter out companies that are 2 years or older
    if 'building_since' in new_profiles.columns:
        initial_count = len(new_profiles)
        new_profiles = new_profiles[new_profiles['building_since'].apply(
            lambda x: is_company_less_than_2_years_old(x, max_years=2)
        )].copy()
        filtered_count = initial_count - len(new_profiles)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} companies that are 2+ years old (remaining: {len(new_profiles)})")
    else:
        print("⚠️  Warning: 'building_since' column not found in database, skipping age filter")

    # Filter to founders with "United States" in location only
    if 'location' in new_profiles.columns:
        initial_count = len(new_profiles)
        new_profiles = new_profiles[
            new_profiles['location'].fillna('').astype(str).str.lower().str.contains('united states', na=False)
        ].copy()
        filtered_count = initial_count - len(new_profiles)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} founders without United States in location (remaining: {len(new_profiles)})")
    else:
        print("⚠️  Warning: 'location' column not found in database, skipping location filter")

    # Filter out companies with (YC XXX) in company name
    if 'company_name' in new_profiles.columns:
        initial_count = len(new_profiles)
        new_profiles = new_profiles[
            ~new_profiles['company_name'].fillna('').astype(str).str.contains('(YC ', regex=False, na=False)
        ].copy()
        filtered_count = initial_count - len(new_profiles)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} companies with (YC XXX) in name (remaining: {len(new_profiles)})")

    # Filter out founders at current company 2+ years (using all_experiences)
    if 'all_experiences' in new_profiles.columns:
        initial_count = len(new_profiles)
        new_profiles = new_profiles[
            new_profiles.apply(
                lambda r: founder_tenure_at_company_less_than_2_years(r, max_years=2),
                axis=1
            )
        ].copy()
        filtered_count = initial_count - len(new_profiles)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} founders at company 2+ years (remaining: {len(new_profiles)})")

    # Filter out profiles whose current title is a non-founder "founding" role
    # (e.g. "Founding Engineer", "Founding Advisor") — these are not company founders.
    initial_count = len(new_profiles)
    new_profiles = new_profiles[
        new_profiles.apply(is_founder_or_cofounder_at_current_company, axis=1)
    ].copy()
    filtered_count = initial_count - len(new_profiles)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} non-founder 'founding' roles (remaining: {len(new_profiles)})")

    # Filter out founders estimated to be over 40 (inferred from earliest experience start date)
    from workflows.aviato_processing import estimate_founder_age
    initial_count = len(new_profiles)
    new_profiles = new_profiles[
        new_profiles.apply(lambda r: (estimate_founder_age(r.to_dict()) or 0) <= 40, axis=1)
    ].copy()
    filtered_count = initial_count - len(new_profiles)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} founders estimated over 40 (remaining: {len(new_profiles)})")

    new_profiles = new_profiles.drop_duplicates(subset=['company_name'])
    new_profiles = new_profiles.drop_duplicates(subset=['name'])
    new_profiles = new_profiles.drop_duplicates(subset=['profile_url'])
    new_profiles = new_profiles.reset_index(drop=True)
    print(f"After deduplication: {len(new_profiles)} unique profiles")

    return new_profiles


def send_extra_recs(test=True, include_exa=True):
    """Send personalized recommendations to all users via email based on their interests.

    This function:
    1. Fetches new recommended profiles from the database
    2. Matches them to each user's areas of interest (from the taste tree)
    3. Uses ML model to rank profiles by combined_score
    4. Sends personalized emails with top 3 recommendations per user (or one test email to matthildur when test=True)
    5. When test=False, marks sent recommendations to avoid duplicates
    """
    from concurrent.futures import ThreadPoolExecutor
    from services.tree import get_nodes_and_names
    from services.database import get_db_connection
    from services.path_mapper import get_all_matching_old_paths
    from services.notion import import_pipeline, normalize_string
    from services.model_loader import load_ranker_model
    from services.ranker_inference import rank_profiles
    try:
        from test_models import prepare_test_features
    except ImportError:
        prepare_test_features = None

    # Load ranker-only model (shared loader — no duplicate boilerplate)
    ranker_inference = load_ranker_model()

    # Fetch and filter founders (shared helper — no duplicate filter blocks)
    new_profiles = _fetch_and_filter_founders()
    if new_profiles is None:
        return None

    # Load companies from Notion pipeline/tracked/passed databases (parallelized)
    print("\nLoading companies from Notion databases...")

    def _load_notion_set(db_id, label):
        result = set()
        try:
            df = import_pipeline(db_id)
            for company_name in df['company_name'].dropna():
                normalized = normalize_string(str(company_name))
                if normalized:
                    result.add(normalized)
            print(f"  {label}: {len(result)} companies")
        except Exception as e:
            print(f"  ⚠️  Error loading {label}: {e}")
        return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_pipeline = executor.submit(_load_notion_set, PIPELINE_ID, 'Pipeline')
        fut_tracked = executor.submit(_load_notion_set, TRACKED_ID, 'Tracked')
        fut_passed = executor.submit(_load_notion_set, PASSED_ID, 'Passed')
        pipeline_companies = fut_pipeline.result()
        tracked_companies = fut_tracked.result()
        passed_companies = fut_passed.result()
    
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
    test_emails = {}  # user -> html_body, collected in test mode for preview
    
    # Process each user
    for user, categories in nodes_and_names.items():
        print(f"\nProcessing recommendations for {user}")
        
        # Apply category filter if configured for this user
        categories = filter_categories_by_tree_path(categories, user)
        
        if not categories:
            print(f"  ⚠️  No categories remaining after filtering for {user}")
            continue
        
        # Build profile pool, tracking the depth and the matched user-category per profile.
        # deeper match = more specific alignment with what this user cares about.
        profiles_chunks = []
        for category in categories:
            old_paths = get_all_matching_old_paths(category)
            cat_depth = category.count(' > ') + 1

            # Match profiles using both new and old paths
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
            for old_path in old_paths:
                old_matching = recs[recs['tree_path'].str.contains(old_path, na=False, regex=False)]
                matching = pd.concat([matching, old_matching])

            matching = matching.drop_duplicates(subset=['name', 'company_name'])
            if not matching.empty:
                matching = matching.copy()
                matching['match_depth'] = cat_depth
                matching['matched_category'] = category  # user interest category that triggered this match
                profiles_chunks.append(matching)

        if not profiles_chunks:
            print(f"  ⚠️  No matching profiles found for {user}")
            continue

        profiles = pd.concat(profiles_chunks)
        # Keep each profile's deepest match (sort desc before dedup preserves the best row)
        profiles = profiles.sort_values('match_depth', ascending=False)
        profiles['category'] = profiles['tree_path'].apply(
            lambda x: x.split(' > ')[-2] if len(x.split(' > ')) > 1 else x
        )
        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)
        
        # Merge with new_profiles to get full data for ranker (tenure already filtered globally)
        profile_keys = profiles[['name', 'company_name']].copy()
        profiles_full = profile_keys.merge(
            new_profiles,
            on=['name', 'company_name'],
            how='left'
        )
        
        # Rank profiles with model (batch inference via rank_profiles)
        if ranker_inference is not None and len(profiles) > 0:
            print(f"  Ranking {len(profiles)} profiles using ranker-only model...")
            try:
                profiles_full_ranked = rank_profiles(profiles_full, ranker_inference, prepare_test_features)
                if 'ranker_score' in profiles_full_ranked.columns:
                    score_map = dict(zip(
                        zip(profiles_full_ranked['name'], profiles_full_ranked['company_name']),
                        profiles_full_ranked['ranker_score']
                    ))
                    profiles['ranker_score'] = profiles.apply(
                        lambda r: score_map.get((r['name'], r['company_name']), 0.0), axis=1
                    )
                    profiles.sort_values(by=['ranker_score', 'match_depth'], ascending=[False, False], inplace=True)
                    print(f"  Ranker ranking complete. Top score: {profiles['ranker_score'].iloc[0]:.4f}")
                else:
                    raise RuntimeError("rank_profiles returned no ranker_score")
            except Exception as e:
                print(f"  ⚠️  Error using ranker model: {e}")
                print("  Falling back to match_depth and past_success_indication_score sorting")
                profiles.sort_values(
                    by=['match_depth', 'past_success_indication_score'],
                    ascending=[False, False],
                    inplace=True
                )
        else:
            # Fallback: deeper category match first, then founder success score
            profiles.sort_values(
                by=['match_depth', 'past_success_indication_score'],
                ascending=[False, False],
                inplace=True
            )
        
        # Take top 3 recommendations (with iterative pedigree check)
        profiles, pedigree_results = _select_top_with_pedigree(profiles, new_profiles, target=3)
        _save_pedigree_results(pedigree_results)

        if len(profiles) == 0:
            print(f"No profiles left after filtering for {user}")
            continue
        
        # Track sent recommendations
        all_names.extend(profiles['name'].tolist())
        companies.extend(profiles['company_name'].tolist())
        
        # Build category string from the matched user-interest categories (not the profile paths).
        # Use the last (most specific) node of each matched category, deduplicated.
        seen_cats: set = set()
        category_parts = []
        for cat in profiles['matched_category']:
            label = cat.split(' > ')[-1]
            if label not in seen_cats:
                seen_cats.add(label)
                category_parts.append(label)
        category_string = ", ".join(category_parts).lower()
        
        # Build greeting and HTML body for email
        greeting_text = f"Hey {user}! I came across these profiles in {category_string} that I wanted to share with you."
        message_lines = [greeting_text]
        
        print(greeting_text)
        
        for _, row in profiles.iterrows():
            line = (
                f"• <{fix_profile_url(row['profile_url'])}|*{row['name']}*> at {fix_company_url(row['company_website'], row['company_name'])}\n"
                f"    {row['product']}"
            )
            message_lines.append(line)
            profile_link = fix_profile_url(row['profile_url'])
            score_info = f"ranker_score: {row['ranker_score']:.4f}" if 'ranker_score' in row else f"tree_result: {row['tree_result']}"
            print(f"    - {row['name']} at {row['company_name']} ({score_info})")
            print(f"      Profile: {profile_link}")
        
        # Exa vertical discovery for this user
        exa_profiles_for_user = []
        exa_vertical_for_user = None
        if include_exa:
            exa_vertical_for_user = _current_vertical(user)
            if exa_vertical_for_user:
                print(f"  Running Exa search for vertical: {exa_vertical_for_user!r}")
                exa_profiles_for_user = get_exa_profiles_for_vertical(exa_vertical_for_user, num_results=15)
                print(f"  Exa found {len(exa_profiles_for_user)} qualifying profiles")
                for ep in exa_profiles_for_user:
                    print(f"    • {ep.get('name')} @ {ep.get('company_name')}")

        html_body = build_recommendation_email_html(
            profiles, greeting_text,
            exa_profiles=exa_profiles_for_user or None,
            exa_vertical=exa_vertical_for_user,
        )

        # Remove profiles from the pool so they're not sent to other users
        recs = recs[~recs['company_name'].isin(profiles['company_name'])]
        
        if test:
            test_emails[user] = html_body
            print("Won't send to real recipients (test=True)")
        else:
            to_email = email_map.get(user)
            if to_email:
                print(f"Sending recommendation email to {user} ({to_email})")
                sent_ok = send_recommendation_email(to_email, "Monty recommendations for you", html_body)
                time.sleep(2)  # Rate limiting

                # Only after a successful send: insert Exa profiles to DB and mark as recommended
                if sent_ok and exa_profiles_for_user:
                    from workflows.aviato_processing import safe_insert_profile_to_db, determine_stealth_status
                    from services.database import get_db_connection
                    date_str = datetime.now().strftime("%Y-%m-%d")
                    history_value = f"recommended {user} {date_str}"
                    inserted_profile_urls = []
                    for ep in exa_profiles_for_user:
                        try:
                            ep["history"] = history_value
                            stealth = determine_stealth_status(ep)
                            ok = safe_insert_profile_to_db(ep, stealth=stealth)
                            if ok:
                                inserted_profile_urls.append(ep.get("profile_url") or ep.get("linkedin_url") or "")
                                print(f"    Inserted Exa profile: {ep.get('name')} @ {ep.get('company_name')}")
                            else:
                                # Profile already exists — update history to mark as recommended
                                conn = get_db_connection()
                                if conn:
                                    try:
                                        from psycopg2 import sql as _sql
                                        cur = conn.cursor()
                                        profile_url = ep.get("profile_url") or ep.get("linkedin_url") or ""
                                        for tbl in ["founders", "stealth_founders"]:
                                            cur.execute(
                                                _sql.SQL("UPDATE {} SET history = %s WHERE profile_url = %s").format(_sql.Identifier(tbl)),
                                                (history_value, profile_url),
                                            )
                                        conn.commit()
                                    except Exception as e:
                                        print(f"    ⚠️  Failed to update history for existing profile: {e}")
                                    finally:
                                        if 'cur' in locals():
                                            cur.close()
                                        conn.close()
                        except Exception as e:
                            print(f"    ⚠️  Error inserting Exa profile {ep.get('name')}: {e}")
            else:
                print(f"⚠️  No email found for {user}, skipping")
    
    # test=True: no emails sent
    if test:
        print("\nTEST MODE - skipping all email sends.")
    
    # Mark all sent recommendations in the database
    if all_names or companies:
        if test:
            print("Won't mark anything as recommended")
        else:
            print(f"Marking {len(set(all_names))} profiles as recommended")
            mark_prev_recs(all_names, companies)
    
    print("Recommendation sending complete!")
    return test_emails if test else None


def _print_exa_results_for_all():
    """Run Exa discovery for every user, print results, and save an HTML preview."""
    from services.tree import get_nodes_and_names
    nodes_and_names = get_nodes_and_names(use_supabase=True)
    users = list(nodes_and_names.keys())
    print(f"\nRunning Exa search for {len(users)} users\n")

    html_sections = []

    for user in users:
        vertical = _current_vertical(user)
        if not vertical:
            print(f"  {user}: no vertical configured, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"EXA RESULTS FOR {user.upper()}  |  vertical: {vertical!r}")
        print("="*60)
        profiles = get_exa_profiles_for_vertical(vertical, num_results=15)
        if not profiles:
            print("  No qualifying profiles found.")
            html_sections.append(
                f'<h2 style="font-family:sans-serif;margin-top:2em;border-bottom:1px solid #ccc">'
                f'Exa section for {html_module.escape(user)}</h2>'
                f'<p style="font-family:sans-serif;color:#888">No qualifying profiles found.</p>'
            )
            continue

        for p in profiles:
            name    = p.get("name") or "Unknown"
            company = p.get("company_name") or "Unknown"
            url     = fix_profile_url(p.get("profile_url") or p.get("linkedin_url") or "")
            product = (p.get("product") or p.get("description_1") or "")[:150]
            tree    = p.get("tree_path") or ""
            print(f"  • {name} @ {company}")
            print(f"    {url}")
            if product:
                print(f"    {product}")
            if tree:
                print(f"    tree: {tree}")

        # Build the exact email HTML section this user would receive
        exa_html = build_recommendation_email_html(
            pd.DataFrame(),  # no classic recs — Exa section only
            greeting_text="",
            exa_profiles=profiles,
            exa_vertical=vertical,
        )
        html_sections.append(
            f'<h2 style="font-family:sans-serif;margin-top:2em;border-bottom:1px solid #ccc">'
            f'Exa section for {html_module.escape(user)}'
            f'<span style="font-weight:normal;font-size:0.8em;color:#888"> — {html_module.escape(vertical)}</span></h2>'
            f'<div style="font-family:sans-serif;max-width:600px;margin:0 auto 2em">{exa_html}</div>'
        )

    # Write preview file
    preview_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_exa_preview.html")
    full_html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Exa preview</title></head><body>'
        + "\n".join(html_sections)
        + "</body></html>"
    )
    with open(preview_path, "w") as f:
        f.write(full_html)
    print(f"\nPreview saved → {preview_path}")


def _send_exa_only(users: list):
    """Send just the Exa discovery section as a standalone email to the given users."""
    from services.google_client import send_html_email
    for user in users:
        vertical = _current_vertical(user)
        if not vertical:
            print(f"  {user}: no vertical configured, skipping")
            continue
        to_email = email_map.get(user)
        if not to_email:
            print(f"  {user}: no email address found, skipping")
            continue
        print(f"\n{user}  |  vertical: {vertical!r}")
        profiles = get_exa_profiles_for_vertical(vertical, num_results=15)
        if not profiles:
            print(f"  No qualifying profiles found, skipping send for {user}")
            continue
        standalone_intro = (
            f"Hey {user}, adding on to my previous message: I know you're interested in "
            f"{vertical} so wanted to also share these profiles I just came across."
        )
        html_body = build_recommendation_email_html(
            pd.DataFrame(),
            greeting_text="",
            exa_profiles=profiles,
            exa_vertical=vertical,
            exa_intro=standalone_intro,
        )
        result = send_html_email(to_email, "A few more profiles for you from Monty", html_body)
        if result:
            print(f"  ✅ Sent to {to_email}")
            # Insert profiles to DB and mark as recommended
            from workflows.aviato_processing import safe_insert_profile_to_db, determine_stealth_status
            from services.database import get_db_connection
            date_str = datetime.now().strftime("%Y-%m-%d")
            history_value = f"recommended {user} {date_str}"
            for ep in profiles:
                try:
                    ep["history"] = history_value
                    stealth = determine_stealth_status(ep)
                    ok = safe_insert_profile_to_db(ep, stealth=stealth)
                    if not ok:
                        conn = get_db_connection()
                        if conn:
                            try:
                                from psycopg2 import sql as _sql
                                cur = conn.cursor()
                                profile_url = ep.get("profile_url") or ep.get("linkedin_url") or ""
                                for tbl in ["founders", "stealth_founders"]:
                                    cur.execute(
                                        _sql.SQL("UPDATE {} SET history = %s WHERE profile_url = %s").format(_sql.Identifier(tbl)),
                                        (history_value, profile_url),
                                    )
                                conn.commit()
                            except Exception:
                                pass
                            finally:
                                if 'cur' in locals():
                                    cur.close()
                                conn.close()
                except Exception:
                    pass
        else:
            print(f"  ❌ Failed to send to {to_email}")


def main():
    """Recommendations entrypoint.

    Modes
    -----
    (no args)         test mode — print classic recs + Exa results for all users, no emails sent
    send              send mode — send full emails (classic + Exa section) to all users
    --classic         only print/run classic recommendations
    --exa             only print Exa discovery results (no Aviato scoring, no email)
    """
    import argparse
    parser = argparse.ArgumentParser(description="Monty recommendations")
    parser.add_argument("mode", nargs="?", default="test",
                        choices=["test", "send"],
                        help="'send' to send emails; default is test mode (no emails)")
    parser.add_argument("--classic", action="store_true",
                        help="Only run classic recommendations (skip Exa)")
    parser.add_argument("--exa", action="store_true",
                        help="Only run and print Exa discovery results (skip classic)")
    parser.add_argument("--exa-send", nargs="+", metavar="USER",
                        help="Send just the Exa section email to specific users (e.g. --exa-send Todd Matt)")
    args = parser.parse_args()

    sending = args.mode == "send"
    run_classic = not args.exa   # True unless --exa only
    run_exa     = not args.classic  # True unless --classic only

    print("=" * 60)
    print("Monty Recommendations System")
    print(f"Mode: {'SEND' if sending else 'TEST (no emails)'}")
    if args.classic:
        print("Scope: classic only")
    elif args.exa:
        print("Scope: Exa only")
    print("=" * 60)

    if args.exa_send:
        _send_exa_only(args.exa_send)
        return

    if args.exa and not args.classic:
        _print_exa_results_for_all()
        return

    if run_classic:
        print("\n📧 Classic recommendations:")
        print("-" * 60)
        test_emails = send_extra_recs(test=not sending, include_exa=run_exa)

        # In test mode, write each user's email to a preview HTML file
        if test_emails:
            preview_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test_recs_preview.html")
            sections = []
            for user, html_body in test_emails.items():
                sections.append(
                    f'<h2 style="font-family:sans-serif;margin-top:2em;border-bottom:1px solid #ccc">'
                    f'Email for {html_module.escape(user)}</h2>'
                    f'<div style="font-family:sans-serif;max-width:600px;margin:0 auto 2em">{html_body}</div>'
                )
            full_html = (
                '<!DOCTYPE html><html><head><meta charset="utf-8">'
                '<title>Recs preview</title></head><body>'
                + "\n".join(sections)
                + "</body></html>"
            )
            with open(preview_path, "w") as f:
                f.write(full_html)
            print(f"\nPreview saved → {preview_path}")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()