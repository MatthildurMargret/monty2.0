import json
from services.tree_tools import find_pipeline_companies, find_nodes_by_name
from services.database import get_founders_by_path
import pandas as pd
from datetime import datetime, timedelta
from workflows.recommendations import mark_prev_recs
from services.path_mapper import get_all_matching_old_paths
from workflows.profile_recommendations import get_profile_recommendations, mark_profiles_as_recommended, format_profiles_for_weekly_update

def get_recent_deals():
    deals = pd.read_csv('data/deal_data/early_deals.csv', quotechar='"')
    deals["Investors"] = deals["Investors"].astype(str)
    deals["Investors"] = deals["Investors"].fillna("")
    deals["Vertical"] = deals["Vertical"].fillna("")
    deals["Funding Round"] = deals["Funding Round"].fillna("")
    deals["Category"] = deals["Category"].fillna("Other")
    current_date = datetime.now().strftime("%B %d, %Y")  # e.g., "November 16, 2024"
    past_week = datetime.now() - timedelta(days=6)
    deals["Date"] = pd.to_datetime(deals["Date"], errors="coerce")
    deals = deals[deals["Date"] > past_week].copy()
    return deals

def get_tracking():
    df = pd.read_csv('data/tracking/tracking_db.csv')
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
    return subcategories

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

    for category, subs in subcategories.items():
        print(f"Category: {category}")
        for sub in subs:
            print(f"  Subcategory: {sub['subcategory']} ({sub['count']} companies)")
            # --- NEW CODE: map to old taxonomy paths ---
            new_path = sub['subcategory']
            old_paths = get_all_matching_old_paths(new_path)
            mapped_paths = old_paths if old_paths else [new_path]
            
            # --- Get top founder for this subcategory (search new + old paths) ---
            founders = []
            for path in mapped_paths:
                result = get_founders_by_path(path)
                if result:
                    founders.extend(result)

            if founders:
                founders_df = pd.DataFrame(founders)
                founders_df.sort_values(
                    by='past_success_indication_score',
                    ascending=False,
                    inplace=True
                )
                top_founder = founders_df.iloc[0].to_dict()
                sub['top_founder'] = top_founder
            else:
                sub['top_founder'] = None

            top_founders.append(sub['top_founder'])
            
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
    greeting_text = "Happy Halloween! Apologies for spamming your inbox today, I accidentally sent you last week's update earlier. Ignore it! Now here's what's new this week: "
    greeting_text += "There were a ton of deals done this week, highlighted below, and I'm very excited about the early stage companies I've sourced for you. "
    greeting_text += "\n\nWishing you all a spooky evening and a great weekend!\n\n - Monty"
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

