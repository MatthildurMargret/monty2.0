"""
Recommendations System for Monty
=================================

This module handles personalized founder/company recommendations sent to team members via Slack.

Main Functions:
--------------
1. send_extra_recs() - Main function to send recommendations to all users
   - Fetches new recommended profiles from the database
   - Matches them to each user's areas of interest (from taste_tree.json)
   - Sends personalized Slack DMs with top 3 recommendations per user
   - Marks sent recommendations to avoid duplicates (including date/channel metadata)

2. find_new_recs(username) - Preview recommendations for a specific user (testing)

3. mark_prev_recs(...) - Mark profiles as recommended in the database with contextual history

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
- taste_tree.json must have 'montage_lead' metadata for user assignments
"""

import pandas as pd
import time
import os
import warnings
from datetime import datetime
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

    nodes_and_names = get_nodes_and_names()

    recs['top_category'] = recs['tree_path'].apply(lambda x: x.split(' > ')[0] if pd.notna(x) else '')

    for user, categories in nodes_and_names.items():
        if user != username:
            continue
        print(f"\nProfiles for {user}")
        print("-" * 60)
        
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


def mark_prev_recs(names=None, companies=None, history_entries=None, channel=None, send_date=None):
    """Mark profiles as 'recommended' in the database and annotate context metadata.
    
    Args:
        names: Optional list of founder names to mark (legacy support)
        companies: Optional list of company names to mark (legacy support)
        history_entries: Optional list of dict entries with keys:
            - name: founder name (optional if company provided)
            - company: company name (optional if name provided)
            - channel: channel label (e.g., 'slack', 'newsletter')
            - recipients: list/iterable or string of recipients (for slack DMs)
            - context_detail: explicit detail string to append inside parentheses
            - send_date: explicit date string or datetime; defaults to today
        channel: Default channel used when history_entries not provided
        send_date: Default date (datetime or string YYYY-MM-DD). Defaults to today.
    """
    from services.database import get_db_connection
    
    if not names and not companies and not history_entries:
        print("No names or companies provided to mark")
        return
    
    def format_date(value):
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return datetime.utcnow().strftime("%Y-%m-%d")
    
    def normalize_recipients(value):
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value if v]
    
    def format_detail(channel_value, recipients_list, context_detail):
        if context_detail:
            return context_detail
        detail_parts = []
        recipients_list = [r for r in recipients_list if r]
        channel_value = channel_value.strip().lower() if isinstance(channel_value, str) else None
        if channel_value:
            if recipients_list:
                recipients_text = ", ".join(recipients_list)
                detail_parts.append(f"{channel_value} to {recipients_text}")
            else:
                detail_parts.append(channel_value)
        elif recipients_list:
            detail_parts.append(f"to {', '.join(recipients_list)}")
        return ", ".join(detail_parts) if detail_parts else ""
    
    send_date_default = format_date(send_date)
    
    entries_to_process = []
    if history_entries:
        for entry in history_entries:
            entry = entry or {}
            entry_name = entry.get('name')
            entry_company = entry.get('company')
            if not entry_name and not entry_company:
                continue
            entry_channel = entry.get('channel', channel)
            entry_recipients = normalize_recipients(entry.get('recipients') or entry.get('recipient'))
            entry_detail = entry.get('context_detail')
            entry_date = format_date(entry.get('send_date') or send_date_default)
            entries_to_process.append({
                'name': entry_name,
                'company': entry_company,
                'channel': entry_channel,
                'recipients': entry_recipients,
                'detail': entry_detail,
                'send_date': entry_date
            })
    else:
        unique_names = set(names or [])
        unique_companies = set(companies or [])
        for n in unique_names:
            entries_to_process.append({
                'name': n,
                'channel': channel,
                'recipients': [],
                'detail': None,
                'send_date': send_date_default
            })
        for c in unique_companies:
            entries_to_process.append({
                'company': c,
                'channel': channel,
                'recipients': [],
                'detail': None,
                'send_date': send_date_default
            })
    
    if not entries_to_process:
        print("No valid entries to process for marking recommendations.")
        return
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to the database")
        return
        
    try:
        cur = conn.cursor()
        total_updates = 0
        for table_name in ["founders", "stealth_founders"]:
            for entry in entries_to_process:
                conditions = []
                params = []
                if entry.get('name'):
                    conditions.append(sql.SQL("name = %s"))
                    params.append(entry['name'])
                if entry.get('company'):
                    conditions.append(sql.SQL("company_name = %s"))
                    params.append(entry['company'])
                if not conditions:
                    continue
                
                detail_text = format_detail(entry.get('channel'), entry.get('recipients'), entry.get('detail'))
                history_value = f"recommended ({entry.get('send_date')}"
                if detail_text:
                    history_value += f", {detail_text}"
                history_value += ")"

                query = sql.SQL("UPDATE {} SET history = %s WHERE ").format(sql.Identifier(table_name))
                query = query + sql.SQL(" OR ").join(conditions)
                cur.execute(query, [history_value] + params)
                total_updates += cur.rowcount
        print(f"Updated {total_updates} records across founders tables.")
        conn.commit()
        print(f"✅ Successfully marked profiles as recommended with context.")
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
    3. Sends personalized Slack DMs with top 3 recommendations per user
    4. Marks sent recommendations to avoid duplicates
    """
    from services.tree import get_nodes_and_names
    from services.database import get_db_connection
    from services.path_mapper import get_all_matching_old_paths
    
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
    
    # Select relevant columns
    columns_to_share = ['name', 'company_name', 'company_website', 'profile_url', 
                       'tree_thesis', 'product', 'market', 'tree_path', 
                       'past_success_indication_score', 'tree_result']
    recs = new_profiles[columns_to_share].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Remove duplicates
    recs = recs.drop_duplicates(subset=['company_name'])
    recs = recs.drop_duplicates(subset=['name'])
    recs = recs.drop_duplicates(subset=['profile_url'])
    recs = recs.reset_index(drop=True)
    
    print(f"After deduplication: {len(recs)} unique profiles")
    
    # Get user interest mappings from the taste tree
    nodes_and_names = get_nodes_and_names()
    
    # Add top category for filtering
    recs['top_category'] = recs['tree_path'].apply(lambda x: x.split(' > ')[0] if pd.notna(x) else '')
    
    all_names = []
    companies = []
    history_entries_map = {}
    send_date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    # Process each user
    for user, categories in nodes_and_names.items():
        print(f"\nProcessing recommendations for {user}")
        
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
        
        # Get top profile from each category
        profiles = profiles.groupby('category').head(1)
        
        # Remove duplicates
        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)
        
        # Sort by recommendation strength and success score
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
            print(f"    - {row['name']} at {row['company_name']} ({row['tree_result']})")
            print(f"      Profile: {profile_link}")
            
            entry_key = (row['name'], row['company_name'])
            entry = history_entries_map.setdefault(
                entry_key,
                {
                    'name': row['name'] if pd.notna(row['name']) else None,
                    'company': row['company_name'] if pd.notna(row['company_name']) else None,
                    'channel': 'slack',
                    'recipients': set(),
                    'send_date': send_date_str
                }
            )
            entry['recipients'].add(user)
        
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
        history_entries = []
        for entry in history_entries_map.values():
            entry['recipients'] = sorted(list(entry['recipients']))
            history_entries.append(entry)
        if test:
            print("Won't mark anything as recommended")
        else:
            print(f"Marking {len(set(all_names))} profiles as recommended")
            mark_prev_recs(history_entries=history_entries)
    
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