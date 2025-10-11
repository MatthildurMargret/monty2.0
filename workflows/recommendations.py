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
   - Marks sent recommendations to avoid duplicates

2. find_new_recs(username) - Preview recommendations for a specific user (testing)

3. mark_prev_recs(names, companies) - Mark profiles as 'recommended' in database

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
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
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
        greeting_text = f"Hey {user}! I came across these profiles that I wanted to share with you, given your interest in {category_string}."
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
    3. Sends personalized Slack DMs with top 3 recommendations per user
    4. Marks sent recommendations to avoid duplicates
    """
    from services.tree import get_nodes_and_names
    from services.database import get_db_connection
    
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
    
    # Process each user
    for user, categories in nodes_and_names.items():
        print(f"\nProcessing recommendations for {user}")
        
        # Filter profiles matching this user's categories
        profiles = pd.DataFrame()
        for category in categories:
            # Use regex=False to avoid regex interpretation warnings
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
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
        category_string = category_string[:-2]  # Remove trailing comma
        
        # Build Slack message
        greeting_text = f"Hey {user}! I came across these profiles that I wanted to share with you, given your interest in {category_string}."
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