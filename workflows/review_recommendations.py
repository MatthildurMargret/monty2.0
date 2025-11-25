"""
Review Recommendations - Preview and filter recommendations before sending.

This script shows all profiles that would be recommended to each team member,
allowing you to review and mark profiles as "pass" in the database before sending.
"""

import pandas as pd
import webbrowser
from services.tree import get_nodes_and_names
from services.database import get_db_connection
from services.path_mapper import get_all_matching_old_paths

def fix_url(url):
    """Ensure URL has https:// prefix."""
    if not url or pd.isna(url):
        return None
    url = str(url).strip()
    if not url:
        return None
    if not url.startswith('http://') and not url.startswith('https://'):
        return f'https://{url}'
    return url

def get_recommendations_preview():
    """Get all recommendations that would be sent to each user."""
    
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
    recs = new_profiles[columns_to_share].copy()
    
    # Remove duplicates
    recs = recs.drop_duplicates(subset=['company_name'])
    recs = recs.drop_duplicates(subset=['name'])
    recs = recs.drop_duplicates(subset=['profile_url'])
    recs = recs.reset_index(drop=True)
    
    print(f"After deduplication: {len(recs)} unique profiles\n")
    
    # Get user interest mappings from the taste tree
    nodes_and_names = get_nodes_and_names()
    
    # Build recommendations for each user
    user_recommendations = {}
    already_recommended = set()  # Track profiles already assigned to avoid duplicates
    
    for user, categories in nodes_and_names.items():
        profiles = pd.DataFrame()
        
        for category in categories:
            # Get all old paths that map to this new category
            old_paths = get_all_matching_old_paths(category)
            
            # Match profiles using both new and old paths
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
            
            for old_path in old_paths:
                old_matching = recs[recs['tree_path'].str.contains(old_path, na=False, regex=False)]
                matching = pd.concat([matching, old_matching])
            
            matching = matching.drop_duplicates(subset=['name', 'company_name'])
            profiles = pd.concat([profiles, matching])
        
        if len(profiles) == 0:
            continue
        
        # Extract category for grouping
        profiles['category'] = profiles['tree_path'].apply(
            lambda x: x.split(' > ')[-2] if len(x.split(' > ')) > 1 else x
        )
        
        # Get top profile from each category
        profiles = profiles.groupby('category').head(1)
        profiles = profiles.drop_duplicates(subset=['company_name'])
        profiles = profiles.drop_duplicates(subset=['name'])
        profiles = profiles.reset_index(drop=True)
        profiles = profiles.sort_values(
            by=['tree_result', 'past_success_indication_score'], 
            ascending=[False, False]
        )
        
        # Filter out profiles already recommended to other users
        profiles = profiles[~profiles['name'].isin(already_recommended)]
        
        # Take top 3
        profiles = profiles.head(3)
        
        # Mark these profiles as recommended
        for name in profiles['name']:
            already_recommended.add(name)
        
        # Fix URLs
        profiles['profile_url'] = profiles['profile_url'].apply(fix_url)
        profiles['company_website'] = profiles['company_website'].apply(fix_url)
        
        user_recommendations[user] = profiles
    
    return user_recommendations

def display_recommendations(user_recommendations):
    """Display all recommendations in a readable format."""
    
    print("=" * 80)
    print("RECOMMENDATIONS PREVIEW")
    print("=" * 80)
    
    total_recs = sum(len(profiles) for profiles in user_recommendations.values())
    print(f"\nTotal recommendations across all users: {total_recs}\n")
    
    for user, profiles in user_recommendations.items():
        print("\n" + "=" * 80)
        print(f"👤 {user.upper()} - {len(profiles)} recommendations")
        print("=" * 80)
        
        for idx, row in profiles.iterrows():
            profile_url = fix_url(row['profile_url'])
            company_url = fix_url(row['company_website'])
            
            print(f"\n[{idx + 1}] {row['name']} at {row['company_name']}")
            print(f"    Rating: {row['tree_result']}")
            print(f"    Score: {row['past_success_indication_score']}")
            print(f"    Category: {row['tree_path']}")
            print(f"    Product: {row['product'][:100] if pd.notna(row['product']) else 'N/A'}...")
            print(f"    Profile: {profile_url}")
            print(f"    Company: {company_url if company_url else 'N/A'}")

def interactive_review():
    """Interactive review mode - review profiles that would actually be recommended to users."""
    
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to database")
        return
    
    try:
        # Fetch all profiles that could be recommended
        all_profiles = pd.read_sql("""
        SELECT *
        FROM founders
        WHERE founder = true
        AND history = ''
        AND (tree_result = 'Strong recommend' OR tree_result = 'Recommend')
        """, conn)
        
        all_profiles = all_profiles.drop_duplicates(subset=['name'])
        all_profiles = all_profiles.reset_index(drop=True)
        
        print(f"\n📋 Found {len(all_profiles)} total profiles")
        
    except Exception as e:
        print(f"❌ Error fetching profiles: {e}")
        return
    
    # Get user interest mappings from the taste tree
    nodes_and_names = get_nodes_and_names()
    
    # Filter to only profiles that match user categories
    columns_to_share = ['name', 'company_name', 'company_website', 'profile_url', 
                       'tree_thesis', 'product', 'market', 'tree_path', 
                       'past_success_indication_score', 'tree_result']
    recs = all_profiles[columns_to_share].copy()
    
    matched_profiles = pd.DataFrame()
    
    print("🔍 Filtering to profiles that match team interests...")
    
    for user, categories in nodes_and_names.items():
        for category in categories:
            # Get all old paths that map to this new category
            old_paths = get_all_matching_old_paths(category)
            
            # Match profiles using both new and old paths
            matching = recs[recs['tree_path'].str.contains(category, na=False, regex=False)]
            
            for old_path in old_paths:
                old_matching = recs[recs['tree_path'].str.contains(old_path, na=False, regex=False)]
                matching = pd.concat([matching, old_matching])
            
            matching = matching.drop_duplicates(subset=['name', 'company_name'])
            matched_profiles = pd.concat([matched_profiles, matching])
    
    # Remove duplicates
    matched_profiles = matched_profiles.drop_duplicates(subset=['name'])
    matched_profiles = matched_profiles.reset_index(drop=True)
    
    # Sort by priority: Strong recommend first, then by score
    matched_profiles = matched_profiles.sort_values(
        by=['tree_result', 'past_success_indication_score'], 
        ascending=[False, False]
    )
    matched_profiles = matched_profiles.reset_index(drop=True)
    
    print(f"✅ Filtered to {len(matched_profiles)} profiles that match team interests")
    print(f"   (Skipping {len(all_profiles) - len(matched_profiles)} profiles that don't match any categories)\n")
    
    print("\n" + "=" * 80)
    print("INTERACTIVE REVIEW MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  ENTER - Keep profile (do nothing)")
    print("  'p' - Mark as PASS (won't be recommended)")
    print("  'o' - Open profile URL in browser")
    print("  'c' - Open company website in browser")
    print("  'q' - Quit review")
    print("\n💡 Tip: You can quit anytime and resume later. Marked profiles stay marked.")
    print()
    
    marked_as_pass = []
    reviewed_count = 0
    
    try:
        for idx, row in matched_profiles.iterrows():
            profile_url = fix_url(row['profile_url'])
            company_url = fix_url(row['company_website'])
            
            print("\n" + "=" * 80)
            print(f"[{idx + 1}/{len(matched_profiles)}] {row['name']} at {row['company_name']}")
            print("=" * 80)
            print(f"Rating: {row['tree_result']} | Score: {row.get('past_success_indication_score', 'N/A')}")
            print(f"Category: {row['tree_path']}")
            print(f"\nProduct: {row['product'][:200] if pd.notna(row['product']) else 'N/A'}...")
            print(f"\nProfile: {profile_url}")
            print(f"Company: {company_url if company_url else 'N/A'}")
            
            while True:
                action = input("\nAction (Enter/p/o/c/q): ").strip().lower()
                
                if action == '':
                    print("✅ Keeping profile")
                    reviewed_count += 1
                    break
                elif action == 'p':
                    # Mark as pass in database
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE founders SET history = 'pass' WHERE name = %s AND company_name = %s",
                        (row['name'], row['company_name'])
                    )
                    conn.commit()
                    cursor.close()
                    marked_as_pass.append(f"{row['name']} at {row['company_name']}")
                    print("❌ Marked as PASS - won't be recommended")
                    reviewed_count += 1
                    break
                elif action == 'o':
                    if profile_url:
                        print(f"🌐 Opening profile: {profile_url}")
                        webbrowser.open(profile_url)
                    else:
                        print("⚠️  No profile URL available")
                elif action == 'c':
                    if company_url:
                        print(f"🌐 Opening company: {company_url}")
                        webbrowser.open(company_url)
                    else:
                        print("⚠️  No company website available")
                elif action == 'q':
                    print("\n👋 Exiting review...")
                    break
                else:
                    print("❓ Invalid command")
            
            if action == 'q':
                break
    
    finally:
        if conn:
            conn.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("REVIEW SUMMARY")
    print("=" * 80)
    print(f"\n📊 Reviewed: {reviewed_count}/{len(matched_profiles)} profiles")
    
    if marked_as_pass:
        print(f"\n❌ Marked {len(marked_as_pass)} profiles as PASS:")
        for profile in marked_as_pass:
            print(f"  - {profile}")
    else:
        print("\n✅ No profiles marked as PASS")
    
    remaining = len(matched_profiles) - reviewed_count
    if remaining > 0:
        print(f"\n⏸️  {remaining} profiles remaining (run again to continue)")
    
    print("\n💡 To send recommendations, run: python -m workflows.recommendations")

def export_to_csv():
    """Export all recommendations to CSV for review."""
    
    user_recommendations = get_recommendations_preview()
    
    if not user_recommendations:
        print("No recommendations to export")
        return
    
    # Combine all recommendations into one dataframe
    all_recs = []
    for user, profiles in user_recommendations.items():
        profiles_copy = profiles.copy()
        profiles_copy['recommended_to'] = user
        all_recs.append(profiles_copy)
    
    combined = pd.concat(all_recs, ignore_index=True)
    
    # Reorder columns
    cols = ['recommended_to', 'name', 'company_name', 'tree_result', 
            'past_success_indication_score', 'tree_path', 'product', 
            'profile_url', 'company_website']
    combined = combined[cols]
    
    output_file = 'data/recommendations_preview.csv'
    combined.to_csv(output_file, index=False)
    
    print(f"✅ Exported {len(combined)} recommendations to: {output_file}")
    print("\nYou can review this file and mark profiles as 'pass' in the database")

def main():
    import sys
    
    print("=" * 80)
    print("RECOMMENDATIONS REVIEW TOOL")
    print("=" * 80)
    print("\nOptions:")
    print("  1. Preview all recommendations (read-only)")
    print("  2. Interactive review (mark as pass)")
    print("  3. Export to CSV")
    print("  4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        user_recommendations = get_recommendations_preview()
        if user_recommendations:
            display_recommendations(user_recommendations)
    elif choice == '2':
        interactive_review()
    elif choice == '3':
        export_to_csv()
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid option")

if __name__ == '__main__':
    main()
