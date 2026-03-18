"""
Script to process pipeline news articles CSV and add relevant articles to tracking database.

This script:
1. Reads the pipeline news articles CSV file
2. Filters to articles from the past 6 days
3. Verifies article relevance using founder and company description
4. Creates summaries using Parallel API extract
5. Adds new entries to the tracking database
"""
import os
import sys
import argparse
import glob
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import re

# Add the parent directory to the Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.notion import import_pipeline
from services.parallel_client import Parallel
from services.openai_api import ask_monty
from dotenv import load_dotenv

# Import tracking functions
from workflows.tracking import (
    load_tracking_from_supabase,
    save_tracking_to_supabase,
    get_detailed_info_on_alert,
    WORKSPACE_ROOT
)

# Load environment variables
load_dotenv()


def verify_article_matches_company(article_title, article_content, company_name, founder=None, brief_description=None):
    """
    Verify that an article is about the correct company using founder name and description.
    This is the same function from analyze_data.py
    
    Args:
        article_title (str): Article title
        article_content (str): Full article content/description
        company_name (str): Name of the company
        founder (str, optional): Founder name from Notion
        brief_description (str, optional): Company description from Notion
        
    Returns:
        bool: True if article matches the company, False otherwise
    """
    if not article_content or len(article_content.strip()) < 50:
        # Not enough content to verify, default to True
        return True
    
    full_text = f"{article_title} {article_content}".lower()
    
    # First pass: Text matching
    verification_score = 0
    max_score = 0
    
    # Check founder name (if available)
    if founder and str(founder).strip():
        founder_clean = str(founder).strip().lower()
        # Remove common titles
        founder_clean = re.sub(r'\b(mr|mrs|ms|dr|professor|prof)\b\.?\s*', '', founder_clean, flags=re.IGNORECASE)
        founder_words = founder_clean.split()
        
        if len(founder_words) >= 2:
            # Check if both first and last name appear
            first_name = founder_words[0]
            last_name = founder_words[-1]
            if first_name in full_text and last_name in full_text:
                verification_score += 2
                max_score += 2
        elif len(founder_words) == 1:
            # Single name - check if it appears
            if founder_clean in full_text:
                verification_score += 1
                max_score += 1
    
    # Extract key terms from brief_description
    if brief_description and str(brief_description).strip():
        desc = str(brief_description).strip().lower()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        
        # Extract significant words (3+ characters, not stop words)
        words = re.findall(r'\b[a-z]{3,}\b', desc)
        significant_words = [w for w in words if w not in stop_words and len(w) >= 3]
        
        # Get top 5 most frequent significant words
        word_counts = defaultdict(int)
        for word in significant_words:
            word_counts[word] += 1
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_words = [word for word, count in top_words]
        
        if top_words:
            max_score += 3
            # Check if at least 2-3 key terms appear
            matches = sum(1 for word in top_words if word in full_text)
            if matches >= 2:
                verification_score += 3
            elif matches == 1:
                verification_score += 1
    
    # Decision logic
    # If we have founder match, that's strong evidence
    if founder and str(founder).strip() and verification_score >= 2:
        return True
    
    # If we have description match (2+ terms), that's good evidence
    if brief_description and str(brief_description).strip() and verification_score >= 3:
        return True
    
    # If we have both founder and description matches, even partial is good
    if founder and brief_description and str(founder).strip() and str(brief_description).strip():
        if verification_score >= 2:
            return True
    
    # If text matching is inconclusive, use OpenAI for verification
    if max_score > 0 and 0 < verification_score < max_score * 0.6:
        try:
            prompt = (
                "You are a fact-checker. Determine if the given article is about the specified company. "
                "Respond with only 'YES' or 'NO', nothing else."
            )
            
            # Use first 2000 chars of content for OpenAI (to stay within token limits)
            content_preview = article_content[:2000] if len(article_content) > 2000 else article_content
            
            data = (
                f"Company: {company_name}\n"
                f"Company Description: {brief_description or 'Not provided'}\n"
                f"Founder: {founder or 'Not provided'}\n\n"
                f"Article Title: {article_title}\n"
                f"Article Content: {content_preview}"
            )
            
            response = ask_monty(prompt, data, max_tokens=10)
            response_clean = response.strip().upper()
            
            if 'YES' in response_clean:
                return True
            elif 'NO' in response_clean:
                return False
        except Exception as e:
            print(f"      Warning: OpenAI verification failed: {e}")
            # Fail-safe: if OpenAI fails, be conservative and skip
            return False
    
    # If no verification data available, default to True (existing behavior)
    if not founder and not brief_description:
        return True
    
    # If we have verification data but no matches, return False
    return False


def parse_article_date(date_str):
    """Parse article date string into datetime object."""
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    if not date_str:
        return None
    
    # Try various date formats
    date_formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m/%d/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str.split()[0], fmt)
        except (ValueError, IndexError):
            continue
    
    # Try pandas parsing as fallback
    try:
        return pd.to_datetime(date_str).to_pydatetime()
    except:
        return None


def find_latest_pipeline_news_csv():
    """Find the most recent pipeline news articles CSV file."""
    tracking_dir = os.path.join(WORKSPACE_ROOT, 'data', 'tracking')
    pattern = os.path.join(tracking_dir, 'pipeline_news_articles_*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        return None
    
    # Sort by modification time, most recent first
    csv_files.sort(key=os.path.getmtime, reverse=True)
    return csv_files[0]


def process_and_add_articles(csv_file=None, days_back=6):
    """
    Process pipeline news articles CSV and add relevant articles to tracking database.
    
    Args:
        csv_file (str, optional): Path to CSV file. If None, finds latest automatically.
        days_back (int): Number of days to look back (default: 6)
    """
    # Find CSV file if not provided
    if csv_file is None:
        csv_file = find_latest_pipeline_news_csv()
        if csv_file is None:
            print("❌ Error: No pipeline news articles CSV file found.")
            print(f"   Expected pattern: {os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'pipeline_news_articles_*.csv')}")
            return
        print(f"✓ Found CSV file: {os.path.basename(csv_file)}")
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file not found: {csv_file}")
        return
    
    # Load CSV file
    print(f"Loading articles from CSV...")
    try:
        articles_df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(articles_df)} articles from CSV")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Filter to past 6 days (or days_back)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    print(f"Filtering to articles published after: {cutoff_date.strftime('%Y-%m-%d')}")
    
    articles_df['parsed_date'] = articles_df['article_date'].apply(parse_article_date)
    recent_articles = articles_df[
        (articles_df['parsed_date'].notna()) & 
        (articles_df['parsed_date'] >= cutoff_date)
    ].copy()
    
    if recent_articles.empty:
        print(f"⚠️  No articles found within the last {days_back} days.")
        return
    
    print(f"✓ Found {len(recent_articles)} articles within the last {days_back} days")
    print()
    
    # Load pipeline to get company descriptions
    print("Loading pipeline data for company verification...")
    pipeline_id = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
    try:
        pipeline = import_pipeline(pipeline_id)
        print(f"✓ Loaded {len(pipeline)} companies from pipeline")
        
        # Create mapping of company name to metadata
        company_metadata = {}
        for idx, row in pipeline.iterrows():
            company_name = row.get('company_name', '')
            if company_name:
                company_metadata[company_name] = {
                    'founder': row.get('founder', ''),
                    'description': row.get('description', '') or row.get('Brief Description', ''),
                    'priority': row.get('priority', ''),
                    'website': row.get('website', ''),
                    'page_id': row.get('page_id', '')  # We'll need to generate a unique page_id for new entries
                }
    except Exception as e:
        print(f"⚠️  Warning: Could not load pipeline data: {e}")
        print("   Will proceed without company descriptions for verification")
        company_metadata = {}
    
    # Load existing tracking database
    print("Loading existing tracking database...")
    tracking_df = load_tracking_from_supabase()
    
    # Fallback to CSV if Supabase is empty
    if tracking_df.empty:
        tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
        if os.path.exists(tracking_db_path):
            tracking_df = pd.read_csv(tracking_db_path)
            print(f"✓ Loaded {len(tracking_df)} companies from CSV backup")
        else:
            print("⚠️  Warning: No existing tracking database found. Creating new entries.")
            tracking_df = pd.DataFrame()
    
    # Get existing company names to check for duplicates
    existing_companies = set(tracking_df['company_name'].tolist()) if not tracking_df.empty and 'company_name' in tracking_df.columns else set()
    
    # Process articles
    print("=" * 80)
    print("VERIFYING ARTICLE RELEVANCE")
    print("=" * 80)
    print()
    
    relevant_articles = []
    skipped_count = 0
    verified_count = 0
    
    for idx, article in recent_articles.iterrows():
        company_name = article['company_name']
        founder = article.get('founder', '')
        article_title = article['article_title']
        article_link = article['article_link']
        article_date = article['article_date']
        article_summary = article.get('article_summary', '')
        
        print(f"[{idx+1}/{len(recent_articles)}] Checking: {company_name}")
        print(f"  Article: {article_title[:70]}...")
        
        # Get company metadata for verification
        metadata = company_metadata.get(company_name, {})
        founder_from_pipeline = metadata.get('founder', '') or founder
        description = metadata.get('description', '')
        
        # Verify article relevance
        print(f"  → Verifying article relevance...")
        is_relevant = verify_article_matches_company(
            article_title=article_title,
            article_content=article_summary,
            company_name=company_name,
            founder=founder_from_pipeline,
            brief_description=description
        )
        
        if not is_relevant:
            print(f"  ✗ Article does not match company. Skipping.")
            skipped_count += 1
            continue
        
        print(f"  ✓ Article verified as relevant")
        verified_count += 1
        
        # Get detailed summary using Parallel API extract (similar to get_detailed_info_on_alert)
        print(f"  → Creating detailed summary...")
        try:
            detailed_summary = get_detailed_info_on_alert(
                link=article_link,
                company_name=company_name,
                search_keywords=f"{company_name} {founder_from_pipeline}"
            )
            if detailed_summary:
                article_summary = detailed_summary
                print(f"  ✓ Summary created ({len(detailed_summary)} chars)")
            else:
                print(f"  ⚠️  Could not create detailed summary, using original")
        except Exception as e:
            print(f"  ⚠️  Error creating summary: {e}, using original")
        
        relevant_articles.append({
            'company_name': company_name,
            'founder': founder_from_pipeline,
            'priority': metadata.get('priority', article.get('priority', '')),
            'website': metadata.get('website', ''),
            'description': description,
            'article_title': article_title,
            'article_link': article_link,
            'article_date': article_date,
            'article_summary': article_summary
        })
        print()
    
    print("=" * 80)
    print(f"VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Articles checked: {len(recent_articles)}")
    print(f"Verified as relevant: {verified_count}")
    print(f"Skipped (not relevant): {skipped_count}")
    print()
    
    if not relevant_articles:
        print("⚠️  No relevant articles found. Exiting.")
        return
    
    # Create new tracking entries
    print("=" * 80)
    print("CREATING TRACKING ENTRIES")
    print("=" * 80)
    print()
    
    new_entries = []
    entries_to_update = []
    
    for article in relevant_articles:
        company_name = article['company_name']
        
        # Check if company already exists in tracking
        if company_name in existing_companies:
            # Update existing entry
            print(f"Updating existing entry for: {company_name}")
            matching_rows = tracking_df[tracking_df['company_name'] == company_name]
            
            if not matching_rows.empty:
                # Update the most recent matching entry
                idx = matching_rows.index[0]
                current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Only update if this is a new article (different link)
                existing_link = tracking_df.loc[idx, 'most_recent_update_link']
                if pd.isna(existing_link) or str(existing_link) != article['article_link']:
                    tracking_df.loc[idx, 'most_recent_update'] = article['article_summary']
                    tracking_df.loc[idx, 'most_recent_update_link'] = article['article_link']
                    tracking_df.loc[idx, 'update_date'] = article['article_date']
                    tracking_df.loc[idx, 'last_checked'] = current_timestamp
                    entries_to_update.append(company_name)
                    print(f"  ✓ Updated with new article")
                else:
                    print(f"  ⚠️  Article already in database, skipping")
            else:
                # Company name matched but couldn't find row (shouldn't happen)
                print(f"  ⚠️  Warning: Company found in set but not in DataFrame")
        else:
            # Create new entry
            print(f"Creating new entry for: {company_name}")
            
            # Generate a unique page_id for new entries (using hash of company name + timestamp)
            import hashlib
            page_id_str = f"{company_name}_{datetime.now().isoformat()}"
            page_id = hashlib.md5(page_id_str.encode()).hexdigest()
            
            # Create new entry matching tracking database structure
            # Only include columns that exist in the tracking_db schema
            new_entry = {
                'page_id': page_id,
                'company_name': company_name,
                'founder': article['founder'],
                'priority': article['priority'],
                'website': article.get('website', ''),
                'most_recent_update': article['article_summary'],
                'most_recent_update_link': article['article_link'],
                'update_date': article['article_date'],
                'last_checked': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'co_founder': '',  # Initialize empty fields (note: co_founder, not co-founder)
                'personal_linkedin': '',
                'page_links': None
            }
            
            new_entries.append(new_entry)
            print(f"  ✓ Created new entry (page_id: {page_id[:12]}...)")
    
    print()
    
    # Add new entries to tracking DataFrame
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        if tracking_df.empty:
            tracking_df = new_df
        else:
            # Ensure all columns exist in both DataFrames
            for col in tracking_df.columns:
                if col not in new_df.columns:
                    new_df[col] = ''
            for col in new_df.columns:
                if col not in tracking_df.columns:
                    tracking_df[col] = ''
            
            tracking_df = pd.concat([tracking_df, new_df], ignore_index=True)
    
    # Save to tracking database
    if new_entries or entries_to_update:
        print("=" * 80)
        print("SAVING TO TRACKING DATABASE")
        print("=" * 80)
        print(f"New entries: {len(new_entries)}")
        print(f"Updated entries: {len(entries_to_update)}")
        print()
        
        # Filter to only valid columns that exist in the tracking_db schema
        # These are the actual columns in the schema (excluding auto-generated ones: id, created_at, updated_at)
        valid_columns = [
            'page_id', 'company_name', 'priority', 'founder', 'website', 
            'page_links', 'co_founder', 'personal_linkedin', 
            'most_recent_update', 'most_recent_update_link', 'update_date', 'last_checked'
        ]
        
        # Only keep columns that exist in both tracking_df and valid_columns
        columns_to_keep = [col for col in valid_columns if col in tracking_df.columns]
        
        # Also keep 'co-founder' if it exists (will be renamed by save_tracking_to_supabase)
        if 'co-founder' in tracking_df.columns and 'co-founder' not in columns_to_keep:
            columns_to_keep.append('co-founder')
        
        # Filter DataFrame to only valid columns
        tracking_df = tracking_df[columns_to_keep].copy()
        
        # Save to Supabase
        print("Saving to Supabase...")
        save_success = save_tracking_to_supabase(tracking_df)
        
        # Also save to CSV backup
        tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
        os.makedirs(os.path.dirname(tracking_db_path), exist_ok=True)
        tracking_df.to_csv(tracking_db_path, index=False)
        print(f"✓ Saved to CSV backup: {tracking_db_path}")
        
        if save_success:
            print("✓ Successfully saved to Supabase")
        else:
            print("⚠️  Warning: Supabase save failed, but CSV backup saved")
        
        print()
        print("=" * 80)
        print("COMPLETE")
        print("=" * 80)
        print(f"Total new entries added: {len(new_entries)}")
        print(f"Total entries updated: {len(entries_to_update)}")
    else:
        print("⚠️  No changes to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add pipeline news articles to tracking database')
    parser.add_argument('--csv', type=str, help='Path to pipeline news articles CSV file (default: finds latest)')
    parser.add_argument('--days', type=int, default=6, help='Number of days to look back (default: 6)')
    args = parser.parse_args()
    
    process_and_add_articles(csv_file=args.csv, days_back=args.days)
