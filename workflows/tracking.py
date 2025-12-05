import os
import sys
import time
import json
import hashlib
import re
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

# Add the parent directory to the Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.notion import import_tracked
from services.deal_processing import clean_company_name, score_result

# Load environment variables
load_dotenv()

id = "912974853b494f98a5652fcbff3ad795"

# Get workspace root directory (parent of workflows directory)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to store processed alerts (workspace-relative)
PROCESSED_ALERTS_FILE = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'processed_alerts.json')

def load_processed_alerts():
    """Load the list of already processed alert IDs from file."""
    try:
        if os.path.exists(PROCESSED_ALERTS_FILE):
            with open(PROCESSED_ALERTS_FILE, 'r') as f:
                return json.load(f)
        else:
            return {'processed_ids': []}
    except Exception as e:
        print(f"Error loading processed alerts: {e}")
        return {'processed_ids': []}

def save_processed_alerts(processed_alerts):
    """Save the list of processed alert IDs to file."""
    try:
        os.makedirs(os.path.dirname(PROCESSED_ALERTS_FILE), exist_ok=True)
        with open(PROCESSED_ALERTS_FILE, 'w') as f:
            json.dump(processed_alerts, f, indent=2)
    except Exception as e:
        print(f"Error saving processed alerts: {e}")

def generate_alert_id(alert):
    """Generate a unique ID for an alert based on its content."""
    # Create a string with key alert information
    id_string = f"{alert.get('subject', '')}-{alert.get('date', '')}-{alert.get('snippet', '')}"
    # Generate a hash of this string
    return hashlib.md5(id_string.encode()).hexdigest()

def add_companies_to_news_alert():
    tracked = import_tracked(id)
    high = tracked[tracked['priority'] == 'High']
    medium = tracked[tracked['priority'] == 'Medium']
    low = tracked[tracked['priority'] == 'Low']
    no_priority = tracked[tracked['priority'] == 'No priority']

    # Combine high and medium priority companies
    priority_companies = pd.concat([high, medium, low, no_priority])
    
    # Create search terms by combining company name with founder name and "startup"
    # First ensure the founder column exists and handle NaN values
    if 'founder' in priority_companies.columns:
        # Create a new column with combined search terms
        priority_companies['search_term'] = priority_companies.apply(
            lambda row: f"{row['company_name']} {row['founder']} startup" 
            if pd.notna(row['founder']) else f"{row['company_name']} startup", 
            axis=1
        )
    elif 'founders' in priority_companies.columns:
        # Create a new column with combined search terms
        priority_companies['search_term'] = priority_companies.apply(
            lambda row: f"{row['company_name']} {row['founders']} startup" 
            if pd.notna(row['founders']) else f"{row['company_name']} startup", 
            axis=1
        )
    else:
        # If no founder column, just use company name
        priority_companies['search_term'] = priority_companies['company_name'] + " startup"
    
    # Extract the list of search terms
    search_terms = priority_companies['search_term'].tolist()
    
    # Add alerts for these search terms
    print("About to add alerts")
    # TODO: Implement add_company_alerts function or import from appropriate module
    try:
        add_company_alerts(search_terms)
        print(f"Added alerts for {len(search_terms)} companies")
    except NameError:
        print(f"⚠️  Warning: add_company_alerts() function is not defined. Skipping alert creation.")
        print(f"   Would have added alerts for {len(search_terms)} companies")
    
    return search_terms

def find_relevant_links():
    tracked = import_tracked(id, page_text=True)
    high = tracked[tracked['priority'] == 'High']
    medium = tracked[tracked['priority'] == 'Medium']
    low = tracked[tracked['priority'] == 'Low']
    no_priority = tracked[tracked['priority'] == 'No priority']

    # Combine high and medium priority companies
    priority_companies = pd.concat([high, medium, low, no_priority])

    priority_companies["links_to_check"] = ""

    for index, row in priority_companies.iterrows():
        name = row['founder']
        company_name = row['company_name']
        # TODO: Implement find_linkedin_profile function or import from appropriate module
        try:
            linkedin_profile = find_linkedin_profile(name, company_name)
        except NameError:
            print(f"⚠️  Warning: find_linkedin_profile() function is not defined. Using empty value.")
            linkedin_profile = None
        
        links_dict = {"linkedin": linkedin_profile,
                      "website": row['website'],
                      "company_linkedin": row['company_linkedin']}
        priority_companies.at[index, 'links_to_check'] = links_dict

    return priority_companies

def process_tracking_list(tracking):
        
        # Count statistics
        personal_linkedin_count = 0
        cofounder_count = 0

        for index, row in tracking.iterrows():
            # First check if the website or founder field contains LinkedIn URLs
            if pd.notna(row['website']):
                website = str(row['website'])
                if 'linkedin.com/in' in website:
                    tracking.at[index, 'personal_linkedin'] = website
                    personal_linkedin_count += 1
                    
            if pd.notna(row['founder']):
                founder = str(row['founder'])
                if 'linkedin.com/in' in founder:
                    # Extract the LinkedIn URL using regex
                    import re
                    linkedin_urls = re.findall(r'https?://(?:www\.)?linkedin\.com/in/[^\s,)"\']+', founder)
                    if linkedin_urls:
                        tracking.at[index, 'personal_linkedin'] = linkedin_urls[0]
                        personal_linkedin_count += 1
                        # Remove the LinkedIn URL from the founder field to clean it up
                        tracking.at[index, 'founder'] = re.sub(r'https?://(?:www\.)?linkedin\.com/in/[^\s,)"\']+', '', founder).strip()
            
            # Handle the links from page_links
            if pd.notna(row['page_links']) and row['page_links'] != 'None':
                try:
                    # Convert the string representation of a Python dict to an actual dict
                    links_dict = eval(row['page_links'])
                    
                    # Identify personal and/or company linkedin url and move to the right key in dict
                    if 'personal_linkedin' in links_dict and links_dict['personal_linkedin'] and not tracking.at[index, 'personal_linkedin']:
                        tracking.at[index, 'personal_linkedin'] = links_dict['personal_linkedin']
                        personal_linkedin_count += 1
                    
                    # If we don't have specific LinkedIn URLs but have other URLs, check if any are LinkedIn
                    if 'other' in links_dict and links_dict['other']:
                        for url in links_dict['other']:
                            if isinstance(url, str):  # Ensure URL is a string
                                if 'linkedin.com/in' in url and not tracking.at[index, 'personal_linkedin']:
                                    tracking.at[index, 'personal_linkedin'] = url
                                    personal_linkedin_count += 1
                except Exception as e:
                    print(f"Error parsing links for row {index}: {e}")
            
            # Handle co-founders
            if pd.notna(row['founder']):
                founder_name = str(row['founder'])
                
                # Check for commas indicating multiple founders
                if ', ' in founder_name:
                    founders = [name.strip() for name in founder_name.split(', ')]
                    if len(founders) >= 2:
                        # Update the founder with the first name
                        tracking.at[index, 'founder'] = founders[0]
                        # Add the second name to co-founder
                        tracking.at[index, 'co-founder'] = founders[1]
                        cofounder_count += 1

                if ' + ' in founder_name:
                    founders = [name.strip() for name in founder_name.split(' + ')]
                    if len(founders) >= 2:
                        # Update the founder with the first name
                        tracking.at[index, 'founder'] = founders[0]
                        # Add the second name to co-founder
                        tracking.at[index, 'co-founder'] = founders[1]
                        cofounder_count += 1

        return tracking

def initialize_tracking_csv():
    """
    Initialize the tracking CSV file from Notion.
    This function should only be called once when the file doesn't exist.
    
    Returns:
        pandas.DataFrame: DataFrame with all tracking data initialized
    """
    print("Loading tracking data from Notion...")
    tracking_df = import_tracked(id, page_text=True)
    print(f"Loaded {len(tracking_df)} companies from Notion database")
    
    # Process the tracking list to extract LinkedIn URLs, handle co-founders, etc.
    print("Processing and enriching company data...")
    tracking_df = process_tracking_list(tracking_df)
    
    # Initialize update tracking columns
    if 'most_recent_update' not in tracking_df.columns:
        tracking_df['most_recent_update'] = ''
    if 'most_recent_update_link' not in tracking_df.columns:
        tracking_df['most_recent_update_link'] = ''
    if 'update_date' not in tracking_df.columns:
        tracking_df['update_date'] = ''
    if 'last_checked' not in tracking_df.columns:
        tracking_df['last_checked'] = ''
    
    # Save to CSV
    tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
    os.makedirs(os.path.dirname(tracking_db_path), exist_ok=True)
    tracking_df.to_csv(tracking_db_path, index=False)
    print(f"✅ Initialized tracking database with {len(tracking_df)} companies")
    
    return tracking_df

def look_for_updates(profile_dict, company_info):
    # Compare company name
    if "stealth" not in str(profile_dict['company_name_1']).lower():
        linkedin_company = profile_dict['company_name_1'].split(' · ')[0]
        if company_info['company_name'].lower().strip() not in linkedin_company.lower().strip() or linkedin_company.lower().strip() not in company_info['company_name'].lower().strip():
            if company_info['company_name'].lower().strip() in profile_dict['position_1'].lower().strip():
                print("Company name matches:")
                print("Profile: ", linkedin_company)
                print("Company info: ", company_info['position_1'])
            else:
                print("Company name mismatch:")
                print("Profile: ", linkedin_company)
                print("Company info: ", company_info['company_name'])
                print("For reference, here is the profile dict:")
                print(profile_dict)
        else:
            print("Company name matches:")
            print("Profile: ", linkedin_company)
            print("Company info: ", company_info['company_name'])

def parse_list_for_updates():
    """
    Process LinkedIn profiles from the tracking list CSV file in a safe way that avoids timeouts.
    Uses fresh browser instances for each profile, proper error handling, and delays between requests.
    """
    # Use workspace-relative path or fallback to tracking_db if available
    notion_export_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
    
    # Fallback to old path if tracking_db doesn't exist (for backward compatibility)
    if not os.path.exists(notion_export_path):
        old_path = os.path.join(workspace_root, 'data', 'tracking_names', 'notion_export_processed.csv')
        if os.path.exists(old_path):
            notion_export_path = old_path
        else:
            # Try loading from Notion if no file exists
            tracking = import_tracked(id, page_text=True)
            tracking = process_tracking_list(tracking)
            return tracking
    
    tracking = pd.read_csv(notion_export_path)
    
    processed_count = 0
    max_profiles = 10  # Limit to 10 profiles per run to avoid overloading
    delay_seconds = 5  # Increased delay between profile requests
    
    print(f"Found {len(tracking)} profiles in tracking list. Processing up to {max_profiles}.")
    
    try:
        for index, row in tracking.iterrows():
            if processed_count >= max_profiles:
                print(f"Reached maximum of {max_profiles} profiles. Stopping.")
                break
                
            name = row['founder']
            company_name = row['company_name']
            personal_linkedin = row['personal_linkedin']
            
            if not personal_linkedin or personal_linkedin == "" or str(personal_linkedin) == "nan" or str(personal_linkedin) == "None":
                print(f"Skipping {name} - No LinkedIn URL provided")
                continue
                
            print(f"Processing {processed_count+1}/{max_profiles}: {name} from {company_name}")
            print(f"LinkedIn URL: {personal_linkedin}")
            
            # Process the profile with a fresh browser instance each time
            try:
                # TODO: Implement process_profile function or import from appropriate module
                # Always use a fresh browser instance (reuse_browser=False)
                try:
                    profile_dict = process_profile(personal_linkedin, reuse_browser=False)
                except NameError:
                    print(f"⚠️  Warning: process_profile() function is not defined. Skipping profile processing.")
                    continue
                
                # Check if we got valid profile data
                if profile_dict and not isinstance(profile_dict, str):
                    # Only process if we got valid data (not an error string)
                    look_for_updates(profile_dict, row)
                    processed_count += 1
                    
                    # Update the tracking CSV with the last checked date
                    tracking.at[index, 'last_checked'] = datetime.now().strftime('%Y-%m-%d')
                    
                    # Save progress after each successful profile to avoid losing data
                    if processed_count % 2 == 0:  # Save every 2 profiles
                        tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
                        os.makedirs(os.path.dirname(tracking_db_path), exist_ok=True)
                        tracking.to_csv(tracking_db_path, index=False)
                else:
                    error_msg = profile_dict if isinstance(profile_dict, str) else "Unknown error"
                    print(f"Could not retrieve profile data for {name}: {error_msg}")
                
                end_time = time.time()
            
            except Exception as e:
                print(f"Error processing {name}'s profile: {str(e)}")
                # Continue to the next profile instead of failing the entire batch
            
            # Add a delay between requests to avoid rate limiting
            time.sleep(delay_seconds)
        
        # Save the updated tracking CSV with last checked dates
        tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
        os.makedirs(os.path.dirname(tracking_db_path), exist_ok=True)
        tracking.to_csv(tracking_db_path, index=False)
        print(f"Successfully processed {processed_count} profiles.")
        
    except Exception as e:
        print(f"Unexpected error in profile processing batch: {str(e)}")
        # Save progress in case of unexpected errors
        tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
        os.makedirs(os.path.dirname(tracking_db_path), exist_ok=True)
        tracking.to_csv(tracking_db_path, index=False)
        print("Progress saved despite error.")
    
    return processed_count

def scrape_with_firecrawl():
    firecrawl_api_key = "fc-f80912783c0941a5b3c1f44cb24a3fa8"
    url = "https://www.visir.is/"
    app = FirecrawlApp(api_key=firecrawl_api_key)
    scrape_status = app.scrape_url(
        url, 
        formats=['markdown']
    )            
    print(scrape_status)
    # Check if we received HTML content
    markdown_content = scrape_status.model_dump().get("markdown", "")
    print(markdown_content)

def parse_raw_alert(email):
    from email.parser import BytesParser
    from email import policy

    msg = BytesParser(policy=policy.default).parsebytes(email['raw_email'])
    text_content = ""
    links = []
    
    # Process the email parts
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            
            # Extract text content
            if content_type == 'text/plain' and not text_content:
                text_content = part.get_content()
            
            # Extract links from HTML content
            if content_type == 'text/html':
                html_content = part.get_content()
                html_soup = BeautifulSoup(html_content, 'html.parser')
                # Find all anchor tags and extract href attributes
                for link in html_soup.find_all('a'):
                    href = link.get('href')
                    if href and href.startswith(('http://', 'https://')):
                        links.append(href)
    else:
        text_content = msg.get_content()
        # Check if content is HTML
        if msg.get_content_type() == 'text/html':
            html_soup = BeautifulSoup(text_content, 'html.parser')
            # Find all anchor tags and extract href attributes
            for link in html_soup.find_all('a'):
                href = link.get('href')
                if href and href.startswith(('http://', 'https://')):
                    links.append(href)
    
    # Extract text content
    soup = BeautifulSoup(text_content, 'html.parser')
    text = soup.get_text(separator="\n")
    
    # If no links found in HTML, try to find URLs in plain text using regex
    if not links:
        import re
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        text_links = re.findall(url_pattern, text_content)
        links.extend(text_links)
    
    # Return both text content and links
    return text, links

def check_if_true_alert(alert):
    from services.groq_api import get_groq_response
    
    # Extract alert text and search keywords
    alert_text = alert.get('text', '')
    search_keywords = alert.get('search_keywords', '')
    
    # If we don't have the necessary information, return a dictionary with is_relevant=False
    if not alert_text or not search_keywords:
        return {"is_relevant": False, "analysis": "Missing alert text or search keywords"}
    
    prompt = f"""
    You are analyzing a Google Alert to determine if it is truly relevant to a specific early stage company or founder we are tracking.

    ALERT TEXT:
    {alert_text}

    SEARCH KEYWORDS USED FOR THIS ALERT:
    {search_keywords}

    Task: Determine if this alert is genuinely about the early stage company/founder mentioned in the search keywords, or if it's just a coincidental mention of similar terms.

    Guidelines for assessment:
    1. A relevant alert will mention the specific company name AND/OR founder name in a context that indicates the alert is primarily about them.
    2. An irrelevant alert will only mention the search keywords in passing, as part of another entity's name, or in an unrelated context.
    3. If the alert mentions the founder name with their correct company affiliation, it's highly relevant.

    Examples of RELEVANT alerts:
    - News specifically about the company's product launches, funding, or business activities
    - Interviews or features about the founder
    - Articles where the company/founder is a primary subject

    Examples of IRRELEVANT alerts:
    - The search keyword appears only as part of another entity's name (e.g., searching for "Basis AI" but the alert is about "Basis Set Ventures")
    - The keyword is used in a generic sense unrelated to the specific company we're tracking

    Analyze the alert text and respond with ONLY "RELEVANT" or "IRRELEVANT" followed by a one-sentence explanation.
    """
    
    response = get_groq_response(prompt)
    
    # Parse the response to get a boolean result - only consider it relevant if it starts with "RELEVANT"
    is_relevant = response.strip().upper().startswith("RELEVANT")
    
    # Return both the boolean result and the full response for logging/debugging
    return {"is_relevant": is_relevant, "analysis": response}

def check_google_alerts(days=4):
    from services.google_client import get_emails_with_label
    start_date = (datetime.today() - timedelta(days=days)).strftime('%Y/%m/%d')
    latest_alerts = get_emails_with_label(start_date=start_date, label_name="google alerts")
    
    # Load the list of already processed alerts
    processed_alerts = load_processed_alerts()
    processed_ids = set(processed_alerts['processed_ids'])
    
    relevant_alerts = []
    irrelevant_count = 0
    already_processed_count = 0
    
    print(f"Processing {len(latest_alerts)} alerts from the last {days} days...")
    
    for alert in latest_alerts:
        
        text, links = parse_raw_alert(alert)
        
        # Extract search keywords from the email subject or metadata
        # Handle different formats of Google Alerts
        if "[" in text and "]" in text:
            search_keywords = text.split("[")[1].split("]")[0].strip()
            text = text.split("]")[1].strip()
        else:
            # Fallback if the expected format is not found
            search_keywords = alert.get('subject', '').split('Daily update')[0].strip()
        
        # Create alert object with all necessary information
        alert_obj = {
            'text': text,
            'links': links,
            'search_keywords': search_keywords,
            'company_name': search_keywords.split(" ")[0],
            'date': alert.get('date', '')
        }
        
        # Check if this is a relevant alert
        relevance_check = check_if_true_alert(alert_obj)
        
        if relevance_check['is_relevant']:
            # Add the relevance analysis to the alert object
            alert_obj['relevance_analysis'] = relevance_check['analysis']
            relevant_alerts.append(alert_obj)
        else:
            irrelevant_count += 1    
    
    print(f"\nSummary: Found {len(relevant_alerts)} relevant alerts, filtered out {irrelevant_count} irrelevant ones.")
    return relevant_alerts

def update_tracking_database(tracking_df, relevant_alert):
    # Ensure the DataFrame has the necessary columns with string dtypes
    if 'most_recent_update' not in tracking_df.columns:
        tracking_df['most_recent_update'] = ''
    if 'most_recent_update_link' not in tracking_df.columns:
        tracking_df['most_recent_update_link'] = ''
    
    # Convert columns to string type if they aren't already
    if tracking_df['most_recent_update'].dtype != 'object':
        tracking_df['most_recent_update'] = tracking_df['most_recent_update'].astype(str)
    if tracking_df['most_recent_update_link'].dtype != 'object':
        tracking_df['most_recent_update_link'] = tracking_df['most_recent_update_link'].astype(str)
    
    # Find the indices of rows in the tracking DataFrame that match the relevant alert
    matching_indices = tracking_df.index[tracking_df['company_name'] == relevant_alert['company_name']].tolist()
    
    if not matching_indices:
        # Try with a different approach if no matches found
        try:
            double_company_name = " ".join(relevant_alert['search_keywords'].split(" ")[:1])
            matching_indices = tracking_df.index[tracking_df['company_name'] == double_company_name].tolist()
        except Exception as e:
            print(f"Error processing search keywords: {e}")
            matching_indices = []

    if not matching_indices:
        print(f"No matching row found for {relevant_alert['company_name']}")
        return tracking_df
    
    # Update the matching rows with the relevant alert information using .loc
    for idx in matching_indices:
        tracking_df.loc[idx, 'most_recent_update'] = str(relevant_alert['relevance_analysis'])
        tracking_df.loc[idx, 'most_recent_update_link'] = str(relevant_alert['links'][1])
        print("Added link: ", relevant_alert['links'][1])

    print("Updating ", relevant_alert['company_name'], " with ", relevant_alert['relevance_analysis'])
    return tracking_df

def get_detailed_info_on_alert(link, company_name=None, search_keywords=None):
    """Extract information from a Google Alert article using Parallel API extract, 
    or fall back to Parallel API search if extraction fails.
    
    Args:
        link (str): The Google Alert link to the article
        company_name (str): Name of the company
        search_keywords (str): Search keywords used for the alert
        
    Returns:
        str: Summary of the article or search results
    """
    import requests
    from urllib.parse import urlparse, parse_qs
    from services.parallel_client import Parallel
    from services.openai_api import ask_monty
    import os
    
    # Extract the actual article URL from the Google Alert link
    try:
        parsed_url = urlparse(link)
        if 'google.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if 'url' in query_params:
                actual_url = query_params['url'][0]
            else:
                response = requests.head(link, allow_redirects=True, timeout=10)
                actual_url = response.url
        else:
            actual_url = link
            
        print(f"  Extracted actual URL: {actual_url}")
    except Exception as e:
        print(f"  Error extracting actual URL: {e}")
        actual_url = link  # Use original link as fallback
    
    # Prepare company information
    target_company = company_name if company_name else (search_keywords.split(" ")[0] if search_keywords else "the company")
    
    # Try Parallel API extract first
    try:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            print("  ⚠️  PARALLEL_API_KEY not found, falling back to search")
            return get_info_via_search(target_company, search_keywords)
        
        parallel_client = Parallel(api_key=api_key)
        
        # Create objective for extraction
        objective = f"""Extract key information about {target_company} from this article. 
        Focus on: what happened or was announced, any metrics or milestones, funding announcements, 
        product launches, expansions, or business updates. Return a concise 2-3 sentence summary 
        of what this article means for {target_company}."""
        
        print(f"  Attempting to extract information from article using Parallel API...")
        extract_result = parallel_client.beta.extract(
            urls=[actual_url],
            objective=objective,
            excerpts=True,
            full_content=False
        )
        
        # Check if extraction was successful
        if extract_result.results and len(extract_result.results) > 0:
            result = extract_result.results[0]
            
            # Check for errors (locked/paywall, no content, etc.)
            if extract_result.errors:
                error_msg = str(extract_result.errors[0]) if extract_result.errors else ""
                if any(keyword in error_msg.lower() for keyword in ['locked', 'paywall', 'subscription', 'access denied', 'premium']):
                    print(f"  ⚠️  Article appears to be locked/paywalled, falling back to search")
                    return get_info_via_search(target_company, search_keywords)
            
            # Extract excerpts from response
            excerpts = []
            if isinstance(result, dict):
                excerpts = result.get('excerpts', [])
            elif hasattr(result, 'excerpts'):
                excerpts = result.excerpts
            
            # Combine excerpts into text
            if excerpts and isinstance(excerpts, list) and len(excerpts) > 0:
                text_content = ' '.join(str(ex) for ex in excerpts if ex)
                
                if text_content and len(text_content.strip()) > 50:  # Ensure we got meaningful content
                    # Use AI to create a focused summary
                    # Limit content to avoid token issues
                    limited_content = text_content[:8000]
                    summary_prompt = f"""Create a concise 2-3 sentence summary about what this article means for {target_company}.
                    
                    Focus on:
                    - What happened or was announced
                    - Any specific metrics, expansion, milestone, or business impact
                    - Funding announcements (include amount and round if mentioned)
                    - Product launches or expansions
                    
                    Article excerpts:
                    {limited_content}
                    """
                    
                    try:
                        summary = ask_monty(summary_prompt, "", max_tokens=200)
                        print(f"  ✓ Successfully extracted information from article")
                        return summary
                    except Exception as e:
                        print(f"  ⚠️  Error creating summary from excerpts: {e}")
                        # Fall back to search
                        return get_info_via_search(target_company, search_keywords)
            
            # No excerpts or empty excerpts - fall back to search
            print(f"  ⚠️  No content extracted from article, falling back to search")
            return get_info_via_search(target_company, search_keywords)
        else:
            # No results - fall back to search
            print(f"  ⚠️  No results from extract, falling back to search")
            return get_info_via_search(target_company, search_keywords)
            
    except Exception as e:
        print(f"  ⚠️  Error with Parallel API extract: {e}, falling back to search")
        return get_info_via_search(target_company, search_keywords)


def get_info_via_search(company_name, search_keywords=None):
    """Fallback: Use Parallel API search to find updates about the company.
    
    Args:
        company_name (str): Name of the company
        search_keywords (str): Optional search keywords
        
    Returns:
        str: Summary of search results
    """
    from services.parallel_client import Parallel
    from services.openai_api import ask_monty
    import os
    
    try:
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            return f"Update found for {company_name} (unable to get details - API key missing)"
        
        parallel_client = Parallel(api_key=api_key)
        
        # Build search query
        if search_keywords:
            search_query = f"{search_keywords} startup news updates"
        else:
            search_query = f"{company_name} startup news updates"
        
        objective = f"Find recent news articles or updates about {company_name}. Focus on funding announcements, product launches, business milestones, or significant company updates."
        
        print(f"  Searching for updates about {company_name} using Parallel API...")
        search_results = parallel_client.beta.search(
            mode="one-shot",
            search_queries=[search_query],
            max_results=5,
            objective=objective,
            max_chars_per_result=5000
        )
        
        # Extract results
        results_text = []
        if isinstance(search_results, dict) and "results" in search_results:
            for result in search_results["results"][:3]:  # Use top 3 results
                if isinstance(result, dict):
                    title = result.get("title", "")
                    excerpts = result.get("excerpts", [])
                    if excerpts and isinstance(excerpts, list):
                        snippet = " ".join(str(ex) for ex in excerpts[:2] if ex)  # First 2 excerpts
                    else:
                        snippet = result.get("snippet", "") or result.get("description", "")
                    
                    if title or snippet:
                        results_text.append(f"Title: {title}\nContent: {snippet}")
        
        if results_text:
            # Use AI to create a summary from search results
            # Join results outside f-string to avoid backslash issues
            separator = '\n\n---\n\n'
            joined_results = separator.join(results_text)
            summary_prompt = f"""Based on these search results, create a concise 2-3 sentence summary about recent updates for {company_name}.
            
            Focus on the most relevant and recent information. If there are funding announcements, product launches, 
            or significant business updates, prioritize those.
            
            Search results:
            {joined_results}
            """
            
            try:
                summary = ask_monty(summary_prompt, "", max_tokens=200)
                print(f"  ✓ Found updates via search")
                return summary
            except Exception as e:
                print(f"  ⚠️  Error creating summary from search: {e}")
                return f"Update found for {company_name} (details unavailable)"
        else:
            return f"Update found for {company_name} (no detailed information available)"
            
    except Exception as e:
        print(f"  ⚠️  Error with Parallel API search: {e}")
        return f"Update found for {company_name} (unable to get details)"

def process_alerts(tracking_df, relevant_alerts):
    # Ensure all update tracking columns exist and are string type
    if 'most_recent_update' not in tracking_df.columns:
        tracking_df['most_recent_update'] = ''
    if 'most_recent_update_link' not in tracking_df.columns:
        tracking_df['most_recent_update_link'] = ''
    if 'update_date' not in tracking_df.columns:
        tracking_df['update_date'] = ''
    if 'last_checked' not in tracking_df.columns:
        tracking_df['last_checked'] = ''
    
    # Convert columns to string type if they aren't already
    if tracking_df['most_recent_update'].dtype != 'object':
        tracking_df['most_recent_update'] = tracking_df['most_recent_update'].astype(str)
    if tracking_df['most_recent_update_link'].dtype != 'object':
        tracking_df['most_recent_update_link'] = tracking_df['most_recent_update_link'].astype(str)
    if tracking_df['update_date'].dtype != 'object':
        tracking_df['update_date'] = tracking_df['update_date'].astype(str)
    if tracking_df['last_checked'].dtype != 'object':
        tracking_df['last_checked'] = tracking_df['last_checked'].astype(str)
    
    # Track which companies were just updated from alerts (to skip in funding search)
    companies_updated_from_alerts = set()
    
    # Process each alert
    processed_count = 0
    skipped_count = 0
    
    for alert in relevant_alerts:
        # Find the matching company row
        company_rows = tracking_df[tracking_df['company_name'] == alert['company_name']]
        
        if company_rows.empty:
            print(f"No matching company found for {alert['company_name']}. Trying with double company name")
            two_words = alert['search_keywords'].split(" ")[0:2]
            two_word_company = " ".join(two_words)
            company_rows = tracking_df[tracking_df['company_name'] == two_word_company]
            if company_rows.empty:
                print(f"No matching company found for {alert['company_name']}. Trying with triple company name")
                three_words = alert['search_keywords'].split(" ")[0:3]
                three_word_company = " ".join(three_words)
                company_rows = tracking_df[tracking_df['company_name'] == three_word_company]
                if company_rows.empty:
                    print(f"No matching company found for {alert['company_name']}. Skipping alert")
                else:
                    print(f"Found match with three-word name: {three_word_company}")
                    alert['company_name'] = three_word_company
            else:
                # Update the company name in the alert to the matched two-word company name
                print(f"Found match with two-word name: {two_word_company}")
                alert['company_name'] = two_word_company
            
        # Check if this link is already in the tracking DataFrame
        alert_link = alert['links'][1]
        already_processed = False
        
        for idx in company_rows.index:
            existing_link = tracking_df.loc[idx, 'most_recent_update_link']
            # Check if existing_link is a valid string and not NaN or empty
            if isinstance(existing_link, str) and existing_link.strip():
                if alert_link in existing_link:
                    print(f"Skipping already processed alert for {alert['company_name']}")
                    skipped_count += 1
                    already_processed = True
                    break
        
        if already_processed:
            continue
        
        # Process new alert
        print(f"Processing new alert for {alert['company_name']}")
        
        # Get detailed information about the alert
        summary = get_detailed_info_on_alert(
            link=alert_link,
            company_name=alert['company_name'],
            search_keywords=alert.get('search_keywords', '')
        )
        
        # Update the tracking DataFrame with the summary, link, and dates
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Try to extract date from alert, otherwise use current date
        alert_date = alert.get('date', '')
        update_date_str = datetime.now().strftime('%Y-%m-%d')  # Default to current date
        
        if alert_date:
            try:
                # Try to parse the alert date and format it
                if isinstance(alert_date, str):
                    # Try common date formats
                    date_parsed = False
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            parsed_date = datetime.strptime(alert_date.split()[0], fmt)
                            update_date_str = parsed_date.strftime('%Y-%m-%d')
                            date_parsed = True
                            break
                        except ValueError:
                            continue
                    # If parsing fails, update_date_str already has default value
            except Exception:
                # If any error occurs, update_date_str already has default value
                pass
        
        for idx in company_rows.index:
            tracking_df.loc[idx, 'most_recent_update'] = summary
            # Store the alert link to avoid re-processing in the future
            tracking_df.loc[idx, 'most_recent_update_link'] = alert_link
            # Store the date when the update was found
            tracking_df.loc[idx, 'update_date'] = update_date_str
            # Update the last_checked timestamp
            tracking_df.loc[idx, 'last_checked'] = current_timestamp
            # Track that this company was just updated from alerts
            companies_updated_from_alerts.add(alert['company_name'])
        
        processed_count += 1
    
    print(f"\nSummary: Processed {processed_count} new alerts and skipped {skipped_count} already processed alerts.")
    
    # Note: We don't save here - let the main function save at the end to avoid multiple writes
    # The main function will save after both alerts and funding search are complete
    
    # Return both the updated DataFrame and the set of companies that were just updated
    return tracking_df, companies_updated_from_alerts

def search_funding_with_parallel(company_name, last_checked_date=None, num_results=10):
    """
    Search for funding announcements for a company using Parallel API.
    
    Args:
        company_name (str): Name of the company to search for
        last_checked_date (datetime, optional): Only return results after this date
        num_results (int): Number of search results to return
        
    Returns:
        dict: Dictionary with 'found' (bool), 'link' (str or None), 'title' (str or None), 
              and 'date' (str or None) if found
    """
    if not company_name or pd.isna(company_name):
        return {'found': False, 'link': None, 'title': None, 'date': None}
    
    # Clean company name for search
    company = clean_company_name(str(company_name))
    search_query = f"{company} startup funding raises"
    
    try:
        # Initialize Parallel API client
        from services.parallel_client import Parallel
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            print("Error: PARALLEL_API_KEY not found in environment variables")
            return {'found': False, 'link': None, 'title': None, 'date': None}
        
        # Use Parallel SDK interface - same format as get_results.py
        client = Parallel(api_key=api_key)
        if last_checked_date:
            objective = f"Find news articles about funding announcements for {company} startup published after {last_checked_date.strftime('%Y-%m-%d')}. Only return actual news articles about funding rounds, not company profile pages."
        else:
            objective = f"Find news articles about funding announcements for {company} startup. Only return actual news articles about funding rounds, not company profile pages."
        response_data = client.beta.search(
            mode="one-shot",
            search_queries=[search_query],
            max_results=num_results,
            objective=objective,
            max_chars_per_result=10000  # Get more content to analyze
        )
        
        # Parse response - structure may vary, so we'll handle it flexibly
        links = []
        
        # Try to extract results from response
        results = []
        if isinstance(response_data, dict):
            if "results" in response_data:
                results = response_data["results"]
            elif "data" in response_data:
                results = response_data["data"]
            elif "items" in response_data:
                results = response_data["items"]
            elif "search_results" in response_data:
                results = response_data["search_results"]
            else:
                results = [response_data]
        elif isinstance(response_data, list):
            results = response_data
        else:
            results = [response_data]
        
        # Funding-related keywords to look for
        funding_keywords = [
            "raised", "raises", "funding", "funded", "investment", "invested",
            "series a", "series b", "series c", "seed", "seed round", "pre-seed",
            "million", "billion", "funding round", "closed", "secured", "announces",
            "led by", "backed by", "investors", "venture capital", "vc"
        ]
        
        # URLs/domains to exclude (not funding announcements)
        exclude_patterns = [
            "linkedin.com/in",  # LinkedIn profiles
            "/our-team", "/team", "/about", "/contact",  # Company pages
            "crunchbase.com/organization",  # General Crunchbase pages (not announcements)
            "startup-seeker.com/company",  # Company listings (not announcements)
            "tracxn.com/d/companies",  # Company profile pages
            "pitchbook.com/profiles/company",  # Company profile pages
            "bouncewatch.com/explore/startup",  # Company listing pages
            "cryptorank.io/funds",  # Crypto funding pages (not startup funding)
        ]
        
        # Domains that are good sources for news articles
        news_domains = [
            "techcrunch.com", "venturebeat.com", "businesswire.com",
            "prnewswire.com", "forbes.com", "wsj.com", "bloomberg.com",
            "reuters.com", "fortune.com", "finsmes.com", "vcnewsdaily.com",
            "eu-startups.com", "techfundingnews.com", "siliconcanals.com",
            "pymnts.com", "finextra.com", "news.crunchbase.com",
        ]
        
        # Process each result
        for r in results:
            # Extract link, title, and snippet from various possible formats
            link = None
            title = None
            snippet = None
            date_str = None
            excerpts = []  # Parallel API provides excerpts array
            full_text = ""  # Combined text for analysis
            
            if isinstance(r, dict):
                link = r.get("url") or r.get("link") or r.get("href")
                title = r.get("title") or r.get("name") or r.get("headline")
                # Parallel API provides excerpts array with content
                excerpts = r.get("excerpts", [])
                if excerpts and isinstance(excerpts, list):
                    # Combine all excerpts into snippet
                    snippet = " ".join(str(ex) for ex in excerpts if ex)
                else:
                    # Fallback to other fields
                    snippet = r.get("snippet") or r.get("description") or r.get("text") or r.get("summary") or r.get("content")
                # Use publish_date from Parallel API response
                date_str = r.get("publish_date") or r.get("date") or r.get("published_date") or r.get("published")
            elif hasattr(r, '__dict__'):
                # Object with attributes
                link = getattr(r, 'url', None) or getattr(r, 'link', None) or getattr(r, 'href', None)
                title = getattr(r, 'title', None) or getattr(r, 'name', None) or getattr(r, 'headline', None)
                excerpts = getattr(r, 'excerpts', [])
                if excerpts and isinstance(excerpts, list):
                    snippet = " ".join(str(ex) for ex in excerpts if ex)
                else:
                    snippet = getattr(r, 'snippet', None) or getattr(r, 'description', None) or getattr(r, 'text', None) or getattr(r, 'content', None)
                date_str = getattr(r, 'publish_date', None) or getattr(r, 'date', None) or getattr(r, 'published_date', None) or getattr(r, 'published', None)
            
            if not link:
                continue
            
            # Skip excluded URL patterns
            if any(pattern in link.lower() for pattern in exclude_patterns):
                continue
            
            # Combine title and snippet for analysis
            full_text = f"{title or ''} {snippet or ''}".lower()
            
            # Strict company name matching - company must appear in the content
            # Normalize company name for matching
            company_lower = company.lower()
            company_words = company_lower.split()
            
            # Check if company name appears in title or snippet
            # Try exact match first, then word-by-word match for multi-word companies
            company_found = False
            if len(company_words) == 1:
                # Single word company - must appear as whole word
                pattern = r'\b' + re.escape(company_lower) + r'\b'
                if re.search(pattern, full_text):
                    company_found = True
            else:
                # Multi-word company - check if all significant words appear
                # Skip common words like "inc", "llc", "ltd", "corp"
                significant_words = [w for w in company_words if w not in ['inc', 'llc', 'ltd', 'corp', 'inc.', 'llc.', 'ltd.', 'corp.', 'the', 'a', 'an']]
                if significant_words:
                    # All significant words must appear
                    words_found = sum(1 for word in significant_words if word in full_text)
                    if words_found >= len(significant_words):
                        company_found = True
                else:
                    # Fallback: just check if company name appears
                    company_found = company_lower in full_text
            
            # Skip if company name not found in content
            if not company_found:
                continue
            
            # Check if this looks like a funding announcement
            has_funding_keywords = any(keyword in full_text for keyword in funding_keywords)
            
            # Skip if no funding keywords found
            if not has_funding_keywords:
                continue
            
            # Try to parse date from date_str or snippet
            result_date = None
            if date_str:
                try:
                    date_formats = [
                        '%Y-%m-%d',
                        '%B %d, %Y',
                        '%b %d, %Y',
                        '%d %B %Y',
                        '%d %b %Y',
                        '%Y/%m/%d',
                        '%m/%d/%Y',
                    ]
                    for fmt in date_formats:
                        try:
                            result_date = datetime.strptime(str(date_str), fmt)
                            break
                        except ValueError:
                            continue
                    if result_date is None:
                        result_date = pd.to_datetime(date_str).to_pydatetime()
                except:
                    pass
            
            # Also try to extract date from snippet if not found
            if not result_date and snippet:
                date_patterns = [
                    r'(\w+ \d{1,2}, \d{4})',
                    r'(\d{1,2}/\d{1,2}/\d{4})',
                    r'(\d{4}-\d{2}-\d{2})',
                ]
                for pattern in date_patterns:
                    match = re.search(pattern, snippet)
                    if match:
                        try:
                            result_date = pd.to_datetime(match.group(1)).to_pydatetime()
                            break
                        except:
                            continue
            
            # Check if result date is after last_checked_date - ONLY include if date is after last_checked_date
            if result_date and last_checked_date and result_date.date() > last_checked_date.date():
                # Check if this is a news article (not a company profile page)
                link_lower = link.lower()
                is_news_article = any(domain in link_lower for domain in news_domains)
                is_profile_page = any(pattern in link_lower for pattern in exclude_patterns)
                
                # Only include if it's a news article or if we can't determine (but not a profile page)
                if is_news_article or (not is_profile_page and not any(domain in link_lower for domain in ["tracxn.com", "pitchbook.com", "bouncewatch.com", "cryptorank.io"])):
                    links.append({
                        "link": link,
                        "title": title or "No title",
                        "snippet": snippet or "",
                        "date": result_date,
                        "full_text": full_text,
                        "is_news": is_news_article
                    })
            elif not last_checked_date:
                # No date filter provided, include all results that pass validation
                link_lower = link.lower()
                is_news_article = any(domain in link_lower for domain in news_domains)
                is_profile_page = any(pattern in link_lower for pattern in exclude_patterns)
                
                # Only include if it's a news article or if we can't determine (but not a profile page)
                if is_news_article or (not is_profile_page and not any(domain in link_lower for domain in ["tracxn.com", "pitchbook.com", "bouncewatch.com", "cryptorank.io"])):
                    links.append({
                        "link": link,
                        "title": title or "No title",
                        "snippet": snippet or "",
                        "date": result_date,
                        "full_text": full_text,
                        "is_news": is_news_article
                    })
        
        # Score and pick best link - same format as get_results.py
        if links:
            # Score each link - prioritize actual funding announcements with strict company matching
            scored_links = []
            for link_data in links:
                result_dict = {"link": link_data["link"], "title": link_data["title"]}
                score = score_result(result_dict, company, None, None)
                
                # Boost score for funding-related content
                full_text = link_data.get("full_text", "").lower()
                funding_keyword_count = sum(1 for keyword in funding_keywords if keyword in full_text)
                score += funding_keyword_count * 2  # Boost for funding keywords
                
                # Boost for company name appearing in title (stronger signal)
                title_lower = (link_data.get("title", "") or "").lower()
                company_lower = company.lower()
                if company_lower in title_lower:
                    score += 15
                
                # Boost for news articles (actual funding announcements)
                if link_data.get("is_news"):
                    score += 20
                
                # Boost for having a date after last_checked_date
                if link_data.get("date") and last_checked_date and link_data["date"].date() > last_checked_date.date():
                    score += 10
                
                # Penalize generic articles (list articles, industry overviews)
                generic_patterns = [
                    "biggest funding rounds",
                    "top 10",
                    "this week",
                    "this month",
                    "industry overview",
                    "sector analysis",
                    "have raised over",
                    "startups have raised",
                ]
                if any(pattern in full_text for pattern in generic_patterns):
                    score -= 20  # Heavy penalty for generic articles
                
                # Penalize for excluded patterns (shouldn't happen but just in case)
                link_lower = link_data["link"].lower()
                if any(pattern in link_lower for pattern in exclude_patterns):
                    score -= 20
                
                scored_links.append((score, link_data))
            
            # Sort by score
            scored_links.sort(reverse=True, key=lambda x: x[0])
            
            # Get best result - only return if score is positive (indicates it's likely a funding announcement)
            best_score, best_result = scored_links[0]
            
            # Additional validation: check if company name is clearly mentioned
            title_lower = (best_result.get("title", "") or "").lower()
            snippet_lower = (best_result.get("snippet", "") or "").lower()
            company_lower = company.lower()
            
            # Use word boundary matching for company name to avoid partial matches
            company_words = company_lower.split()
            significant_words = [w for w in company_words if w not in ['inc', 'llc', 'ltd', 'corp', 'inc.', 'llc.', 'ltd.', 'corp.', 'the', 'a', 'an']]
            
            # Check if company name appears in title or snippet with word boundaries
            company_in_title = False
            company_in_snippet = False
            
            if significant_words:
                # Check if all significant words appear in title
                title_words_found = sum(1 for word in significant_words if re.search(r'\b' + re.escape(word) + r'\b', title_lower))
                company_in_title = title_words_found >= len(significant_words)
                
                # Check if all significant words appear in snippet
                snippet_words_found = sum(1 for word in significant_words if re.search(r'\b' + re.escape(word) + r'\b', snippet_lower))
                company_in_snippet = snippet_words_found >= len(significant_words)
            else:
                # Fallback to simple substring match
                company_in_title = company_lower in title_lower
                company_in_snippet = company_lower in snippet_lower
            
            # Only return if:
            # 1. Score is positive AND
            # 2. Company name appears in title OR snippet AND
            # 3. Not a generic article
            if best_score > 0 and (company_in_title or company_in_snippet):
                # Double-check it's not a generic article
                full_text_check = f"{title_lower} {snippet_lower}"
                generic_patterns = [
                    "biggest funding rounds",
                    "top 10",
                    "this week",
                    "this month",
                    "industry overview",
                    "sector analysis",
                    "have raised over",
                    "startups have raised",
                ]
                is_generic = any(pattern in full_text_check for pattern in generic_patterns)
                
                if not is_generic:
                    # Final validation: ensure the article is actually about THIS company
                    # Check that company name appears prominently (in title is best)
                    # For multi-word companies, prefer title match over snippet match
                    if len(significant_words) > 1:
                        # Multi-word company: require title match for higher confidence
                        if not company_in_title:
                            # If not in title, check if snippet has very strong match
                            # Count how many times company words appear
                            snippet_mentions = sum(1 for word in significant_words if word in snippet_lower)
                            if snippet_mentions < len(significant_words):
                                # Company name not clearly mentioned, skip
                                return {'found': False, 'link': None, 'title': None, 'date': None}
                    
                    # CRITICAL: Only return if date is after last_checked_date (if provided)
                    if last_checked_date:
                        if best_result.get("date") and best_result["date"].date() > last_checked_date.date():
                            return {
                                'found': True,
                                'link': best_result["link"],
                                'title': best_result["title"],
                                'date': best_result["date"].strftime("%Y-%m-%d")
                            }
                        else:
                            # No date or date not after last_checked_date - don't return
                            return {'found': False, 'link': None, 'title': None, 'date': None}
                    else:
                        # No date filter, return best result
                        return {
                            'found': True,
                            'link': best_result["link"],
                            'title': best_result["title"],
                            'date': best_result["date"].strftime("%Y-%m-%d") if best_result.get("date") else "Date unknown - verify manually"
                        }
        
        return {'found': False, 'link': None, 'title': None, 'date': None}
        
    except Exception as e:
        print(f"Error searching for {company_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'found': False, 'link': None, 'title': None, 'date': None}


def search_company_updates_with_parallel(company_name, last_checked_date=None):
    """
    Search for recent news and updates about a company using Parallel API.
    This searches for any type of update (funding, product launches, partnerships, etc.)
    and returns a summary of what happened.
    
    Args:
        company_name (str): Name of the company to search for
        last_checked_date (datetime, optional): Only return results after this date
        
    Returns:
        dict: Dictionary with 'found' (bool), 'summary' (str), 'link' (str), 
              'title' (str), and 'date' (str) if found
    """
    if not company_name or pd.isna(company_name):
        return {'found': False, 'summary': None, 'link': None, 'title': None, 'date': None}
    
    # Clean company name for search
    company = clean_company_name(str(company_name))
    
    try:
        from services.parallel_client import Parallel
        from services.openai_api import ask_monty
        import os
        
        api_key = os.getenv("PARALLEL_API_KEY")
        if not api_key:
            print(f"  ⚠️  PARALLEL_API_KEY not found")
            return {'found': False, 'summary': None, 'link': None, 'title': None, 'date': None}
        
        parallel_client = Parallel(api_key=api_key)
        
        # Build search query for general company updates
        search_query = f"{company} startup news updates"
        
        # Create objective for finding recent updates
        if last_checked_date:
            objective = f"Find recent news articles and updates about {company} published after {last_checked_date.strftime('%Y-%m-%d')}. Look for any significant updates: funding announcements, product launches, partnerships, business milestones, expansions, or other notable company news. Only return actual news articles, not company profile pages."
        else:
            objective = f"Find recent news articles and updates about {company}. Look for any significant updates: funding announcements, product launches, partnerships, business milestones, expansions, or other notable company news. Only return actual news articles, not company profile pages."
        
        print(f"  Searching for updates about {company} using Parallel API...")
        search_results = parallel_client.beta.search(
            mode="one-shot",
            search_queries=[search_query],
            max_results=5,
            objective=objective,
            max_chars_per_result=10000
        )
        
        # Extract results
        results = []
        if isinstance(search_results, dict):
            if "results" in search_results:
                results = search_results["results"]
            elif "data" in search_results:
                results = search_results["data"]
            elif isinstance(search_results, list):
                results = search_results
        
        if not results:
            return {'found': False, 'summary': None, 'link': None, 'title': None, 'date': None}
        
        # URLs/domains to exclude (not news articles)
        exclude_patterns = [
            "linkedin.com/in",  # LinkedIn profiles
            "/our-team", "/team", "/about", "/contact",  # Company pages
            "crunchbase.com/organization",  # General Crunchbase pages
            "startup-seeker.com/company",  # Company listings
            "tracxn.com/d/companies",  # Company profile pages
            "pitchbook.com/profiles/company",  # Company profile pages
        ]
        
        # Domains that are good sources for news articles
        news_domains = [
            "techcrunch.com", "venturebeat.com", "businesswire.com",
            "prnewswire.com", "forbes.com", "wsj.com", "bloomberg.com",
            "reuters.com", "fortune.com", "finsmes.com", "vcnewsdaily.com",
            "eu-startups.com", "techfundingnews.com", "siliconcanals.com",
        ]
        
        # Process results and find the best one
        valid_results = []
        for r in results:
            link = None
            title = None
            snippet = None
            date_str = None
            excerpts = []
            
            if isinstance(r, dict):
                link = r.get("url") or r.get("link") or r.get("href")
                title = r.get("title") or r.get("name") or r.get("headline")
                excerpts = r.get("excerpts", [])
                if excerpts and isinstance(excerpts, list):
                    snippet = " ".join(str(ex) for ex in excerpts if ex)
                else:
                    snippet = r.get("snippet") or r.get("description") or r.get("text")
                date_str = r.get("publish_date") or r.get("date") or r.get("published_date")
            
            if not link or not title:
                continue
            
            # Skip excluded URL patterns
            if any(pattern in link.lower() for pattern in exclude_patterns):
                continue
            
            # Check if company name appears in title or snippet
            company_lower = company.lower()
            full_text = f"{title or ''} {snippet or ''}".lower()
            
            # Check if company name appears (word boundary matching for single word, or all words for multi-word)
            company_words = company_lower.split()
            significant_words = [w for w in company_words if w not in ['inc', 'llc', 'ltd', 'corp', 'inc.', 'llc.', 'ltd.', 'corp.', 'the', 'a', 'an']]
            
            company_found = False
            if significant_words:
                words_found = sum(1 for word in significant_words if re.search(r'\b' + re.escape(word) + r'\b', full_text))
                company_found = words_found >= len(significant_words)
            
            if not company_found:
                continue
            
            # Parse date
            result_date = None
            if date_str:
                try:
                    date_formats = ['%Y-%m-%d', '%B %d, %Y', '%b %d, %Y', '%Y/%m/%d', '%m/%d/%Y']
                    for fmt in date_formats:
                        try:
                            result_date = datetime.strptime(str(date_str).split()[0], fmt)
                            break
                        except ValueError:
                            continue
                    if result_date is None:
                        result_date = pd.to_datetime(date_str).to_pydatetime()
                except:
                    pass
            
            # Check date filter
            if last_checked_date and result_date:
                if result_date.date() <= last_checked_date.date():
                    continue
            
            # Prefer news articles
            is_news = any(domain in link.lower() for domain in news_domains)
            
            valid_results.append({
                'link': link,
                'title': title,
                'snippet': snippet or '',
                'date': result_date,
                'is_news': is_news,
                'full_text': full_text
            })
        
        if not valid_results:
            return {'found': False, 'summary': None, 'link': None, 'title': None, 'date': None}
        
        # Pick the best result (prefer news articles, prefer more recent)
        valid_results.sort(key=lambda x: (x['is_news'], x['date'] is not None), reverse=True)
        best_result = valid_results[0]
        
        # Use Parallel extract or get_info_via_search to summarize
        print(f"  Found update: {best_result['title']}")
        print(f"  Extracting summary from article...")
        
        summary = get_detailed_info_on_alert(
            link=best_result['link'],
            company_name=company,
            search_keywords=f"{company} startup"
        )
        
        # Format date
        if best_result['date']:
            date_str = best_result['date'].strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        return {
            'found': True,
            'summary': summary,
            'link': best_result['link'],
            'title': best_result['title'],
            'date': date_str
        }
        
    except Exception as e:
        print(f"  ⚠️  Error searching for {company_name}: {e}")
        import traceback
        traceback.print_exc()
        return {'found': False, 'summary': None, 'link': None, 'title': None, 'date': None}


def search_updates_for_tracked_companies(tracking_df, days_back=30, skip_companies=None):
    """
    Search for recent news and updates for all companies in the tracking database using Parallel API.
    Summarizes what happened, what's being said about the company (could be funding, product launches, etc.).
    Only searches for companies that haven't been checked recently or have no recent updates.
    
    Args:
        tracking_df (pandas.DataFrame): DataFrame with tracking information
        days_back (int): Only search for companies checked more than this many days ago
        skip_companies (set): Set of company names to skip (e.g., companies just updated from alerts)
        
    Returns:
        pandas.DataFrame: Updated tracking DataFrame with company updates
    """
    if skip_companies is None:
        skip_companies = set()
    print("=" * 80)
    print("Searching for company updates using Parallel API")
    print("=" * 80)
    print()
    
    # Ensure required columns exist
    if 'most_recent_update' not in tracking_df.columns:
        tracking_df['most_recent_update'] = ''
    if 'most_recent_update_link' not in tracking_df.columns:
        tracking_df['most_recent_update_link'] = ''
    if 'update_date' not in tracking_df.columns:
        tracking_df['update_date'] = ''
    if 'last_checked' not in tracking_df.columns:
        tracking_df['last_checked'] = ''
    
    # Filter companies to check
    companies_to_check = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    for idx, row in tracking_df.iterrows():
        company_name = row.get('company_name', '')
        if not company_name or pd.isna(company_name):
            continue
        
        # Skip companies that were just updated from alerts
        if company_name in skip_companies:
            continue
        
        last_checked = row.get('last_checked', '')
        if pd.isna(last_checked) or last_checked == '':
            # Never checked, include it
            companies_to_check.append((idx, company_name, None))
        else:
            try:
                # Parse last_checked date
                if isinstance(last_checked, str):
                    last_checked_date = datetime.strptime(last_checked.split()[0], '%Y-%m-%d')
                else:
                    last_checked_date = pd.to_datetime(last_checked).to_pydatetime()
                
                # Only check if last checked more than days_back ago
                if last_checked_date < cutoff_date:
                    companies_to_check.append((idx, company_name, last_checked_date))
            except:
                # If we can't parse the date, include it anyway
                companies_to_check.append((idx, company_name, None))
    
    if not companies_to_check:
        print("No companies need to be checked for updates.")
        return tracking_df
    
    print(f"Checking {len(companies_to_check)} companies for recent updates...")
    print()
    
    updates_found_count = 0
    
    for idx, company_name, last_checked_date in companies_to_check:
        print(f"Checking {company_name}...")
        
        # Search for company updates
        search_result = search_company_updates_with_parallel(company_name, last_checked_date)
        
        if search_result['found']:
            updates_found_count += 1
            print(f"  ✓ Update found!")
            print(f"    Title: {search_result['title']}")
            print(f"    Link: {search_result['link']}")
            if search_result['date']:
                print(f"    Date: {search_result['date']}")
            
            # Update tracking DataFrame
            # Check if this is a new update (different from existing link)
            existing_link = tracking_df.loc[idx, 'most_recent_update_link']
            if pd.isna(existing_link) or existing_link == '' or existing_link != search_result['link']:
                # Update with new summary, link, and date
                tracking_df.loc[idx, 'most_recent_update'] = search_result['summary']
                tracking_df.loc[idx, 'most_recent_update_link'] = search_result['link']
                tracking_df.loc[idx, 'update_date'] = search_result['date']
                tracking_df.loc[idx, 'last_checked'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"    Summary: {search_result['summary'][:100]}...")
            else:
                print(f"    (Update already in database)")
        else:
            print(f"  ✗ No recent updates found")
            # Update last_checked even if no update found
            tracking_df.loc[idx, 'last_checked'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print()
    
    print("=" * 80)
    print(f"Summary: Found {updates_found_count} updates out of {len(companies_to_check)} companies checked")
    print("=" * 80)
    print()
    
    return tracking_df


def check_and_append_new_companies():
    """
    Check for new companies in Notion that aren't in the tracking CSV yet.
    Only appends new companies, doesn't recreate the entire file.
    
    Returns:
        pandas.DataFrame: DataFrame with new companies that were added, or None if no new companies
    """
    processed_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
    
    if not os.path.exists(processed_path):
        print("Tracking database doesn't exist. Use initialize_tracking_csv() first.")
        return None
    
    # Load processed data
    processed_df = pd.read_csv(processed_path)
    
    # Check if page_id column exists, if not try to match by company_name
    if 'page_id' in processed_df.columns:
        existing_ids = set(processed_df['page_id'].dropna().astype(str))
        # Load latest Notion data
        new_df = import_tracked(id, page_text=True)  
        new_df['page_id'] = new_df['page_id'].astype(str)
        new_entries = new_df[~new_df['page_id'].isin(existing_ids)]
    else:
        # Fallback: match by company_name if page_id doesn't exist
        existing_companies = set(processed_df['company_name'].dropna().astype(str))
        new_df = import_tracked(id, page_text=True)
        new_entries = new_df[~new_df['company_name'].isin(existing_companies)]

    if new_entries.empty:
        print("No new companies found in Notion.")
        return None

    print(f"Found {len(new_entries)} new companies. Processing...")

    # Add missing columns for consistency with existing tracking structure
    required_cols = ['co-founder', 'personal_linkedin', 'last_checked', 
                     'most_recent_update', 'most_recent_update_link', 'update_date']
    for col in required_cols:
        if col not in new_entries.columns:
            new_entries[col] = ""

    # Reuse the logic from `process_tracking_list()` to enrich new_entries
    enriched = process_tracking_list(new_entries)
    
    # Ensure all columns from processed_df exist in enriched
    for col in processed_df.columns:
        if col not in enriched.columns:
            enriched[col] = ""

    # Append new entries to processed file
    updated_df = pd.concat([processed_df, enriched], ignore_index=True)
    updated_df.to_csv(processed_path, index=False)

    print(f"Appended {len(enriched)} new entries to the processed file.")
    return enriched
    
    
if __name__ == "__main__":
    # Define tracking database path (use absolute path based on workspace root)
    tracking_db_path = os.path.join(WORKSPACE_ROOT, 'data', 'tracking', 'tracking_db.csv')
    
    # Step 1: Initialize CSV only if it doesn't exist, otherwise load existing
    if not os.path.exists(tracking_db_path):
        print("📂 Initializing tracking database from Notion...")
        try:
            tracking_df = initialize_tracking_csv()
            print(f"✅ Created tracking database with {len(tracking_df)} companies")
        except Exception as e:
            print(f"❌ Error initializing tracking database: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to workspace-relative path if Notion fails
            workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            notion_export_path = os.path.join(workspace_root, 'data', 'tracking_names', 'notion_export_processed.csv')
            if os.path.exists(notion_export_path):
                print(f"   Falling back to Notion export: {notion_export_path}")
                tracking_df = pd.read_csv(notion_export_path)
            else:
                print(f"❌ Error: Could not load from Notion or find export file at {notion_export_path}")
                print(f"   Please ensure the tracking database exists or Notion connection is working.")
                exit(1)
    else:
        print(f"📂 Loading existing tracking database from {tracking_db_path}")
        tracking_df = pd.read_csv(tracking_db_path)
        print(f"   Loaded {len(tracking_df)} companies from existing database")
    
    # Step 2: Check Google Alerts for updates (only new alerts)
    relevant_alerts = check_google_alerts(days=6)
    
    if not relevant_alerts:
        print("\n✅ No new Google Alerts to process. Exiting.")
        sys.exit(0)
    
    # Step 3: Process alerts and update CSV
    tracking_df, companies_updated_from_alerts = process_alerts(tracking_df, relevant_alerts)
    
    # Step 4: (Optional) Search for company updates using Parallel API
    # Uncomment the line below to enable update search
    # print(f"\nSkipping {len(companies_updated_from_alerts)} companies that were just updated from Google Alerts...")
    # tracking_df = search_updates_for_tracked_companies(tracking_df, days_back=30, skip_companies=companies_updated_from_alerts)
    
    # Step 5: Save updated CSV (updates existing file, doesn't recreate)
    try:
        os.makedirs(os.path.dirname(tracking_db_path), exist_ok=True)
        tracking_df.to_csv(tracking_db_path, index=False)
        print(f"\n✅ Updated tracking database saved to {tracking_db_path}")
        print(f"   Total companies in database: {len(tracking_df)}")
        
        # Verify the file was saved
        if os.path.exists(tracking_db_path):
            file_size = os.path.getsize(tracking_db_path)
            print(f"   File size: {file_size:,} bytes")
        else:
            print(f"   ⚠️  Warning: File was not saved at {tracking_db_path}")
    except Exception as e:
        print(f"❌ Error saving tracking database: {e}")
        import traceback
        traceback.print_exc()
    
    
