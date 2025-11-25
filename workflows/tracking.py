import os
import sys
import time
import json
import hashlib
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup
from firecrawl import FirecrawlApp


# Add the parent directory to the Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.notion import import_tracked

id = "912974853b494f98a5652fcbff3ad795"

# Path to store processed alerts
PROCESSED_ALERTS_FILE = '/Users/matthildur/Desktop/monty2.0/data/tracking/processed_alerts.json'

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
    add_company_alerts(search_terms)
    
    print(f"Added alerts for {len(search_terms)} companies")
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
        linkedin_profile = find_linkedin_profile(name, company_name)
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
    tracking = pd.read_csv('/Users/matthildur/monty/data/tracking_names/notion_export_processed.csv')
    
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
                # Always use a fresh browser instance (reuse_browser=False)
                profile_dict = process_profile(personal_linkedin, reuse_browser=False)
                
                # Check if we got valid profile data
                if profile_dict and not isinstance(profile_dict, str):
                    # Only process if we got valid data (not an error string)
                    look_for_updates(profile_dict, row)
                    processed_count += 1
                    
                    # Update the tracking CSV with the last checked date
                    tracking.at[index, 'last_checked'] = datetime.now().strftime('%Y-%m-%d')
                    
                    # Save progress after each successful profile to avoid losing data
                    if processed_count % 2 == 0:  # Save every 2 profiles
                        tracking.to_csv('/Users/matthildur/Desktop/monty2.0/data/tracking/tracking_db.csv', index=False)
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
        tracking.to_csv('/Users/matthildur/Desktop/monty2.0/data/tracking/tracking_db.csv', index=False)
        print(f"Successfully processed {processed_count} profiles.")
        
    except Exception as e:
        print(f"Unexpected error in profile processing batch: {str(e)}")
        # Save progress in case of unexpected errors
        tracking.to_csv('/Users/matthildur/Desktop/monty2.0/data/tracking/tracking_db.csv', index=False)
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
    """Scrape an article from a Google Alert link and analyze its content.
    
    Args:
        link (str): The Google Alert link to the article
        
    Returns:
        dict: A dictionary containing the article content, summary, and relevance analysis
    """
    from services.groq_api import get_groq_response
    import re
    import requests
    from urllib.parse import urlparse, parse_qs
    
    # Extract the actual article URL from the Google Alert link
    try:
        # Google Alert links are redirects, so we need to extract the actual URL
        parsed_url = urlparse(link)
        if 'google.com' in parsed_url.netloc:
            # Extract the actual URL from the 'url' parameter in the query string
            query_params = parse_qs(parsed_url.query)
            if 'url' in query_params:
                actual_url = query_params['url'][0]
            else:
                # If we can't extract the URL, follow the redirect
                response = requests.head(link, allow_redirects=True)
                actual_url = response.url
        else:
            actual_url = link
            
        print(f"Extracted actual URL: {actual_url}")
    except Exception as e:
        print(f"Error extracting actual URL: {e}")
        return {
            "error": f"Failed to extract actual URL: {str(e)}",
            "content": "",
            "summary": "",
            "relevance": ""
        }
    
    # Use Firecrawl to scrape the content
            
    # Fallback to requests + BeautifulSoup if Firecrawl fails
    try:
        import requests
        from bs4 import BeautifulSoup
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
        }
        
        response = requests.get(actual_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text content
        content = soup.get_text(separator='\n', strip=True)
        
        # Truncate if too long
        if len(content) > 10000:
            content = content[:10000] + "\n\n[Content truncated due to length...]\n"
            
    except Exception as fallback_error:
        print(f"Fallback scraping also failed: {fallback_error}")
        return {
            "error": f"Failed to scrape content: {str(e)}. Fallback also failed: {str(fallback_error)}",
            "content": "",
            "summary": "",
            "relevance": ""
        }
    
    # Use Groq to analyze the content
    # Prepare company information for the prompt
    target_company = company_name if company_name else (search_keywords.split(" ")[0] if search_keywords else "the company")
    
    # Create a focused prompt about what this means for the specific company
    summary_prompt = f"""
    You are analyzing a news article that mentions {target_company}. Create a concise 2-3 sentence summary that focuses ONLY on what this article means for {target_company}.
    
    Your summary should capture the following:
    - What happened or was announced
    - The purpose of the article
    - Any specific metrics, expansion, milestone, or business impact mentioned for {target_company}
    
    For example: "Galleon was featured in a Business Insider article showcasing startups in real estate, with a quote from the Founder (Amanda Orson). They mention that the platform is now available nationwide."
    
    Focus only on factual information directly stated in the article. If there's a funding announcement, include the amount and round. If there's a product launch, mention what was launched. If there's an expansion, mention the new markets.
    
    Article content:
    {content}
    """
    
    try:
        summary = get_groq_response(summary_prompt)
    except Exception as e:
        print(f"Error generating summary: {e}")
        summary = "Failed to generate summary due to an error."
    
    return summary

def process_alerts(tracking_df, relevant_alerts):
    # Ensure the most_recent_update_link column exists and is a string type
    if 'most_recent_update_link' not in tracking_df.columns:
        tracking_df['most_recent_update_link'] = ''
    if tracking_df['most_recent_update_link'].dtype != 'object':
        tracking_df['most_recent_update_link'] = tracking_df['most_recent_update_link'].astype(str)
    
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
        
        # Update the tracking DataFrame with the summary and link
        for idx in company_rows.index:
            tracking_df.loc[idx, 'most_recent_update'] = summary
            # Store the alert link to avoid re-processing in the future
            tracking_df.loc[idx, 'most_recent_update_link'] = alert_link
            # Update the last_checked timestamp
            tracking_df.loc[idx, 'last_checked'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        processed_count += 1
    
    print(f"\nSummary: Processed {processed_count} new alerts and skipped {skipped_count} already processed alerts.")
    
    # Save the updated tracking DataFrame
    tracking_df.to_csv('/Users/matthildur/Desktop/monty2.0/data/tracking/tracking_db.csv', index=False)
    return tracking_df

def check_and_append_new_companies():
    processed_path = "/Users/matthildur/Desktop/monty2.0/data/tracking/tracking_db.csv"
    
    # Load processed data
    processed_df = pd.read_csv(processed_path)
    existing_ids = set(processed_df['page_id'].dropna().astype(str))

    # Load latest Notion data
    new_df = import_tracked(id, page_text=True)  
    new_df['page_id'] = new_df['page_id'].astype(str)
    new_entries = new_df[~new_df['page_id'].isin(existing_ids)]

    if new_entries.empty:
        print("No new companies found in Notion.")
        return

    print(f"Found {len(new_entries)} new companies. Processing...")

    # Add missing columns for consistency
    for col in ['co-founder', 'personal_linkedin', 'last_checked', 
                'most_recent_update', 'most_recent_update_link', 'most_recent_update_date']:
        new_entries[col] = ""

    # Reuse the logic from `process_tracking_list()` to enrich new_entries
    enriched = process_tracking_list(new_entries)

    # Append new entries to processed file
    updated_df = pd.concat([processed_df, enriched], ignore_index=True)
    updated_df.to_csv(processed_path, index=False)

    print(f"Appended {len(enriched)} new entries to the processed file.")
    return enriched
    
    
if __name__ == "__main__":

    # Add new entries to Tracking to the csv file 
    # new_entries = check_and_append_new_companies()

    # Check latest updates from Google Alerts
    relevant_alerts = check_google_alerts(days=6)
    tracking_df = pd.read_csv('/Users/matthildur/monty/data/tracking_names/notion_export_processed.csv')
    tracking_df = process_alerts(tracking_df, relevant_alerts)
    
    
    
