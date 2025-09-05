import requests
import pandas as pd
import json
import os
import re
import math
from dotenv import load_dotenv
import time
from psycopg2.extras import execute_values
from services.database import get_db_connection
from services.groq_api import get_groq_response

load_dotenv()
def get_headers():
    return {
        'Authorization': f"Bearer {NOTION_KEY}",
        'Content-Type': 'application/json',
        'Notion-Version': '2022-06-28',
    }

NOTION_KEY = os.getenv('NOTION_KEY')

def normalize_string(s):
    """Normalize string by removing specific terms, converting to lowercase, and removing special characters."""
    if pd.notna(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s.lower().strip())
    return ''

def sanitize_number(value):
    """Ensure the number is JSON-compliant (no NaN, inf, -inf)."""
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value  # Valid number
    return 0  # Default to 0 if invalid

def safe_string(value):
    """Ensure the value is a valid string, replacing NaN with an empty string."""
    if isinstance(value, float) and pd.isna(value):  # Check if it's NaN
        return ""
    return str(value)  # Convert to string

def merge_pipeline_portfolio(df):
    """Merge the given pipeline dataframe with the Montage portfolio and
    persist the combined data to the `pipeline` table in Postgres.

    The function keeps the original dataframe behaviour (returning the merged
    dataframe) but no longer writes a CSV file. Instead it uses the shared
    `get_db_connection` helper to bulk-insert/upsert the rows into the
    `pipeline` table which has the following schema:

        id            SERIAL  PRIMARY KEY
        company_name  TEXT
        priority      TEXT
        founder       TEXT

    If a row with the same `company_name` already exists we update its priority
    and founder columns instead of creating a duplicate.
    """
    # 1. Build the portfolio slice we want to merge
    portfolio = pd.read_csv('data/portfolio_all.csv')
    columns = ['Company Name', 'Co-Founder/CEO']
    portfolio_simple = portfolio[columns].copy()
    portfolio_simple.loc[:, 'priority'] = 'portfolio'
    portfolio_simple = portfolio_simple.rename(columns={
        'Co-Founder/CEO': 'founder',
        'Company Name': 'company_name'
    })

    # 2. Merge with the dataframe that was passed in
    merged_df = pd.concat([df, portfolio_simple], ignore_index=True)
    # just before preparing `records`
    merged_df = (
        merged_df
        .drop_duplicates(subset=['company_name'], keep='last')   # keep the newest info
    )

    # 3. Persist to the database
    conn = get_db_connection()
    if conn is None:
        print("[merge_pipeline_portfolio] Could not obtain DB connection – returning dataframe without persisting.")
        return merged_df

    try:
        cursor = conn.cursor()
        # Prepare the records as a list of tuples (company_name, priority, founder)
        records = list(
            merged_df[['company_name', 'priority', 'founder']]
            .fillna('')  # Replace NaNs with empty strings to avoid DB errors
            .itertuples(index=False, name=None)
        )

        if not records:
            return merged_df  # Nothing to insert

        insert_query = """
            INSERT INTO pipeline (company_name, priority, founder)
            VALUES %s
            ON CONFLICT (company_name) DO UPDATE
            SET priority = EXCLUDED.priority,
                founder  = EXCLUDED.founder
        """
        execute_values(cursor, insert_query, records)
        conn.commit()
    except Exception as e:
        # Roll back so we don't leave the transaction open in error state
        if conn:
            conn.rollback()
        print(f"[merge_pipeline_portfolio] Error inserting into pipeline table: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return merged_df

def import_pipeline(ID):
    """
    ID: The id of the Notion database

    This function imports all the entries from Montage Pipeline in Notion.

    Returns: DataFrame with the current Montage Pipeline entries.
    """

    all_entries = []
    has_more = True
    next_cursor = None
    headers = get_headers()
    while has_more:
        # Set up the request payload with the cursor for pagination
        payload = {"page_size": 100}
        if next_cursor:
            payload["start_cursor"] = next_cursor

        # Make the request
        response = requests.post(
            f"https://api.notion.com/v1/databases/{ID}/query",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            data = response.json()
            all_entries.extend(data["results"])  # Add the current batch of results to all_entries

            # Check if there are more results
            has_more = data["has_more"]
            next_cursor = data["next_cursor"]  # Update the cursor for the next batch
        else:
            print(f"Failed to fetch data: {response.status_code} - {response.text}")
            break

    data = []

    for entry in all_entries:
        # Extract company name
        company_name = entry['properties']['Name']['title'][0]['plain_text'] if entry['properties']['Name'][
            'title'] else "No name"

        # Extract priority
        priority = entry['properties']['Priority']['select']['name'] if entry['properties']['Priority'][
            'select'] else "No priority"

        # Extract founder name
        founder = entry['properties']['Founder']['rich_text'][0]['plain_text'] if entry['properties']['Founder'][
            'rich_text'] else ""

        if 'Last Contact' in entry['properties']:
            date = entry['properties']['Last Contact']['date']['start'] if entry['properties']['Last Contact']['date'] else "No date"
        else:
            date = "No date"

        location = entry['properties']['Location']['select']['name'] if entry['properties']['Location'][
            'select'] else "No location"

        description = entry['properties']['Brief Description']['rich_text'][0]['plain_text'] if entry['properties']['Brief Description'][
            'rich_text'] else "No description"

        website = entry['properties']['Website']['rich_text'][0]['plain_text'] if entry['properties']['Website']['rich_text'] else "No website"

        sector = [t["name"] for t in entry["properties"]["Sector"]["multi_select"]] if entry["properties"]["Sector"]["multi_select"] else []
    
        # Append as a dictionary to the list
        data.append({
            "company_name": company_name,
            "priority": priority,
            "founder": founder,
            "date": date,
            "location": location,
            "website": website,
            "description": description,
            "sector": sector
        })

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

def import_tracked(ID, page_text=False):
    """
    ID: The id of the Notion database

    This function imports all the entries from Montage Pipeline in Notion.
    Extracts company name, priority, founder name, website, and any links found in page content.

    Returns: DataFrame with the current Montage Pipeline entries.
    """

    all_entries = []
    has_more = True
    next_cursor = None
    headers = get_headers()
    while has_more:
        # Set up the request payload with the cursor for pagination
        payload = {"page_size": 100}
        if next_cursor:
            payload["start_cursor"] = next_cursor

        # Make the request
        response = requests.post(
            f"https://api.notion.com/v1/databases/{ID}/query",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            data = response.json()
            all_entries.extend(data["results"])  # Add the current batch of results to all_entries

            # Check if there are more results
            has_more = data["has_more"]
            next_cursor = data["next_cursor"]  # Update the cursor for the next batch
        else:
            print(f"Failed to fetch data: {response.status_code} - {response.text}")
            break

    data = []

    for entry in all_entries:
        # Extract company name
        company_name = entry['properties']['Name']['title'][0]['plain_text'] if entry['properties']['Name'][
            'title'] else "No name"

        # Extract priority
        priority = entry['properties']['Priority']['select']['name'] if entry['properties']['Priority'][
            'select'] else "No priority"

        # Extract founder name
        founder = entry['properties']['Founder']['rich_text'][0]['plain_text'] if entry['properties']['Founder'][
            'rich_text'] else ""

        # Extract website
        website = entry['properties']['Website']['rich_text'][0]['plain_text'] if entry['properties']['Website']['rich_text'] else ""
        
        # Get page content to extract links
        if page_text:
            page_id = entry['id']
            print("Getting links from page")
            page_links = extract_links_from_page(page_id)
            print("Got the links: ", page_links)
        
        # Append as a dictionary to the list
        data.append({
            "company_name": company_name,
            "priority": priority,
            "founder": founder,
            "website": website,
            "page_links": page_links if page_text else None,
            "page_id": page_id if page_text else None
        })

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

def extract_links_from_page(page_id):
    """
    Extract all links from a Notion page's content.
    
    Args:
        page_id: The ID of the Notion page
        
    Returns:
        A dictionary with categorized links (company_linkedin, personal_linkedin, other)
    """
    headers = get_headers()
    
    # Get the page blocks
    response = requests.get(
        f"https://api.notion.com/v1/blocks/{page_id}/children",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Failed to fetch page content: {response.status_code} - {response.text}")
        return {"company_linkedin": "", "personal_linkedin": "", "other": []}
    
    blocks = response.json()["results"]
    
    # Initialize link categories
    links = {
        "company_linkedin": "",
        "personal_linkedin": "",
        "other": []
    }
    
    # Process each block to find links
    for block in blocks:
        block_type = block.get("type", "")
        
        # Skip unsupported block types
        if block_type not in ["paragraph", "bulleted_list_item", "numbered_list_item", "to_do", "toggle", "quote"]:
            continue
            
        # Get rich text content from the block
        rich_text = block.get(block_type, {}).get("rich_text", [])
        
        for text in rich_text:
            # Check for links in the text
            if "href" in text and text["href"]:
                url = text["href"]
                
                # Categorize the link
                if "linkedin.com/company" in url:
                    links["company_linkedin"] = url
                elif "linkedin.com/in" in url:
                    links["personal_linkedin"] = url
                else:
                    links["other"].append(url)
            
            # Also check for URLs in plain text
            if "plain_text" in text:
                plain_text = text["plain_text"]
                url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
                matches = re.findall(url_pattern, plain_text)
                
                for url in matches:
                    if "linkedin.com/company" in url and not links["company_linkedin"]:
                        links["company_linkedin"] = url
                    elif "linkedin.com/in" in url and not links["personal_linkedin"]:
                        links["personal_linkedin"] = url
                    elif url not in links["other"]:
                        links["other"].append(url)
    
    return links

def define_priority_dict():
    """Build dictionaries mapping normalized company / founder names to their
    priority level based on the `pipeline` table.

    Returns
    -------
    tuple(dict, dict)
        (priority_dict_person, priority_dict_company)
    """
    conn = get_db_connection()
    if conn is None:
        print("[define_priority_dict] Could not obtain DB connection – returning empty dictionaries.")
        return {}, {}

    try:
        # Load directly into a DataFrame for convenience
        known = pd.read_sql(
            "SELECT company_name, priority, founder FROM pipeline WHERE priority IS NOT NULL", conn
        )
    except Exception as e:
        print(f"[define_priority_dict] Error reading pipeline table: {e}")
        return {}, {}
    finally:
        if conn:
            conn.close()

    # Drop rows with missing priority just in case
    known = known.dropna(subset=['priority'])

    priority_dict_company = {}
    priority_dict_person = {}

    for _, row in known.iterrows():
        # Normalize company names and add to company dictionary
        normalized_company = normalize_string(row['company_name'])
        priority_dict_company[normalized_company] = row['priority']

        # Normalize founder names and add to person dictionary
        normalized_founder = normalize_string(row['founder'])
        priority_dict_person[normalized_founder] = row['priority']

    return priority_dict_person, priority_dict_company

def update_pipeline_data(update=True):
    pipeline_ID = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
    tracked_ID = "912974853b494f98a5652fcbff3ad795"
    passed_ID = "bc5f875961234aa6aa4b293cf1915ac2"
    if update:
        pipeline = import_pipeline(pipeline_ID)
        tracked = import_pipeline(tracked_ID)
        passed = import_pipeline(passed_ID)
        all_pipeline = pd.concat([pipeline, tracked, passed], ignore_index=True)
        df = merge_pipeline_portfolio(all_pipeline)  # It's saved to data/pipeline.csv
        time.sleep(2)
    priority_dict_person, priority_dict_company = define_priority_dict()
    return priority_dict_person, priority_dict_company  

def import_feedback():
    """
    Import the recommendations from Notion, with any feedback.
    
    This function imports feedback data from multiple Notion databases (commerce, fintech, healthcare, stealth)
    including Name, Profile URL, Corrected Score, Status from tool, and descriptive feedback.
    
    Returns:
        DataFrame: Combined feedback data from all databases
    """
    commerce_id = "15eea546f3968088a208d004de27036d"
    fintech_id = "15fea546f39680e984f8ccfc80fa88d0"
    healthcare_id = "15fea546f396804caa91e07408f266d3"
    stealth_id = "17aea546f3968064b590e5d45871a591"
    
    # List of database IDs and their categories
    databases = [
        {"id": commerce_id, "category": "commerce"},
        {"id": fintech_id, "category": "fintech"},
        {"id": healthcare_id, "category": "healthcare"},
        {"id": stealth_id, "category": "stealth"}
    ]
    
    all_feedback = []
    
    for db in databases:
        db_id = db["id"]
        category = db["category"]
        
        # Fetch data from the current database
        all_entries = []
        has_more = True
        next_cursor = None
        headers = get_headers()
        
        while has_more:
            # Set up the request payload with the cursor for pagination
            payload = {"page_size": 100}
            if next_cursor:
                payload["start_cursor"] = next_cursor
            
            # Make the request
            response = requests.post(
                f"https://api.notion.com/v1/databases/{db_id}/query",
                headers=headers,
                json=payload,
            )
            
            if response.status_code == 200:
                data = response.json()
                all_entries.extend(data["results"])  # Add the current batch of results
                
                # Check if there are more results
                has_more = data["has_more"]
                if has_more:
                    next_cursor = data["next_cursor"]  # Update the cursor for the next batch
                else:
                    next_cursor = None
            else:
                print(f"Failed to fetch data from {category} database: {response.status_code} - {response.text}")
                break
        
        # Process the entries from this database
        for entry in all_entries:
            feedback_item = {
                "category": category,
                "page_id": entry["id"]  # Store the page ID for fetching descriptive feedback
            }
            
            properties = entry.get("properties", {})
            
            # Extract Name
            if "Name" in properties and properties["Name"].get("title"):
                name_data = properties["Name"]["title"]
                feedback_item["name"] = name_data[0]["plain_text"] if name_data else ""
            else:
                feedback_item["name"] = ""
            
            # Extract Profile URL
            if "Profile URL" in properties:
                url_data = properties["Profile URL"]
                if url_data.get("url"):
                    feedback_item["profile_url"] = url_data["url"]
                elif url_data.get("rich_text") and url_data["rich_text"]:
                    feedback_item["profile_url"] = url_data["rich_text"][0]["plain_text"]
                else:
                    feedback_item["profile_url"] = ""
            else:
                feedback_item["profile_url"] = ""
            
            # Extract Corrected Score
            if "Corrected Score" in properties:
                score_data = properties["Corrected Score"]
                if score_data.get("number") is not None:
                    feedback_item["corrected_score"] = score_data["number"]
                else:
                    feedback_item["corrected_score"] = None
            else:
                feedback_item["corrected_score"] = None
            
            # Extract Status from tool
            if "Status from tool" in properties:
                status_data = properties["Status from tool"]
                if status_data.get("type") == "status" and status_data.get("status"):
                    feedback_item["status"] = status_data["status"].get("name", "")
                else:
                    feedback_item["status"] = ""
            else:
                feedback_item["status"] = ""
            
            # Fetch the page content to get the descriptive feedback
            try:
                page_response = requests.get(
                    f"https://api.notion.com/v1/blocks/{feedback_item['page_id']}/children",
                    headers=headers
                )
                
                if page_response.status_code == 200:
                    blocks = page_response.json().get("results", [])
                    descriptive_feedback = ""
                    found_header = False
                    
                    for block in blocks:
                        # Look for the header "Descriptive feedback on recommendation"
                        if not found_header and block.get("type") == "heading_2":
                            heading_text = block["heading_2"]["rich_text"][0]["plain_text"] if block["heading_2"]["rich_text"] else ""
                            if "Descriptive feedback on recommendation" in heading_text:
                                found_header = True
                                continue
                        
                        # If we found the header, collect the text until we hit another header
                        if found_header:
                            if block.get("type") == "paragraph" and block["paragraph"]["rich_text"]:
                                for text_item in block["paragraph"]["rich_text"]:
                                    descriptive_feedback += text_item["plain_text"] + " "
                            elif block.get("type") in ["heading_1", "heading_2", "heading_3"]:
                                # Stop collecting if we hit another header
                                break
                    
                    feedback_item["descriptive_feedback"] = descriptive_feedback.strip()
                else:
                    print(f"Failed to fetch page content: {page_response.status_code} - {page_response.text}")
                    feedback_item["descriptive_feedback"] = ""
            except Exception as e:
                print(f"Error fetching descriptive feedback: {e}")
                feedback_item["descriptive_feedback"] = ""
            
            all_feedback.append(feedback_item)
    
    # Convert to DataFrame
    feedback_df = pd.DataFrame(all_feedback)
    
    # Save to CSV for reference
    feedback_df.to_csv('data/feedback.csv', index=False)
    
    return feedback_df

def clean_verticals(verticals):
    # Remove the confidence level from the string of verticals. 
    # Input is formatted like: "TechBio (high), AI (medium), Consumer (low)"
    # Output should be: "TechBio, AI, Consumer"
    return verticals.replace("(high)", "").replace("(medium)", "").replace("(low)", "").strip()

def ai_draft_message(row):
    verticals = clean_verticals(row['verticals'])
    thesis = row.get('gpt_thesis_check', '')

    prompt = (
        "I am an investor at Montage Ventures. I want to reach out to this person that I came across on LinkedIn"
        " because I am interested in learning more about"
        " what they are building. "
        "Please write a short and casual message for reaching out to a founder I found on LinkedIn. "
        "The tone should be thoughtful and warm, not too salesy. "
        "Use simple language, avoid buzzwords, and keep it vague if the company is stealth. "
        "Do not use greetings like 'Hi' or 'Best'. "
        "Avoid the words: keen, eager, profound, synergies. "
        "Aim for 2–3 sentences only.\n\n"
        "Here is an example I liked:\n"
        "\"I'm an investor with Montage Ventures and came across your profile. We invest in TechBio and Healthcare. "
        "Your background is really impressive and I'd love to learn more about you and what you're building. "
        "Let me know if you have some time to chat in the next few weeks?\"\n\n"
        "Use the following data to generate the message:\n\n"
    )

    row_text = f"**Verticals (high-confidence only)**: {verticals}\n"
    row_text += f"**Company**: {row['company_name']}\n"
    row_text += f"**Description**: {row['description_1']}\n"
    
    if thesis and str(thesis).lower() != "none" and str(thesis).lower() != "negative":
        row_text += f"**Thesis Match**: {thesis}"

    full_prompt = f"{prompt}{row_text}"
    return get_groq_response(full_prompt)

def add_messages(top_df, output_file):
    for index, row in top_df.iterrows():
        #if index >= 5:
        #    break
        msg = ai_draft_message(row)
        top_df.at[index, 'Message draft'] = msg

    top_df.to_csv(output_file, index=False)

def fetch_all_pages(database_id):
    """
    Fetch all existing pages (rows) in the Notion database.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    all_pages = []
    has_more = True
    start_cursor = None
    headers = get_headers()
    while has_more:
        payload = {"page_size": 100}
        if start_cursor:
            payload["start_cursor"] = start_cursor

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            all_pages.extend(data["results"])
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor", None)
        else:
            print(f"Failed to fetch pages: {response.status_code} - {response.text}")
            break

    return all_pages

def truncate_after_industry(text):
    """
    Remove unnecessary information from the company description block in Notion
    If 'Industry:' is not present, return the original text.
    """
    marker = "Industry:"
    # Find the index of "Industry:" in the text
    industry_index = text.find(marker)

    if industry_index != -1:
        # Return the text up to "Industry:" (including it, if desired)
        return text[:industry_index].strip()
    else:
        # Return the original text if "Industry:" is not found
        return text.strip()

def combine_education_and_work_info(row):
    """
    Combine education and work experience information into two separate strings,
    skipping "Not Available" and empty values.
    """
    school_entries = []
    work_entries = []

    # Process Education Information
    for i in range(1, 4):  # Adjust the range for the number of schools
        school = row.get(f"school_name_{i}", "")
        degree = row.get(f"degree_{i}", "")
        dates = row.get(f"school_dates_{i}", "")
        details = row.get(f"details_{i}", "")

        # Filter out "Not Available" and empty values for the current school
        parts = [p for p in [school, degree, dates] if p and p != "Not available" and pd.notna(p)]
        
        # Create the main entry with school, degree, and dates
        if parts:  # Add the entry only if there are valid parts
            main_entry = ", ".join(map(str, parts))
            
            # Add details as a separate paragraph if available
            if details and details != "Not available" and pd.notna(details):
                school_entries.append(f"{main_entry}\n  {details}")
            else:
                school_entries.append(f"{main_entry}")

    # Process Work Experience Information
    for i in range(1, 6):  # Adjust the range for the number of work experiences
        company = row.get(f"company_name_{i}", "")
        position = row.get(f"position_{i}", "")
        dates = row.get(f"dates_{i}", "")

        # Filter out "Not Available" and empty values for the current work experience
        parts = [p for p in [company, position, dates] if p and p != "Not available" and pd.notna(p)]

        # Combine the remaining parts
        if parts:  # Add the entry only if there are valid parts
            work_entries.append(f"{', '.join(map(str, parts))}")

    # Combine all entries with proper line breaks
    return "\n".join(school_entries), "\n".join(work_entries)

def combine_education_and_work_info_from_jsonb(row):
    """
    Combine education and work experience information using regular education columns
    but pulling work experiences from the all_experiences JSONB field (top 5 entries).
    """
    import json
    from datetime import datetime
    import pandas as pd
    
    school_entries = []
    work_entries = []
    
    # Process Education Information (same as original function)
    for i in range(1, 4):  # Adjust the range for the number of schools
        school = row.get(f"school_name_{i}", "")
        degree = row.get(f"degree_{i}", "")
        dates = row.get(f"school_dates_{i}", "")
        details = row.get(f"details_{i}", "")

        # Filter out "Not Available" and empty values for the current school
        parts = [p for p in [school, degree, dates] if p and p != "Not available" and pd.notna(p)]
        
        # Create the main entry with school, degree, and dates
        if parts:  # Add the entry only if there are valid parts
            main_entry = ", ".join(map(str, parts))
            
            # Add details as a separate paragraph if available
            if details and details != "Not available" and pd.notna(details):
                school_entries.append(f"{main_entry}\n  {details}")
            else:
                school_entries.append(f"{main_entry}")
    
    # Process Work Experience Information from JSONB
    all_experiences = row.get('all_experiences')
    if all_experiences:
        # Parse JSON if it's a string
        if isinstance(all_experiences, str):
            try:
                all_experiences = json.loads(all_experiences)
            except (json.JSONDecodeError, TypeError):
                all_experiences = []
        
        # Ensure it's a list
        if isinstance(all_experiences, list):
            # Helper function to format dates
            def format_date(date_str):
                if not date_str:
                    return ""
                try:
                    # Parse ISO format date and return just year-month
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m')
                except:
                    return date_str
            
            # Process top 5 work experiences
            for i, experience in enumerate(all_experiences[:5]):
                if not isinstance(experience, dict):
                    continue
                    
                company_name = experience.get('company_name', '')
                position = experience.get('position', '')
                start_date = format_date(experience.get('start_date'))
                end_date = format_date(experience.get('end_date'))
                description = experience.get('description', '')
                
                # Skip if no company name or position
                if not company_name and not position:
                    continue
                    
                # Format date range
                date_range = ""
                if start_date and end_date:
                    date_range = f"({start_date} - {end_date})"
                elif start_date:
                    date_range = f"({start_date} - Present)"
                elif end_date:
                    date_range = f"(Until {end_date})"
                    
                # Create main entry parts
                parts = [p for p in [company_name, position, date_range] if p]
                
                if parts:
                    main_entry = ", ".join(parts)
                    
                    # Add description as a separate line if available and not too long
                    if description and len(description) <= 200:  # Limit description length
                        entry_text = f"{main_entry}\n  {description}"
                    else:
                        entry_text = main_entry
                        
                    work_entries.append(entry_text)
    
    # Combine all entries with proper line breaks
    return "\n".join(school_entries), "\n".join(work_entries)

def sanitize_payload(payload):
    for block in payload.get('children', []):
        for key, value in block.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        for item in sub_value:
                            if 'content' in item['text'] and isinstance(item['text']['content'], float):
                                if math.isnan(item['text']['content']):
                                    # Replace NaN with an empty string
                                    item['text']['content'] = ""
    return payload

def summarize_analysis(row):
    table_rows = [
        {"label": "Company tech score", "value": row['company_tech_score']},
        {"label": "Past success indication score", "value": row['past_success_indication_score']},
        {"label": "Startup experience score", "value": row['startup_experience_score']},
        {"label": "Industry expertise score", "value": row['industry_expertise_score']},
        {"label": "Technical founder?", "value": row['technical']},
        {"label": "Repeat founder?", "value": row['repeat_founder']},
        {"label": "Thesis match", "value": row['tree_thesis']},
        {"label": "Category", "value": row['tree_path']}
    ]

    table_blocks = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"{row['label']:<15} | {row['value']}",
                        },
                        "annotations": {"code": True}
                    }
                ]
            }
        }
        for row in table_rows
    ]
    return table_blocks

def add_blocks_to_page(page_id, row):
    """
    Add Description and Message as paragraph blocks to the created Notion page.
    """
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    title = row.get("title", "")
    product = str(row.get("product", ""))
    market = str(row.get("market", ""))
    if product != "" and market != "" and product != "nan" and market != "nan":
        description = product + "\n\n" + market
    else:
        description = str(row.get("description_1", ""))
    message = row.get("Message draft", "")
    background = combine_education_and_work_info(row)
    school_entries, work_entries = background
    school_entries = school_entries.split("\n")
    work_entries = work_entries.split("\n")
    analysis_block = summarize_analysis(row)
    school_blocks = [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"text": {"content": entry.strip()}}]}
        }
        for entry in school_entries if entry.strip()  # Skip empty lines
    ]
    work_blocks = [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"text": {"content": entry.strip()}}]}
        }
        for entry in work_entries if entry.strip()  # Skip empty lines
    ]

    payload = {
        "children": [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Company Description"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": description[:1000]}}]},
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Founder Background"}}]}
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"text": {"content": "Education"}}]}
            },
            *school_blocks,
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": [{"text": {"content": "Experience"}}]}
            },
            *work_blocks,
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Initial Analysis"}}]}
            },
            *analysis_block,
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Suggested message"}}]}
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": [{"text": {"content": message}}]},
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"text": {"content": "Descriptive feedback on recommendation"}}]}
            },
        ]
    }
    headers = get_headers()
    payload = sanitize_payload(payload)
    response = requests.patch(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Failed to add blocks: {response.status_code} - {response.text}")

def calculate_ai_score(row):
    founder_score = row.get('founder_score', 0)
    past_success_indication_score = row.get('past_success_indication_score', 0)
    startup_experience_score = row.get('startup_experience_score', 0)
    company_tech_score = row.get('company_tech_score', 0)
    fintech_score = row.get('fintech_score', 0)
    healthcare_score = row.get('healthcare_score', 0)
    commerce_score = row.get('commerce_score', 0)
    vertical_score = max(fintech_score, healthcare_score, commerce_score)
    ai_score = (founder_score + past_success_indication_score + startup_experience_score + company_tech_score + vertical_score) / 5
    return ai_score
    

def create_page_with_blocks(database_id, row):
    """
    Create a new page in the Notion database and add Description and Message as blocks.
    """
    # Step 1: Create a new page with basic metadata
    url = "https://api.notion.com/v1/pages"
    title = row.get("title", "")
    description = str(row.get("description_1", "")) if row.get("description_1", "") else ""
    description = description.split("\n")[0]
    description = truncate_after_industry(description)
    message = row.get("Message draft", "")
    if row['all_experiences'] != '' and row['all_experiences'] is not None:
        background = combine_education_and_work_info_from_jsonb(row)
    else:
        background = combine_education_and_work_info(row)
    school_entries, work_entries = background
    school_entries = school_entries.split("\n")
    work_entries = work_entries.split("\n")
    ai_score = calculate_ai_score(row)
    school_blocks = [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"text": {"content": entry.strip()}}]}
        }
        for entry in school_entries if entry.strip()  # Skip empty lines
    ]
    work_blocks = [
        {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"text": {"content": entry.strip()}}]}
        }
        for entry in work_entries if entry.strip()  # Skip empty lines
    ]

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            "Name": {"title": [{"text": {"content": safe_string(row["company_name"]) if "stealth" not in row["company_name"].lower() else "Stealth"}}]},
            "Founder": {"rich_text": [{"text": {"content": safe_string(row["name"])}},]},
            "Profile URL": {"url": row.get("profile_url", None)},
            "Sector": {
                "multi_select": [
                    {"name": re.sub(r'\s*\([^)]*\)', '', v).strip()}  # Remove text in parentheses and strip whitespace
                    for v in row.get("verticals", "").split(",") if v.strip()
                ]
            },
            "Background Tags": {
                "multi_select": [{"name": str(t.strip())} for t in str(row.get("company_tags", "")).split(",") if t.strip()]
            },
            "Building since": {"rich_text": [{"text": {"content": safe_string(row.get("building_since", ""))}}]},
            "AI Score": {"number": sanitize_number(ai_score)},
            "Location": {"rich_text": [{"text": {"content": safe_string(row.get("location", ""))}}]},
            "Funding": {"rich_text": [{"text": {"content": safe_string(row.get("funding", ""))}}]},
            "Source": {"rich_text": [{"text": {"content": safe_string(row.get("source", ""))}}]},
            "Priority": {"select": {"name": "Qualifying"}},
            "Website": {"rich_text": [{"text": {"content": safe_string(row.get("company_website", ""))}}]}
        },
    }
    headers = get_headers()
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        print(f"Successfully created page for: {row['name']}")
        page_id = response.json()["id"]

        # Step 2: Add Description and Message as content blocks
        add_blocks_to_page(page_id, row)
    else:
        print(f"Failed to create page for {row['name']}: {response.status_code} - {response.text}")


def sync_notion_database(csv_file, database_id):
    """
    Sync new entries from the CSV files with the top recommendations to the Notion database.
    Skip entries that already exist in Notion or are in the pipeline.csv file.
    """
    # Step 1: Fetch all existing pages in Notion
    print("Fetching all existing rows...")
    existing_pages = fetch_all_pages(database_id)

    # Extract unique identifiers (e.g., Name) from existing pages
    existing_entries = {}
    for page in existing_pages:
        name = page["properties"]["Name"]["title"][0]["plain_text"] if page["properties"]["Name"]["title"] else None
        existing_entries[name] = page["id"]

    # Step 2: Read the pipeline.csv file to get companies already in the pipeline
    pipeline_companies = set()
    pipeline_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pipeline.csv')
    if os.path.exists(pipeline_path):
        print("Reading pipeline data...")
        try:
            pipeline_df = pd.read_csv(pipeline_path)
            if 'company_name' in pipeline_df.columns:
                # Process company names to match the format used in Notion
                pipeline_companies = set(
                    safe_string(name) if isinstance(name, str) and "stealth" not in name.lower() else "Stealth"
                    for name in pipeline_df['company_name'] if pd.notna(name) and name
                )
                print(f"Found {len(pipeline_companies)} companies in pipeline")
        except Exception as e:
            print(f"Error reading pipeline file: {e}")

    # Step 3: Read the CSV file with recommendations
    print("Reading data from CSV...")
    df = pd.read_csv(csv_file)
    #df = df.iloc[:5]
    required_columns = ["name", "company_name", "profile_url", "verticals", "company_tags",
                        "description_1", "building_since", "founder_score", "Message draft"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print("Missing columns: ", missing_columns)
        return

    # Reverse the DataFrame to ensure new entries go to the top
    df = df.iloc[::-1].reset_index(drop=True)

    # Step 4: Compare and sync entries
    print("Syncing entries...")
    skipped_pipeline = 0
    for _, row in df.iterrows():
        # Use company name as the key identifier, matching the Notion database structure
        company_name = safe_string(row["company_name"]) if "stealth" not in row["company_name"].lower() else "Stealth"
        founder_name = row["name"]
        
        # Create a unique identifier combining company and founder
        unique_id = f"{company_name}"
        
        # Check if this entry already exists in Notion
        exists = False
        for existing_name, page_id in existing_entries.items():
            # Check if either the company name or the unique ID matches
            if existing_name and (existing_name == company_name or existing_name == unique_id):
                exists = True
                #print(f"Entry already exists in Notion: {unique_id}")
                break
        
        # Check if this entry is in the pipeline
        in_pipeline = company_name in pipeline_companies
        if in_pipeline:
            skipped_pipeline += 1
                
        # If the entry exists or is in pipeline, skip it
        if exists or in_pipeline:
            continue
        else:
            # If the entry is new, create a new page
            try:
                create_page_with_blocks(database_id, row)
                print(f"Added entry to database: {unique_id}")
            except Exception as e:
                print(f"Failed to add entry to database: {unique_id} - {e}")
    
    if skipped_pipeline > 0:
        print(f"Skipped {skipped_pipeline} entries that were already in the pipeline")


def update_notion_recs():
    commerce_id = "15eea546f3968088a208d004de27036d"
    commerce_file = "data/top_profiles/top_commerce.csv"
    fintech_id = "15fea546f39680e984f8ccfc80fa88d0"
    fintech_file = "data/top_profiles/top_fintech.csv"
    healthcare_id = "15fea546f396804caa91e07408f266d3"
    healthcare_file = "data/top_profiles/top_healthcare.csv"
    stealth_id = "17aea546f3968064b590e5d45871a591"
    stealth_file = "data/top_profiles/top_stealth.csv"
    ai_id = "242ea546f39680708d26d83fa1fe5f9a"
    ai_file = "data/top_profiles/top_ai.csv"
    
    top_df = pd.read_csv(commerce_file)
    add_messages(top_df, commerce_file)
    top_df = pd.read_csv(fintech_file)
    add_messages(top_df, fintech_file)
    top_df = pd.read_csv(healthcare_file)
    add_messages(top_df, healthcare_file)
    top_df = pd.read_csv(stealth_file)
    add_messages(top_df, stealth_file)
    top_df = pd.read_csv(ai_file)
    add_messages(top_df, ai_file)

    print("Syncing healthcare")
    sync_notion_database("data/top_profiles/top_healthcare.csv", healthcare_id)
    print("Syncing commerce")
    sync_notion_database("data/top_profiles/top_commerce.csv", commerce_id)
    print("Syncing fintech")
    sync_notion_database("data/top_profiles/top_fintech.csv", fintech_id)
    print("Syncing stealth")
    sync_notion_database("data/top_profiles/top_stealth.csv", stealth_id)
    print("Syncing AI")
    sync_notion_database("data/top_profiles/top_ai.csv", ai_id)
