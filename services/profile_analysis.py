import sys
import os
import re
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path so we can import the services module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.ai_scoring import ai_score_founder, ai_score_fintech, ai_score_healthcare, ai_score_commerce, prompt_industry_score, startup_experience_score, past_success_indication_score, company_tech_score
from services.ai_parsing import get_past_notable_company, get_past_notable_education, generate_verticals
from services.linkedin_company_profiles import get_detailed_company_description
from services.tree import tree_analysis

def check_if_founder(row):

    if row["founder"] == "Yes" or row["founder"] == "True" or row["founder"] is True:
        return True

    # Initialize variables
    matched_keyword = None  # Store the matched keyword
    company_col = "company_name_1"
    position_col = "position_1"
    # Ensure position_1 exists and is valid
    if pd.notna(row.get(position_col)):
        first_position = row[position_col].strip().lower()  # Normalize text
        # Check if any keyword is found and store it
        for keyword in ["founder", "co-founder", "ceo", "founder/ceo", "chief executive officer", "founding engineer", "founding scientist", "founding"]:
            if keyword in first_position:
                matched_keyword = keyword  # Store the first matched keyword
                break  # Stop checking after the first match

        # If a match was found, check venture exclusions
        if matched_keyword:
            company_name = str(row.get(company_col, "")).lower().strip()

            if "partner" in first_position or "ventures" in company_name:
                return False  # Not a true founder

            return True  # Is a founder

    return False  # Default to "Not a Founder"

def get_startup_name(row):
    company_index = 'company_name_1'
    company_name = row[company_index]
    company_name = fix_company_name(company_name)
    if "university " in company_name.lower():
        position = 'position_1'
        pattern = r"\b(?:Co-)?Founder[,| at]+\s*(.+)"
        match = re.search(pattern, row[position], re.IGNORECASE)
        if match:
            company_name = match.group(1).strip()
        else:
            company_name = "Stealth"
    return company_name

def fix_company_name(s):
    pattern = r'\(YC.*?\)$'
    additional_patterns = [
        r'· Full-time$',
        r'· Permanent Full-time$',
        r'· Self-employed$',
        r'· Part-time$',
        r'· Freelance$',
        r'· Permanent$'
    ]

    if pd.notna(s):
        # Remove specific patterns defined above
        for pat in additional_patterns:
            if re.search(pat, s):
                s = re.sub(pat, '', s).strip()

        # Remove "(YC...)" pattern
        if bool(re.search(pattern, s)):
            s = re.sub(pattern, '', s).strip()

    return s

def construct_row_text(row, clean_columns=False, json=False):
    """
    Constructs a structured text summary of a founder's profile for AI scoring.
    Handles variations in column naming using the `clean_columns` flag.

    Parameters:
        row (dict): Dictionary containing founder data.
        clean_columns (bool): If True, uses lowercase keys with underscores for consistency.

    Returns:
        str: Formatted text summary of the founder.
    """

    # Handle column name variations if `clean_columns` is enabled
    if clean_columns:
        mappings = {
            "building_since": "Building Since",
            "verticals": "Verticals",
            "years_of_experience": "Years of experience",
            "company_tags": "Company Tags",
            "school_tags": "school_tags",
            "technical": "technical",
            "repeat_founder": "Repeat founder?",
            "industry_expertise_score": "Industry expertise score",
            "funding": "Funding"
        }
        # Normalize column names
        row = {mappings.get(k, k): v for k, v in row.items()}

    if json:
        current_position = f"{row.get('position_1', 'N/A')} at {row.get('company_name', 'N/A')} - {row.get('description_1', 'N/A')}"
        past_experiences = ""
        all_exp = row.get('all_experiences')
        if all_exp:
            for exp in all_exp:
                past_experiences += f"{exp['position']} at {exp['company_name']} - {exp['description']}"

    # Extract relevant fields safely
    else:
        current_position = f"{row.get('position_1', 'N/A')} at {row.get('company_name_1', 'N/A')} - {row.get('description_1', 'N/A')}, {row.get('dates_1', 'N/A')}"

        past_experiences = "; ".join([
            f"{row.get(f'position_{i}', 'N/A')} at {row.get(f'company_name_{i}', 'N/A')} - {row.get(f'description_{i}', 'N/A')}, {row.get(f'dates_{i}', 'N/A')}"
            for i in range(2, 6) if row.get(f'company_name_{i}')
        ])

    education = "; ".join([
        f"{row.get(f'school_name_{i}', 'N/A')} - {row.get(f'degree_{i}', 'N/A')}"
        for i in range(1, 4) if row.get(f'school_name_{i}')
    ])

    funding_info = f"They have raised funding of: {row.get('funding', 'N/A')}" if row.get("funding") else ""

    # Construct text
    row_text = (
        f"Current Position: {current_position}\n"
        f"Past Experiences: {past_experiences}\n"
        f"Education: {education}\n"
        f"Building Since: {row.get('building_since', 'N/A')}, around {row.get('verticals', 'N/A')}\n"
        f"Years of working experience: {row.get('years_of_experience', 'N/A')}\n"
        f"{funding_info}\n"
        f"Company Tags: {row.get('company_tags', 'N/A')}\n"
        f"School Tags: {row.get('school_tags', 'N/A')}\n"
        f"Technical Founder: {'Yes' if row.get('technical', False) else 'No'}\n"
        f"Repeat Founder: {'Yes' if row.get('repeat_founder', False) else 'No'}\n"
        f"Industry Expertise Score: {row.get('industry_expertise_score', 'N/A')}\n"
    )
    
    # Add product description if available
    if row.get('product'):
        row_text += f"Product Description: {row.get('product')}\n"
        
    # Add market description if available
    if row.get('market'):
        row_text += f"Target Market: {row.get('market')}\n"
        
    # Add company tech score if available
    if row.get('company_tech_score'):
        row_text += f"Technical Uniqueness Score: {row.get('company_tech_score')}/10\n"

    return row_text

def check_if_repeat_founder(row):
    columns_to_search = ['position_2', 'position_3', 'position_4', 'position_5']

    for column in columns_to_search:
        if column in row and pd.notna(row[column]):  # Ensure the column exists and is not NaN
            if 'founder' in row[column].lower() or 'co-founder' in row[column].lower():
                return True  # Return immediately when a match is found

    return False  # Default return if no founder roles are found

def get_profile_category(fintech_score, healthcare_score, commerce_score):
    category_scores = {
        "Fintech": fintech_score,
        "Healthcare": healthcare_score,
        "Commerce": commerce_score
    }
    # Find the category with the highest score
    highest_category = max(category_scores, key=category_scores.get)
    return highest_category

def find_priority(row, priority_dict_company, priority_dict_person):

    # Exact match for person names (Name column)
    if 'name' in row and pd.notna(row['name']) and row['name'].strip():
        normalized_person_name = normalize_string(row['name'])
        if normalized_person_name in priority_dict_person:
            return priority_dict_person[normalized_person_name]

    # Exact and fuzzy matching for company names
    for col in ['company_name_1', 'company_name_2', 'company_name_3']:
        if pd.notna(row[col]) and row[col].strip():
            s = fix_company_name(row[col])
            normalized_company_name = normalize_string(s)
            # Exact match first
            if normalized_company_name in priority_dict_company:
                return priority_dict_company[normalized_company_name]

    return None

def extract_graduation_year(date):
    # Check if the input contains a dash
    if pd.isna(date) or isinstance(date, float):
        return None
    if '-' in date:
        # Match ranges with years around a dash
        match_no_month = re.search(r'(\d{4})\s*-\s*(\d{4})', date)
        month_pattern = r'([A-Za-z]{3})\s+(\d{4})\s*-\s*([A-Za-z]{3})\s+(\d{4})'
        match_with_month = re.search(month_pattern, date)
        if match_no_month:
            # Return the second year in the range
            return int(match_no_month.group(2))
        if match_with_month:
            return int(match_with_month.group(4))  # Captures the second year (e.g., "2018")
    else:
        # Match standalone year or year with a month
        match = re.search(r'(?:[A-Za-z]+\s)?(\d{4})', date)
        if match:
            return int(match.group(1))

    # Return None if no valid year is found
    return None

def get_yoe(profile_dict):
    grad_year = extract_graduation_year(profile_dict['school_dates_1'])
    try:
        current_year = datetime.now().year
        yoe = current_year - grad_year
        if yoe < 0:
            yoe = 0
    except TypeError:
        yoe = ""
    return yoe

def normalize_string(s):
    """Normalize string by removing specific terms, converting to lowercase, and removing special characters."""
    if pd.notna(s):
        return re.sub(r'[^a-zA-Z0-9]', '', s.lower().strip())
    return ''


def clean_linkedin_url(linkedin_url):
    """
    Clean LinkedIn profile URLs by removing tracking parameters.
    
    Args:
        linkedin_url (str): LinkedIn profile URL, potentially with tracking parameters
        
    Returns:
        str: Cleaned LinkedIn profile URL
    """
    if not linkedin_url or not isinstance(linkedin_url, str):
        return linkedin_url
        
    # Remove everything after a comma (Sales Navigator tracking parameters)
    if ',' in linkedin_url:
        linkedin_url = linkedin_url.split(',')[0]
        
    # Ensure URL starts with https://www.linkedin.com/in/
    if linkedin_url and not linkedin_url.startswith('https://www.linkedin.com/in/'):
        if '/in/' in linkedin_url:
            # Extract the profile ID and rebuild the URL
            match = re.search(r'/in/([^/?&#]+)', linkedin_url)
            if match:
                profile_id = match.group(1)
                linkedin_url = f'https://www.linkedin.com/in/{profile_id}'
    
    return linkedin_url

def populate_basic_info(profile_dict, education_data, experience_data, title, about, name, posts_data):
    max_education_entries = 3
    max_experience_entries = 5
    profile_dict['title'] = title
    profile_dict['about'] = about
    profile_dict['name'] = name
    profile_dict['post_data'] = posts_data

    for i in range(min(max_education_entries, len(education_data))):
        profile_dict[f'school_name_{i+1}'] = education_data[i].get('school_name', 'Not available')
        profile_dict[f'degree_{i+1}'] = education_data[i].get('degree', 'Not available')
        profile_dict[f'school_dates_{i+1}'] = education_data[i].get('dates', 'Not available')
        profile_dict[f'details_{i+1}'] = education_data[i].get('details', 'Not available')

    for i in range(min(max_experience_entries, len(experience_data))):
        profile_dict[f'company_name_{i+1}'] = experience_data[i].get('company_name', 'Not available')
        profile_dict[f'position_{i+1}'] = experience_data[i].get('position', 'Not available')
        profile_dict[f'dates_{i+1}'] = experience_data[i].get('dates', 'Not available')
        # Check for both 'description' and 'details' keys to handle both formats
        profile_dict[f'description_{i+1}'] = experience_data[i].get('description', experience_data[i].get('details', 'Not available'))
        profile_dict[f'link_{i+1}'] = experience_data[i].get('link', 'Not available')

    return profile_dict

def is_technical_founder(row):
    technical_degrees = [
        "computer science", "engineering", "electrical engineering",
        "mechanical engineering", "civil engineering", "physics",
        "mathematics", "statistics", "data science", "machine learning"
    ]
    technical_roles = [
        "software engineer", "data scientist", "machine learning engineer",
        "research scientist", "cto", "hardware engineer", "embedded engineer",
        "robotics engineer", "backend engineer", "full stack developer",
        "devops engineer", "site reliability engineer", "security engineer", "engineer"
    ]
    score = 0

    # Check all education-related fields
    for i in range(1, 4):  # Assuming up to 3 education entries
        for key in [f"school_name_{i}", f"degree_{i}", f"details_{i}"]:
            value = row.get(key, "") or ""
            if isinstance(value, str) and any(re.search(rf"\b{degree}\b", value, re.IGNORECASE) for degree in technical_degrees):
                score += 1
                break  # Found one match, move on

    for i in range(1, 6):  # Assuming up to 5 work experiences
        for key in [f"company_name_{i}", f"position_{i}", f"description_{i}"]:
            value = row.get(key, "") or ""
            if isinstance(value, str) and any(re.search(rf"\b{role}\b", value, re.IGNORECASE) for role in technical_roles):
                score += 1
                break  # Found one match, move on

    return score >= 1

def time_since_last_promotion(profile_data):
    """
    Calculate time since last promotion in months.
    
    For positions with multiple roles (containing "previously" and durations in parentheses),
    extracts the duration of the most recent role.
    
    For single positions, extracts duration from dates_1.
    
    Args:
        profile_data (dict): Dictionary containing profile information
        
    Returns:
        int: Time since last promotion in months, or 0 if unable to determine
    """
    current_company_index = 1 if 'board' not in profile_data['position_1'] else 2
    # Check if position_1 exists in profile_data
    if f'position_{current_company_index}' not in profile_data or not profile_data[f'position_{current_company_index}']:
        return 0
    
    position = profile_data[f'position_{current_company_index}']
    
    # Check if this is a multi-position entry (contains "previously")
    if " - previously " in position:
        # Extract the first position (most recent)
        first_position = position.split(" - previously ")[0].strip()
        
        # Look for duration in parentheses
        duration_match = re.search(r'\(([^)]+)\)', first_position)
        if duration_match:
            duration_str = duration_match.group(1)
            return _convert_duration_to_months(duration_str)
    
    # If not a multi-position or no duration found in parentheses,
    # try to get duration from dates_1
    if f'dates_{current_company_index}' in profile_data and profile_data[f'dates_{current_company_index}']:
        dates = profile_data[f'dates_{current_company_index}']
        # Look for duration after the middle dot
        if ' · ' in dates:
            duration_str = dates.split(' · ')[1].strip()
            return _convert_duration_to_months(duration_str)
    
    # Default return if no duration found
    return 0

def _convert_duration_to_months(duration_str):
    """
    Convert a duration string to months.
    
    Examples:
    - "1 yr 5 mos" -> 17
    - "3 mos" -> 3
    - "2 yrs" -> 24
    
    Args:
        duration_str (str): Duration string
        
    Returns:
        int: Duration in months
    """
    total_months = 0
    
    # Extract years
    years_match = re.search(r'(\d+)\s*yr', duration_str)
    if years_match:
        years = int(years_match.group(1))
        total_months += years * 12
    
    # Extract months
    months_match = re.search(r'(\d+)\s*mo', duration_str)
    if months_match:
        months = int(months_match.group(1))
        total_months += months
    
    return total_months

def founder_likelihood_score(profile_data):
    # Get tenure at current company
    current_company_index = 1 if 'board' not in profile_data['position_1'] else 2
    dates_key = f'dates_{current_company_index}'
    building_since = profile_data[dates_key][:8]
    start_date_pattern = r'([A-Z][a-z]{2}\s\d{4})'
    start_date_match = re.search(start_date_pattern, building_since)
    if not start_date_match:
        time_to_leave = False
    else:
        start_date_str = start_date_match.group(1)
        start_date = datetime.strptime(start_date_str, '%b %Y')
        current_date = datetime.now()
        
        # Calculate time at current company in months
        tenure_months = (current_date.year - start_date.year) * 12 + current_date.month - start_date.month
        
        # If they've been at the company for more than 2 years, they might be ready to leave
        time_to_leave = tenure_months >= 24
    
    # Check time since last promotion
    months_since_promotion = time_since_last_promotion(profile_data)
    ready_for_promotion = months_since_promotion >= 18  # If it's been 18+ months since promotion

    # Are they previously founders?
    previously_founder = profile_data['repeat_founder']

    # Technical/visionary role?
    position_1 = profile_data.get(f'position_{current_company_index}', '')  # Default to an empty string if None

    if isinstance(position_1, str) and any(
            title in position_1.lower() for title in [
                "cto", "chief product officer", "chief technology officer", "vp engineering",
                "director of engineering", "engineering director", "vp product",
                "vp of product", "vp of engineering", "vp of product management"
            ]
    ):
        technical_visionary = True
    else:
        technical_visionary = False

    # Check if recently active on LinkedIn
    post_data = profile_data.get('post_data', [])  # Safely get the post_data list

    if post_data and isinstance(post_data, list) and len(post_data) > 0:  # Ensure list is not empty
        last_post = post_data[0].get('date', '')  # Get date, default to empty string if missing

        if "d" in last_post or (
                "m" in last_post and last_post.split("m")[0].isdigit() and int(last_post.split("m")[0]) < 6):
            recently_active = True
        else:
            recently_active = False
    else:
        recently_active = False  # Default if no posts exist

    # Calculate likelihood score
    likelihood_score = 0
    if time_to_leave:
        likelihood_score += 1
    if previously_founder:
        likelihood_score += 3
    if technical_visionary:
        likelihood_score += 2
    if recently_active:
        likelihood_score += 1
    if ready_for_promotion:
        likelihood_score += 1

    likelihood_score = likelihood_score / 8

    return likelihood_score


def initialize_profile_dict():
    max_education_entries = 3
    max_experience_entries = 5

    properties = ['name','profile_url', 'title', 'history','verticals', 'company_tags', 'founder', 'company_name', 'access_date']

    for i in range(1, max_education_entries + 1):
        properties.append(f'school_name_{i}')
        properties.append(f'degree_{i}')
        properties.append(f'school_dates_{i}')
        properties.append(f'details_{i}')

    for i in range(1, max_experience_entries + 1):
        properties.append(f'company_name_{i}')
        properties.append(f'position_{i}')
        properties.append(f'dates_{i}')
        properties.append(f'location_{i}')
        properties.append(f'description_{i}')
        properties.append(f'link_{i}')

    properties += ['building_since', 'founder_score', 'fintech_score', 'healthcare_score', 'commerce_score']
    properties += ['years_of_experience', 'about', 'repeat_founder', 'industry_expertise_score','funding']
    properties += ['category', 'thesis', 'technical', 'school_tags', 'post_data', 'likelihood_score', 'startup_experience_score']
    properties += ['past_success_indication_score', 'product', 'business_stage', 'company_tech_score', 'company_website', 'market']

    # Create a dictionary with all properties as keys and None as values
    profile_dict = {prop: "" for prop in properties}
    
    return profile_dict

def analyze_profile(profile_dict, priority_dict_person, priority_dict_company, experience_data, context=None):
    # Initialize relevant_text to avoid "too many values to unpack" error
    relevant_text = ""
    
    if profile_dict['founder']:
        # Derive information
        company_tags = get_past_notable_company(profile_dict)
        school_tags = get_past_notable_education(profile_dict)    
        company_name = get_startup_name(profile_dict)
        technical = is_technical_founder(profile_dict)

        # Update profile_dict with derived values
        profile_dict['company_tags'] = company_tags
        profile_dict['school_tags'] = school_tags
        profile_dict['company_name'] = company_name
        profile_dict['technical'] = technical
        profile_dict['building_since'] = profile_dict['dates_1'][:8]
        profile_dict['company_name'] = get_startup_name(profile_dict)

        company_description = profile_dict['description_1']
        relevant_text = company_description + ", "
        company_description, funding, website = get_detailed_company_description(experience_data, company_name, company_description, context)
        profile_dict['funding'] = funding
        profile_dict['description_1'] = company_description
        profile_dict['company_website'] = website
        
        # Extract product information from the company website or LinkedIn description
        if website:
            website_info = extract_info_from_website(website, company_description)
            product_description = website_info.get('product_description', '')
            market_description = website_info.get('market_description', '')
            
            # Clean and format the product description
            if product_description:
                # Remove any extra whitespace and newlines
                product_description = ' '.join(product_description.split())
            
            # Clean and format the market description
            if market_description:
                # Remove any extra whitespace and newlines
                market_description = ' '.join(market_description.split())
  
            profile_dict['product'] = product_description
            profile_dict['market'] = market_description
        elif company_description:
            # If no website but we have LinkedIn description, use that
            description_info = extract_info_from_description_only(company_description)
            product_description = description_info.get('product_description', '')
            market_description = description_info.get('market_description', '')
            
            # Clean and format the product description
            if product_description:
                # Remove any extra whitespace and newlines
                product_description = ' '.join(product_description.split())
                # Limit to a reasonable length if needed
                if len(product_description) > 500:
                    product_description = product_description[:497] + '...'
            
            # Clean and format the market description
            if market_description:
                # Remove any extra whitespace and newlines
                market_description = ' '.join(market_description.split())
                # Limit to a reasonable length if needed
                if len(market_description) > 500:
                    market_description = market_description[:497] + '...'
            
            profile_dict['product'] = product_description
            profile_dict['market'] = market_description
        else:
            profile_dict['product'] = ""
            profile_dict['market'] = ""
            
        # Calculate company tech score using the AI scoring function
        tech_score = company_tech_score(profile_dict)
        profile_dict['company_tech_score'] = tech_score

        verticals = generate_verticals(profile_dict)
        profile_dict['verticals'] = verticals

        # Calculate scores
        industry_expertise_score = prompt_industry_score(profile_dict)
        founder_score = ai_score_founder(profile_dict)
        fintech_score = ai_score_fintech(profile_dict)
        healthcare_score = ai_score_healthcare(profile_dict)
        commerce_score = ai_score_commerce(profile_dict)
        startup_exp_score = startup_experience_score(profile_dict)
        past_success_score = past_success_indication_score(profile_dict)
        profile_dict['industry_expertise_score'] = industry_expertise_score
        profile_dict['founder_score'] = founder_score
        profile_dict['fintech_score'] = fintech_score
        profile_dict['healthcare_score'] = healthcare_score
        profile_dict['commerce_score'] = commerce_score
        profile_dict['startup_experience_score'] = startup_exp_score
        profile_dict['past_success_indication_score'] = past_success_score
        
        category = get_profile_category(fintech_score, healthcare_score, commerce_score)
        profile_dict['category'] = category
        
        relevant_text += profile_dict['title'] + ", " + profile_dict['company_name'] + ", " + profile_dict['description_1']
        profile_dict['likelihood_score'] = 1.0

        try:
            profile_dict = tree_analysis(profile_dict)
        except Exception as e:
            print(f"Error analyzing {profile_dict['company_name']}: {e}")

    else:
        profile_dict['founder_score'] = 0
        profile_dict['fintech_score'] = 0
        profile_dict['healthcare_score'] = 0
        profile_dict['commerce_score'] = 0
        likelihood_score = founder_likelihood_score(profile_dict)
        profile_dict['likelihood_score'] = likelihood_score
        # For non-founders, set relevant_text to basic profile info
        relevant_text = f"{profile_dict.get('name', '')}, {profile_dict.get('title', '')}, {profile_dict.get('company_name', '')}"

    # Set other details
    profile_dict['access_date'] = datetime.now().strftime('%Y-%m-%d')
    found_priority = find_priority(profile_dict, priority_dict_company, priority_dict_person)
    if found_priority:
        profile_dict['history'] = found_priority
    yoe = get_yoe(profile_dict)
    profile_dict['years_of_experience'] = yoe

    return profile_dict, relevant_text

def extract_info_from_website(website_url, company_description=None):
    """
    Extract product/service information and target market details from a company website.
    
    Args:
        website_url (str): URL of the company website
        company_description (str, optional): LinkedIn company description to supplement website data
        
    Returns:
        dict: Dictionary containing 'product_description' and 'market_description'
    """
    import requests
    from bs4 import BeautifulSoup
    import re
    from urllib.parse import urljoin, urlparse
    import time
    from services.web_search import extract_article_content
    from services.groq_api import get_groq_response
    
    if not website_url:
        # If no website but we have a company description, try to extract info from just that
        if company_description:
            return extract_info_from_description_only(company_description)
        return {"product_description": "", "market_description": ""}
    
    # Ensure URL has proper scheme
    if not website_url.startswith(('http://', 'https://')):
        website_url = 'https://' + website_url
    
    # Initialize result dictionary
    result = {
        "product_description": "",
        "market_description": ""
    }
    
    try:
        # Extract domain for internal link validation
        domain = urlparse(website_url).netloc
        
        # Make the initial request to the homepage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        
        # Get homepage content
        homepage_response = requests.get(website_url, headers=headers, timeout=10)
        homepage_response.raise_for_status()
        homepage_soup = BeautifulSoup(homepage_response.text, 'html.parser')
        
        # Extract text from homepage
        homepage_text = extract_clean_text(homepage_soup)
        
        # Find important pages to visit (about, product, solutions, etc.)
        important_links = []
        priority_pages = ['about', 'product', 'solution', 'service', 'platform', 'technology', 'how-it-works', 'what-we-do']
        
        for link in homepage_soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            if not href.startswith(('http://', 'https://')):
                href = urljoin(website_url, href)
            
            # Only include links from the same domain
            if domain in urlparse(href).netloc:
                # Check if the link contains any priority keywords
                if any(keyword in href.lower() for keyword in priority_pages):
                    important_links.append(href)
        
        # Limit to 3 important pages to avoid excessive requests
        important_links = list(set(important_links))[:3]
        
        # Collect text from important pages
        all_text = homepage_text
        
        for link in important_links:
            try:
                page_text = extract_article_content(link)
                all_text += "\n\n" + page_text
                # Add a small delay to avoid overwhelming the server
                time.sleep(1)
            except Exception as e:
                print(f"Error extracting content from {link}: {e}")
                continue
        
        # Add LinkedIn company description if available
        if company_description and company_description.strip():
            all_text += "\n\n--- LinkedIn Company Description ---\n" + company_description
        
        # Limit text length to avoid token limits with Groq
        max_text_length = 15000
        if len(all_text) > max_text_length:
            all_text = all_text[:max_text_length]
        
        # Create prompt for Groq to extract product information
        product_prompt = f"""
        Based on the following text from a company website and LinkedIn description, provide a concise description of their product or service.
        Focus on:
        1. What exactly the product or service is
        2. Key features and capabilities
        3. The problem it solves
        
        Website and LinkedIn content:
        {all_text}
        
        Provide a clear, concise description of max. 50 words of the product or service. 
        If you cannot determine the product or service, respond with "Unable to determine product/service from provided information."
        """
        
        # Create prompt for Groq to extract market information
        market_prompt = f"""
        Based on the following text from a company website and LinkedIn description, provide a concise description of their target market.
        Focus on:
        1. Who their customers are (B2B, B2C, specific industries, etc.)
        2. The specific customer segments they target
        3. Any explicit mentions of their ideal customer profile
        
        Website and LinkedIn content:
        {all_text}
        
        Provide a clear, concise description of max. 50 words of their target market.
        If you cannot determine the target market, respond with "Unable to determine target market from provided information."
        """
        
        # Get responses from Groq
        product_description = get_groq_response(product_prompt)
        market_description = get_groq_response(market_prompt)
        
        # Clean up responses
        if "Unable to determine" in product_description:
            product_description = ""
        
        if "Unable to determine" in market_description:
            market_description = ""
        
        result["product_description"] = product_description
        result["market_description"] = market_description
        
        return result
        
    except Exception as e:
        # If website extraction fails but we have a company description, try that
        if company_description:
            return extract_info_from_description_only(company_description)
        return result

def extract_info_from_description_only(company_description):
    """
    Extract product/service information and target market details from just a company description.
    Used as a fallback when no website is available or website extraction fails.
    
    Args:
        company_description (str): LinkedIn company description
        
    Returns:
        dict: Dictionary containing 'product_description' and 'market_description'
    """
    from services.groq_api import get_groq_response
    
    result = {
        "product_description": "",
        "market_description": ""
    }
    
    if not company_description or not company_description.strip():
        return result
    
    try:
        # Create prompt for Groq to extract product information
        product_prompt = f"""
        Based on the following LinkedIn company description, provide a concise description of their product or service.
        Focus on:
        1. What exactly the product or service is
        2. Key features and capabilities
        3. The problem it solves
        
        LinkedIn description:
        {company_description}
        
        Provide a clear, concise description of max. 50 words of the product or service. 
        If you cannot determine the product or service, respond with "Unable to determine product/service from provided information."
        """
        
        # Create prompt for Groq to extract market information
        market_prompt = f"""
        Based on the following LinkedIn company description, provide a concise description of their target market.
        Focus on:
        1. Who their customers are (B2B, B2C, specific industries, etc.)
        2. The specific customer segments they target
        3. Any explicit mentions of their ideal customer profile
        
        LinkedIn description:
        {company_description}
        
        Provide a clear, concise description of max. 50 words of their target market.
        If you cannot determine the target market, respond with "Unable to determine target market from provided information."
        """
        
        # Get responses from Groq
        product_description = get_groq_response(product_prompt)
        market_description = get_groq_response(market_prompt)
        
        # Clean up responses
        if "Unable to determine" in product_description:
            product_description = ""
        
        if "Unable to determine" in market_description:
            market_description = ""
        
        result["product_description"] = product_description
        result["market_description"] = market_description
        
    except Exception as e:
        print(f"Error extracting information from LinkedIn description: {e}")
    
    return result

def extract_clean_text(soup):
    """
    Extract clean text from a BeautifulSoup object.
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object
        
    Returns:
        str: Clean text
    """
    # Remove script and style elements
    for script in soup(["script", "style", "header", "footer", "nav"]):
        script.extract()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text