import requests
import os
from dotenv import load_dotenv
from services.database import clean_linkedin_url, insert_search_results, combine_previous_scrapes, cleanup_search_list
from services.notion import update_pipeline_data
from datetime import datetime
import pandas as pd
import logging
from typing import Optional
import argparse
import time
import random
from urllib.parse import unquote

load_dotenv()

aviato_api = os.getenv("AVIATO_KEY")

# Logger setup
logger = logging.getLogger("aviato_processing")

def setup_logging(level: Optional[str] = None):
    """
    Configure module logging. By default uses LOG_LEVEL env or INFO.
    Logs to stdout with a concise format suitable for Railway.
    """
    if logger.handlers:
        # Already configured - prevent duplicate handlers
        return
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")
    try:
        log_level = getattr(logging, log_level_str.upper())
    except Exception:
        log_level = logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    # Reduce noise from third-party HTTP libs unless explicitly debugging
    if logger.level > logging.DEBUG:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

# ---- Aviato rate limiting and retry/backoff utilities ----

# Configurable via env vars
AVIATO_RPM = int(os.getenv("AVIATO_RPM", "20"))  # default 20 requests/minute (reduced from 30)
AVIATO_MAX_RETRIES = int(os.getenv("AVIATO_MAX_RETRIES", "5"))
AVIATO_BASE_BACKOFF = float(os.getenv("AVIATO_BASE_BACKOFF", "1.0"))  # seconds (increased from 0.8)
AVIATO_FOUNDER_RPM = int(os.getenv("AVIATO_FOUNDER_RPM", "6"))  # separate limiter for founders endpoint

_last_request_ts = 0.0
_min_interval = 60.0 / max(AVIATO_RPM, 1)

# Dedicated limiter for the founder lookup endpoint
_founder_last_request_ts = 0.0
_founder_min_interval = 60.0 / max(AVIATO_FOUNDER_RPM, 1)

def _wait_for_rate_limit():
    """Sleep if needed to maintain the configured RPM."""
    global _last_request_ts
    now = time.monotonic()
    elapsed = now - _last_request_ts
    if elapsed < _min_interval:
        time.sleep(_min_interval - elapsed)
    _last_request_ts = time.monotonic()

def _wait_for_founder_rate_limit():
    """Sleep if needed to respect the founders endpoint RPM."""
    global _founder_last_request_ts
    now = time.monotonic()
    elapsed = now - _founder_last_request_ts
    if elapsed < _founder_min_interval:
        time.sleep(_founder_min_interval - elapsed)
    _founder_last_request_ts = time.monotonic()

def request_with_backoff(method: str, url: str, *, headers=None, json=None, params=None, max_retries: Optional[int] = None):
    """
    Perform an HTTP request with rate limiting and exponential backoff.
    Respects Retry-After and backs off on 429 and 5xx.
    Returns a requests.Response or None if all retries failed.
    """
    retries = 0
    max_retries = AVIATO_MAX_RETRIES if max_retries is None else max_retries

    while True:
        try:
            _wait_for_rate_limit()
            resp = requests.request(method, url, headers=headers, json=json, params=params)

            # Success path
            if resp.status_code < 400:
                return resp

            # If rate limited or server error, compute backoff
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                retries += 1
                if retries > max_retries:
                    logger.error("Max retries exceeded for %s %s | Status %s | Snippet: %s", method, url, resp.status_code, resp.text[:200])
                    return resp

                # Honor Retry-After if present
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        delay = AVIATO_BASE_BACKOFF * (2 ** (retries - 1))
                else:
                    delay = AVIATO_BASE_BACKOFF * (2 ** (retries - 1))

                # Add jitter [0, 0.5s]
                delay += random.random() * 0.5
                logger.warning("Backing off %.2fs after status %s for %s %s (attempt %d/%d)", delay, resp.status_code, method, url, retries, max_retries)
                time.sleep(delay)
                continue

            # Other client errors: return response to let caller handle
            return resp

        except requests.RequestException as e:
            retries += 1
            if retries > max_retries:
                logger.error("HTTP error after retries for %s %s: %s", method, url, e)
                return None
            delay = AVIATO_BASE_BACKOFF * (2 ** (retries - 1)) + random.random() * 0.5
            logger.warning("Exception on %s %s: %s. Retrying in %.2fs (attempt %d/%d)", method, url, str(e), delay, retries, max_retries)
            time.sleep(delay)


relevant_keys = ["id", "fullName", "location", "URLs", "headline", "linkedinID", "linkedinNumID", "linkedinEntityID", 
"linkedinConnections", "linkedinFollowers", "experienceList", "educationList", "degreeList", "locationDetails"]


def get_linkedin_id(url: str) -> str:
    """
    Extract the LinkedIn ID (last path segment) from a LinkedIn profile URL.
    Decodes URL-encoded characters to handle special characters properly.
    """
    linkedin_id = url.rstrip("/").split("/")[-1]
    # Decode URL-encoded characters (e.g., %c2%ae -> ®, %c3%a3 -> ã)
    return unquote(linkedin_id)

def prepare_bulk_payload(urls: list) -> dict:
    """
    Build the JSON payload for Aviato bulk enrich from a list of LinkedIn URLs.
    """
    lookups = [{"linkedinID": get_linkedin_id(url)} for url in urls]
    return {"lookups": lookups}

def sort_search_list(search_df):
    # Sort profiles to prioritize founder-related titles
    if 'title' in search_df.columns:
        # Create a priority score column based on founder-related keywords in the title
        founder_keywords = ['founder', 'building', 'ceo', 'cto', 'chief executive officer', 'chief technical officer', 'co-founder']
        
        # Initialize priority score column with zeros
        search_df['priority_score'] = 0
        
        # Increase score for each keyword found in the title
        for keyword in founder_keywords:
            # Handle NaN values with fillna
            search_df['priority_score'] += search_df['title'].fillna('').str.lower().str.contains(keyword, case=False, na=False).astype(int)
        
        # Sort by priority score (descending)
        search_df = search_df.sort_values(by='priority_score', ascending=False)
        
        # Drop the temporary priority_score column
        search_df = search_df.drop(columns=['priority_score'])
        
        # Profiles sorted by founder-related keywords in title
    return search_df

def bulk_enrich_profiles(urls: list) -> dict:
    """
    Call Aviato's /person/bulk/enrich endpoint with a list of LinkedIn URLs.
    """
    payload = prepare_bulk_payload(urls)
    response = request_with_backoff(
        "POST",
        "https://data.api.aviato.co/person/bulk/enrich",
        headers={"Authorization": "Bearer " + aviato_api},
        json=payload,
    )
    if response is None:
        logger.error("Bulk enrich request failed with no response")
        return {}
    try:
        response.raise_for_status()
    except Exception as e:
        logger.error("Bulk enrich HTTP error: %s | Snippet: %s", e, response.text[:200] if hasattr(response, 'text') else '')
        return {}
    
    # Check if response has content
    if not response.text.strip():
        logger.warning("Empty response from bulk enrich API")
        return {}
    
    # Try to parse JSON
    try:
        return response.json()
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        logger.error("Bulk enrich JSON decode error: %s | Snippet: %s", e, response.text[:200])
        return {}

def enrich_profile(linkedin_id, linkedin_url=None):
    response = request_with_backoff(
        "GET",
        "https://data.api.aviato.co/person/enrich?linkedinID=" + linkedin_id,
        headers={
            "Authorization": "Bearer " + aviato_api
        },
    )
    
    # Check if response is successful
    if response is None or response.status_code != 200:
        # Only log 404 as warning, not error (profile doesn't exist in Aviato)
        if response and response.status_code == 404:
            logger.warning("Profile not found in Aviato for LinkedIn ID %s (404)", linkedin_id)
        else:
            logger.error("API error for LinkedIn ID %s: Status %s | Snippet: %s", linkedin_id, response.status_code if response else "None", response.text[:200] if response else "No response")
        
        # Try with URL if available and first attempt failed
        if linkedin_url:
            response = request_with_backoff(
                "GET",
                "https://data.api.aviato.co/person/enrich?linkedinURL=" + linkedin_url,
                headers={
                    "Authorization": "Bearer " + aviato_api
                },
            )
            if response is None or response.status_code != 200:
                if response and response.status_code == 404:
                    logger.warning("Profile not found in Aviato for LinkedIn URL %s (404)", linkedin_url)
                else:
                    logger.error("API error for LinkedIn URL %s: Status %s | Snippet: %s", linkedin_url, response.status_code if response else "None", response.text[:200] if response else "No response")
                return None
        else:
            return None

    # Check if response has content
    if response is None or not response.text.strip():
        logger.warning("Empty response for LinkedIn ID %s", linkedin_id)
        return None
    
    # Try to parse JSON
    try:
        return response.json()
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        logger.error("JSON decode error for LinkedIn ID %s: %s | Snippet: %s", linkedin_id, e, response.text[:200])
        return None

def sort_search_list(search_df):
    # Sort profiles to prioritize founder-related titles
    if 'title' in search_df.columns:
        # Create a priority score column based on founder-related keywords in the title
        founder_keywords = ['founder', 'building', 'ceo', 'cto', 'chief executive officer', 'chief technical officer', 'co-founder']
        
        # Initialize priority score column with zeros
        search_df['priority_score'] = 0
        
        # Increase score for each keyword found in the title
        for keyword in founder_keywords:
            # Handle NaN values with fillna
            search_df['priority_score'] += search_df['title'].fillna('').str.lower().str.contains(keyword, case=False, na=False).astype(int)
        
        # Sort by priority score (descending)
        search_df = search_df.sort_values(by='priority_score', ascending=False)
        
        # Drop the temporary priority_score column
        search_df = search_df.drop(columns=['priority_score'])
        
        # Profiles sorted by founder-related keywords in title
    return search_df

def get_search_urls():
    from services.database import fetch_profiles_from_db

    search_df = fetch_profiles_from_db("search_list")
    search_df = sort_search_list(search_df)
    return search_df

def map_aviato_to_schema(person: dict) -> dict:
    """
    Flatten Aviato enrichment JSON into schema-compatible dict.
    Prioritizes richer details for education (degreeList) and experience (titles + descriptions).
    """
    mapped = {}

    # Direct mappings
    mapped["name"] = person.get("fullName")
    mapped["linkedin_url"] = person.get("URLs", {}).get("linkedin")
    mapped["title"] = person.get("headline")

    # Location → combine all available parts into one string
    location_details = person.get("locationDetails", {})
    location_parts = []
    for level in ["locality", "region", "country"]:
        if level in location_details and location_details[level].get("name"):
            location_parts.append(location_details[level]["name"])
    mapped["location"] = ", ".join(location_parts) if location_parts else person.get("location")

    # Twitter handle if present
    mapped["twitter"] = person.get("URLs", {}).get("twitter")

    # Aviato ID
    mapped["aviato_id"] = person.get("id")
    
    # LinkedIn metadata
    mapped["linkedinID"] = person.get("linkedinID")
    mapped["linkedinNumID"] = str(person.get("linkedinNumID", ""))
    mapped["linkedinEntityID"] = person.get("linkedinEntityID")
    mapped["linkedinConnections"] = str(person.get("linkedinConnections", ""))
    mapped["linkedinFollowers"] = str(person.get("linkedinFollowers", ""))

    # Education (up to 3) → combine educationList + degreeList
    degree_lookup = {str(d.get("personEducationID")): d for d in person.get("degreeList", [])}

    for i, edu in enumerate(person.get("educationList", [])[:3], start=1):
        mapped[f"school_name_{i}"] = edu.get("school", {}).get("fullName")

        # Prefer degreeList.name if available, else subject
        deg = degree_lookup.get(str(edu.get("id")))
        mapped[f"degree_{i}"] = deg.get("name") if deg else edu.get("subject")

        mapped[f"school_dates_{i}"] = f"{edu.get('startDate', '')} - {edu.get('endDate', '')}"

        # Details → subject + degree name for more context
        details_parts = []
        if edu.get("subject"):
            details_parts.append(edu["subject"])
        if deg and deg.get("fieldOfStudy"):
            details_parts.append(deg["fieldOfStudy"])
        if deg and deg.get("name"):
            details_parts.append(deg["name"])
        mapped[f"details_{i}"] = " | ".join(details_parts) or None

    # Experience → store all roles as JSON
    experiences = []
    experience_list = person.get("experienceList", [])
    
    for i, exp in enumerate(experience_list):
        company = exp.get("company", {}) or {}
        position_list = exp.get("positionList", [])
        
        for pos in position_list:
            # Try multiple sources for company name
            company_name = (
                company.get("name") or 
                company.get("fullName") or 
                exp.get("companyName") or 
                "Unknown Company"
            )
            
            experiences.append({
                "company_id": company.get("id"),
                "company_name": company_name,
                "industry_tags": company.get("industryList", []),
                "position": pos.get("title"),
                "start_date": pos.get("startDate"),
                "end_date": pos.get("endDate"),
                "location": {
                    "locality": company.get("locality"),
                    "region": company.get("region"),
                    "country": company.get("country")
                },
                "description": pos.get("description"),
                "link": company.get("URLs", {}).get("linkedin"),
                "website": company.get("URLs", {}).get("website"),
                "twitter": company.get("URLs", {}).get("twitter")
            })

    mapped["all_experiences"] = experiences    
    founder, company_index = check_if_founder(mapped)
    mapped["founder"] = founder
    mapped["company_index"] = company_index

    return mapped


def enrich_company(company_id, profile_dict):
    # Enrich (GET /company/enrich)
    response = request_with_backoff(
        "GET",
        "https://data.api.aviato.co/company/enrich?id=" + company_id,
        headers={
            "Authorization": "Bearer " + aviato_api
        },
    )
    
    # Check if response is successful
    if response is None or response.status_code != 200:
        logger.error("Company API error for ID %s: Status %s", company_id, response.status_code)
        return profile_dict  # Return unchanged profile
    
    # Check if response has content
    if response is None or not response.text.strip():
        logger.warning("Empty company response for ID %s", company_id)
        return profile_dict  # Return unchanged profile
    
    # Try to parse JSON
    try:
        company = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        logger.error("Company JSON decode error for ID %s: %s", company_id, e)
        return profile_dict  # Return unchanged profile

    profile_dict["company_name"] = company.get("name")
    
    # Safely concatenate tagline and description
    tagline = company.get("tagline") or ""
    description = company.get("description") or ""
    if tagline and description:
        profile_dict["description_1"] = f"{tagline} {description}"
    elif tagline:
        profile_dict["description_1"] = tagline
    elif description:
        profile_dict["description_1"] = description
    else:
        profile_dict["description_1"] = None
        
    profile_dict["fundingamount"] = company.get("totalFunding")
    profile_dict["fundingroundcount"] = company.get("fundingRoundCount")
    profile_dict["latestdealtype"] = company.get("latestDealType")
    profile_dict["latestdealamount"] = company.get("latestDealAmount")
    profile_dict["lastroundvaluation"] = company.get("lastRoundValuation")

    profile_dict["embeddednews"] = company.get("embeddedNews", [])
    profile_dict["governmentawards"] = company.get("governmentAwards", [])
    profile_dict["patentcount"] = company.get("patentCount", 0)  # Should be a number, not array

    profile_dict["building_since"] = company.get("founded")
    profile_dict["headcount"] = company.get("headcount")
    profile_dict["aviato_id"] = company.get("id")

    profile_dict["company_url"] = company.get("URLs", {}).get("linkedin")
    profile_dict["company_website"] = company.get("URLs", {}).get("website")
    
    return profile_dict

def enrich_company_raw(company_website):
    """
    Enrich company data using website URL via Aviato API.
    Returns the raw company data from the API.
    """
    response = request_with_backoff(
        "GET",
        "https://data.api.aviato.co/company/enrich?website=" + company_website,
        headers={
            "Authorization": "Bearer " + aviato_api
        },
    )
    
    # Check if response is successful
    if response is None or response.status_code != 200:
        logger.error("Company API error for website %s: Status %s | Snippet: %s", 
                    company_website, response.status_code, response.text[:200])
        return None
    
    # Check if response has content
    if response is None or not response.text.strip():
        logger.warning("Empty company response for website %s", company_website)
        return None
    
    # Try to parse JSON
    try:
        company = response.json()
        return company
    except (ValueError, requests.exceptions.JSONDecodeError) as e:
        logger.error("Company JSON decode error for website %s: %s | Snippet: %s", 
                    company_website, e, response.text[:200])
        return None


import json

from datetime import datetime

def check_if_founder(row, check_json=True):
    """
    Determine if someone is a current founder and return the index of the
    most relevant experience.
    
    Prioritizes by:
      1. Strongest founder title match
      2. Most recent start_date among equals

    Returns:
        tuple: (is_founder: bool, index: int)
               index = 0 if no founder role found
    """

    experiences = []
    if check_json:
        all_exp = row.get("all_experiences")
        if isinstance(all_exp, str):
            try:
                all_exp = json.loads(all_exp)
            except Exception:
                all_exp = []
        if isinstance(all_exp, list):
            experiences = all_exp
    else:
        experiences = [{
            "position": row.get("position_1") or "",
            "company_name": row.get("company_name_1") or "",
            "start_date": row.get("dates_1") or None,
            "end_date": None
        }]

    if not experiences:
        return False, 0

    # Assign weights to founder keywords
    founder_keywords = {
        "ceo": 5,
        "co-founder": 10,
        "founder & ceo": 10,
        "ceo and co-founder": 10,
        "CEO, co-founder": 10,
        "cto, co-founder": 10,
        "founder": 5,
        "co-founder & ceo": 10,
        "co-founder & cto": 5,
        "co-founder & coo": 5,
        "co-founder & cio": 5,
        "co-founder & cfo": 5,
        "co-founder & cmo": 5,
        "co-founder & cso": 5,
        "co-founder & cpo": 5,
        "co-founder/ceo": 5,
        "co-founder/cto": 5,
        "co-founder/coo": 5,
        "co-founder/cio": 5,
        "co-founder/cfo": 5,
        "co-founder/cmo": 5,
        "co-founder/cso": 5,
        "co-founder/cpo": 5,
        "chief executive officer": 4,
        "founder & chief executive officer": 5,
        "founder & chief technical officer": 5,
        "founding engineer": 3,
        "founding scientist": 3,
        "founder fellow": 2,   # weaker
        "entrepreneur in residence": 1
    }

    founder_exps = []
    for idx, exp in enumerate(experiences):
        if exp.get("end_date"):  # must be current
            continue

        pos = (exp.get("position") or "").lower().strip()
        company = (exp.get("company_name") or "").lower().strip()

        matched_weight = max(
            (weight for kw, weight in founder_keywords.items() if kw == pos),
            default=None
        )
        if not matched_weight:
            if any(keyword in pos for keyword in founder_keywords.keys()):
                matched_weight = 1
            else:
                matched_weight = 0

        # Exclusion filters
        if "partner" in pos or "investor" in pos:
            continue
        if any(ex in company for ex in ["ventures", "capital", "partners", "association", "community"]):
            continue

        # parse start_date for comparison
        start_date = exp.get("start_date")
        parsed_start = None
        if start_date:
            try:
                # Handle YYYY, YYYY-MM, or YYYY-MM-DD
                if len(start_date) == 4:
                    parsed_start = datetime.strptime(start_date, "%Y")
                elif len(start_date) == 7:
                    parsed_start = datetime.strptime(start_date, "%Y-%m")
                else:
                    parsed_start = datetime.strptime(start_date[:10], "%Y-%m-%d")
            except Exception:
                parsed_start = None

        founder_exps.append((idx, matched_weight, parsed_start))

    if not founder_exps:
        return False, 0

    # Sort: first by weight (desc), then by start_date (desc)
    founder_exps.sort(key=lambda x: (x[1], x[2] or datetime.min), reverse=True)
    best_idx, _, _ = founder_exps[0]
    return True, best_idx


def monty_enrich_profile(profile):

    if profile["founder"]:
        idx = profile["company_index"]
        company_info = profile.get("all_experiences", [{}])[idx]
        
        # Always set company_name from the person's experience first
        profile["company_name"] = company_info.get("company_name")
        profile["description_1"] = company_info.get("description")
        
        # If we have a company_id, enrich with additional company data
        # but preserve the company_name from the person's experience
        company_id = company_info.get("company_id")
        if company_id:
            original_company_name = profile["company_name"]
            profile = enrich_company(company_id, profile)
            # Restore the original company name from person's experience
            profile["company_name"] = original_company_name

    return profile


def determine_stealth_status(profile_dict):
    """
    Determine if a profile should be saved to stealth_founders table.
    Uses the same logic as profile_processing.py.
    
    Args:
        profile_dict (dict): Profile data dictionary
        
    Returns:
        bool: True if profile should go to stealth_founders table
    """
    
    # Method 2: Content-based detection (company name contains "stealth")
    company_name = profile_dict.get('company_name', '')
    if company_name and "stealth" in company_name.lower():
        return True
    
    return False

def safe_insert_profile_to_db(profile_data, stealth=False):
    """
    Safely insert a profile to database with ON CONFLICT handling to prevent duplicates.
    
    Args:
        profile_data (dict): Profile data dictionary
        stealth (bool): Whether to insert into stealth_founders table
        
    Returns:
        bool: True if successful, False if failed or duplicate
    """
    from services.database import get_db_connection, clean_column_name

    def normalize_value(val):
        """Ensure safe DB values (None for empty, JSON for dicts/lists)."""
        if val is None:
            return None
        if isinstance(val, (list, dict)):
            return json.dumps(val)  # serialize complex structures
        if isinstance(val, str):
            return val.strip() or None
        
        # Handle boolean values properly
        if isinstance(val, bool):
            return val
        
        # Handle large integers that might exceed PostgreSQL integer limits
        if isinstance(val, (int, float)):
            try:
                # PostgreSQL integer range is -2147483648 to 2147483647
                int_val = int(val)
                if int_val > 2147483647 or int_val < -2147483648:
                    # Convert to string for very large numbers
                    return str(int_val)
                return int_val
            except (ValueError, OverflowError):
                # If conversion fails, return as string
                return str(val)
        
        # Handle string representations of large numbers
        if isinstance(val, str):
            # Check for empty JSON arrays/objects that shouldn't be in numeric fields
            if val in ['[]', '{}', 'null', 'None']:
                return None
            
            if val.isdigit():
                try:
                    int_val = int(val)
                    if int_val > 2147483647 or int_val < -2147483648:
                        return val  # Keep as string
                    return int_val
                except (ValueError, OverflowError):
                    return val
        
        # Handle numpy/pandas objects safely
        try:
            import pandas as pd
            if pd.isna(val):
                return None
        except ImportError:
            pass
        return val
    
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cur = conn.cursor()
        
        # Clean column names and values - ONLY include fields with actual values
        cleaned_columns = {}
        
        for col, val in profile_data.items():
            # Skip id field
            if col.lower() == 'id':
                continue
            
            # Skip empty values entirely - handle pandas/numpy objects safely
            cleaned_val = normalize_value(val)
                        
            # Skip empties explicitly
            if cleaned_val is None or (isinstance(cleaned_val, str) and cleaned_val.strip() == ""):
                continue

            # Clean the column name and value
            cleaned_key = clean_column_name(col)

            if isinstance(cleaned_val, str):
                if cleaned_val.strip():
                    cleaned_columns[cleaned_key] = cleaned_val
            else: 
                cleaned_columns[cleaned_key] = cleaned_val

        
        if not cleaned_columns:
            logger.warning("No valid columns to insert for profile: %s", profile_data.get('name', 'Unknown'))
            return False
            
        # Extract column names and values
        columns = ', '.join(f'"{col}"' for col in cleaned_columns.keys())
        placeholders = ', '.join(['%s'] * len(cleaned_columns))
        values = list(cleaned_columns.values())
        
        # Choose table and create insert query
        table_name = "stealth_founders" if stealth else "founders"
        
        # First check if profile already exists
        if 'profile_url' in cleaned_columns:
            check_query = f"SELECT COUNT(*) FROM {table_name} WHERE profile_url = %s"
            cur.execute(check_query, [cleaned_columns['profile_url']])
            exists = cur.fetchone()[0] > 0
            
            if exists:
                logger.info("Duplicate profile skipped: %s", profile_data.get('name', 'Unknown'))
                return False
        
        # Simple insert without ON CONFLICT
        insert_query = f"""
            INSERT INTO {table_name} ({columns}) 
            VALUES ({placeholders})
        """
        
        cur.execute(insert_query, values)
        conn.commit()
        # If we get here, the insert was successful
        return True
            
    except Exception as e:
        logger.error("Error inserting profile: %s", e, exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if 'cur' in locals() and cur:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

def process_profiles_aviato(max_profiles=10):
    cleanup_search_list()

    already_scraped_urls = combine_previous_scrapes()
    logger.info("Already scraped URLs: %d", len(already_scraped_urls))

    search_df = get_search_urls()
    logger.info("Search list size: %d", len(search_df))
    
    # Count how many profiles are actually new (not already scraped)
    new_profiles = 0
    for index, row in search_df.iterrows():
        url = row.get('profile_url')
        if url:
            url_cleaned = clean_linkedin_url(url)
            if url_cleaned not in already_scraped_urls:
                new_profiles += 1
    
    logger.info("New profiles to process: %d", new_profiles)

    count = 0
    for index, row in search_df.iterrows():
        url = row.get('profile_url')
        
        # Skip if URL is None or empty
        if not url:
            continue
            
        # Clean URL properly using the existing function
        url_cleaned = clean_linkedin_url(url)
        
        # Skip if already processed
        if url_cleaned in already_scraped_urls:
            continue
        linkedin_id = get_linkedin_id(url)
        result = enrich_profile(linkedin_id, url_cleaned)
        
        # Skip if API call failed
        if result is None:
            logger.debug("Skipping profile due to API error: %s", url)
            # Add to already_scraped_urls to prevent retrying 404s
            already_scraped_urls.append(url)
            already_scraped_urls.append(url_cleaned)
            continue
        
        mapped = map_aviato_to_schema(result)
        mapped['profile_url'] = url
        mapped = monty_enrich_profile(mapped)

        mapped["source"] = row.get("source")
        mapped["access_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        success = safe_insert_profile_to_db(mapped)
        if success:
            count += 1
            already_scraped_urls.append(url)
            already_scraped_urls.append(url_cleaned)
        if count >= max_profiles:
            break
    
    logger.info("Successfully processed %d profiles", count)
    return count

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

def check_if_repeat_founder(row):
    columns_to_search = ['position_2', 'position_3', 'position_4', 'position_5']

    for column in columns_to_search:
        if column in row and pd.notna(row[column]):  # Ensure the column exists and is not NaN
            if 'founder' in row[column].lower() or 'co-founder' in row[column].lower():
                return True  # Return immediately when a match is found

    return False  # Default return if no founder roles are found

def add_monty_data():
    """
    Add Monty analysis data to profiles from founders table where founder = true and product/market is null.
    This function enriches existing profiles with additional analysis data.
    """
    from services.database import get_db_connection, update_profile_in_db
    from services.profile_analysis import extract_info_from_website, extract_info_from_description_only, is_technical_founder
    from services.ai_parsing import get_past_notable_company, get_past_notable_education, generate_verticals
    from datetime import datetime
    import re
    import json
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        
        # Fetch profiles that need Monty data
        query = """
        SELECT * FROM founders 
        WHERE founder = true 
        AND all_experiences IS NOT NULL AND verticals IS NULL 
        ORDER BY id DESC
        """
        
        cursor.execute(query)
        profiles = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        if not profiles:
            logger.info("No profiles found that need Monty data")
            return
        
        logger.info("Profiles pending Monty data: %d", len(profiles))
        
        # Load priority dictionaries (you may need to adjust these paths)
        priority_dict_person, priority_dict_company = update_pipeline_data(update=True)
        
        processed_count = 0
        
        for profile_row in profiles:
            try:
                # Convert to dictionary
                profile_dict = dict(zip(column_names, profile_row))
                
                # 1. Add product and market analysis
                try:
                    company_description = profile_dict.get('description_1', '') or ''
                    company_website = profile_dict.get('company_website', '') or ''

                except Exception as e:
                    logger.debug("Error getting company info: %s", e, exc_info=True)
                    company_description = ''
                    company_website = ''
                
                try:
                    if company_website and company_website != 'Not available':
                        # Extract from website
                        website_info = extract_info_from_website(company_website, company_description)
                        product_description = website_info.get('product_description', '')
                        market_description = website_info.get('market_description', '')
                    elif company_description and company_description != 'Not available':
                        # Extract from description only
                        description_info = extract_info_from_description_only(company_description)
                        product_description = description_info.get('product_description', '')
                        market_description = description_info.get('market_description', '')
                    else:
                        product_description = ''
                        market_description = ''
                except Exception as e:
                    product_description = ''
                    market_description = ''
                
                # Clean and format descriptions
                if product_description:
                    product_description = ' '.join(product_description.split())
                    if len(product_description) > 500:
                        product_description = product_description[:497] + '...'
                
                if market_description:
                    market_description = ' '.join(market_description.split())
                    if len(market_description) > 500:
                        market_description = market_description[:497] + '...'
                
                # 2. Adjust building_since to remove time (only if it needs formatting)
                building_since_needs_update = False
                building_since = None
                try:
                    original_building_since = profile_dict.get('building_since', '') or ''
                    if original_building_since and isinstance(original_building_since, str) and 'T' in original_building_since:
                        building_since = original_building_since.split('T')[0]
                        building_since_needs_update = True
                    elif original_building_since and not isinstance(original_building_since, str):
                        building_since = str(original_building_since) if original_building_since else ''
                        building_since_needs_update = True
                except Exception as e:
                    building_since = ''
                    building_since_needs_update = True
                
                # 3. Adjust funding format
                funding_amount = profile_dict.get('fundingamount', '')
                latest_deal_type = profile_dict.get('latestdealtype', '')
                
                formatted_funding = ''
                if funding_amount and latest_deal_type:
                    try:
                        # Convert funding amount to readable format
                        amount_num = float(funding_amount)
                        if amount_num >= 1000000000:  # Billions
                            formatted_amount = f"{amount_num / 1000000000:.1f}B"
                        elif amount_num >= 1000000:  # Millions
                            formatted_amount = f"{amount_num / 1000000:.1f}M"
                        elif amount_num >= 1000:  # Thousands
                            formatted_amount = f"{amount_num / 1000:.1f}K"
                        else:
                            formatted_amount = str(int(amount_num))
                        
                        formatted_funding = f"US$ {formatted_amount}, {latest_deal_type}"
                    except (ValueError, TypeError):
                        formatted_funding = profile_dict.get('funding', '') or ''
                
                # 4. Insert history using priority dict
                try:
                    history = find_priority(profile_dict, priority_dict_company, priority_dict_person) or ''
                except Exception as e:
                    logger.debug("Error processing history: %s", e, exc_info=True)
                    history = ''
                
                # 5. Check if repeat founder using all_experiences
                repeat_founder = False
                all_experiences = profile_dict.get('all_experiences', '')
                if all_experiences:
                    try:
                        if isinstance(all_experiences, str):
                            experiences_data = json.loads(all_experiences)
                        elif isinstance(all_experiences, list):
                            experiences_data = all_experiences
                        else:
                            experiences_data = []
                        
                        founder_count = 0
                        if isinstance(experiences_data, list):
                            for exp in experiences_data:
                                if isinstance(exp, dict):
                                    position = exp.get('position', '')
                                    if position and isinstance(position, str):
                                        position_lower = position.lower()
                                        if any(keyword in position_lower for keyword in ['founder', 'co-founder', 'ceo']):
                                            founder_count += 1
                        
                        repeat_founder = founder_count > 1
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.debug("Error parsing all_experiences for %s: %s", profile_dict.get('name', 'Unknown'), e)
                        # Fallback to existing repeat_founder check
                        repeat_founder = check_if_repeat_founder(profile_dict)
                
                # 6. Check if technical using all_experiences and education
                try:
                    technical = is_technical_founder(profile_dict)
                except Exception as e:
                    logger.debug("Error in technical founder check: %s", e, exc_info=True)
                    technical = False
                
                # Also check all_experiences for technical roles
                if not technical and all_experiences:
                    try:
                        if isinstance(all_experiences, str):
                            experiences_data = json.loads(all_experiences)
                        elif isinstance(all_experiences, list):
                            experiences_data = all_experiences
                        else:
                            experiences_data = []
                        
                        technical_roles = [
                            "software engineer", "data scientist", "machine learning engineer",
                            "research scientist", "cto", "hardware engineer", "embedded engineer",
                            "robotics engineer", "backend engineer", "full stack developer",
                            "devops engineer", "site reliability engineer", "security engineer", "engineer"
                        ]
                        
                        if isinstance(experiences_data, list):
                            for exp in experiences_data:
                                if isinstance(exp, dict):
                                    position = exp.get('position', '')
                                    if position and isinstance(position, str):
                                        position_lower = position.lower()
                                        if any(role in position_lower for role in technical_roles):
                                            technical = True
                                            break
                    except (json.JSONDecodeError, TypeError, AttributeError) as e:
                        logger.debug("Error parsing all_experiences for technical check: %s", e)
                        pass
                
                # Check education for technical degrees
                if not technical:
                    education_data = profile_dict.get('education', '')
                    if education_data:
                        try:
                            if isinstance(education_data, str):
                                edu_data = json.loads(education_data)
                            elif isinstance(education_data, list):
                                edu_data = education_data
                            else:
                                edu_data = []
                            
                            technical_degrees = [
                                "computer science", "engineering", "electrical engineering",
                                "mechanical engineering", "civil engineering", "physics",
                                "mathematics", "statistics", "data science", "machine learning"
                            ]
                            
                            if isinstance(edu_data, list):
                                for edu in edu_data:
                                    if isinstance(edu, dict):
                                        degree = edu.get('degree', '')
                                        if degree and isinstance(degree, str):
                                            degree_lower = degree.lower()
                                            if any(tech_degree in degree_lower for tech_degree in technical_degrees):
                                                technical = True
                                                break
                        except (json.JSONDecodeError, TypeError, AttributeError) as e:
                            logger.debug("Error parsing education data for technical check: %s", e)
                            pass
                
                # 7. Generate tags
                try:
                    company_tags = get_past_notable_company(profile_dict, use_json=True)
                except Exception as e:
                    logger.debug("Error generating company tags for %s: %s", profile_dict.get('name', 'Unknown'), e)
                    company_tags = ''
                
                try:
                    school_tags = get_past_notable_education(profile_dict)
                except Exception as e:
                    logger.debug("Error generating school tags for %s: %s", profile_dict.get('name', 'Unknown'), e)
                    school_tags = ''
                
                try:
                    verticals = generate_verticals(profile_dict, use_json=True)
                except Exception as e:
                    logger.debug("Error generating verticals for %s: %s", profile_dict.get('name', 'Unknown'), e)
                    verticals = ''
                
                # 8. Update the profile in the database
                update_data = {
                    'product': product_description,
                    'market': market_description,
                    'funding': formatted_funding or profile_dict.get('funding', ''),
                    'history': history,
                    'repeat_founder': repeat_founder,
                    'technical': technical,
                    'company_tags': company_tags,
                    'school_tags': school_tags,
                    'verticals': verticals
                }
                
                # Only include building_since if it needs updating
                if building_since_needs_update and building_since is not None:
                    update_data['building_since'] = building_since
                
                # Remove empty values
                update_data = {k: v for k, v in update_data.items() if v is not None}
                
                # Update the profile
                success = update_profile_in_db(profile_dict['profile_url'], update_data, 'founders')
                
                if success:
                    processed_count += 1
                
            except Exception as e:
                continue
        
        logger.info("Processed %d profiles in add_monty_data", processed_count)
        
    except Exception as e:
        logger.error("Database error in add_monty_data: %s", e, exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def add_ai_scoring():
    """
    Add AI scoring to founder profiles using the scoring functions from ai_scoring module.
    Scores: past_success_indication_score, startup_experience_score, company_tech_score, industry_expertise_score
    """
    from services.ai_scoring import (
        past_success_indication_score,
        startup_experience_score,
        company_tech_score,
        prompt_industry_score,
    )
    from services.database import get_db_connection

    conn = get_db_connection()
    cursor = None

    try:
        cursor = conn.cursor()

        # Fetch profiles that need AI scoring (limit to prevent API overload)
        query = """
        SELECT * FROM founders 
        WHERE founder = true 
        AND all_experiences IS NOT NULL 
        AND (past_success_indication_score IS NULL 
             OR startup_experience_score IS NULL 
             OR company_tech_score IS NULL 
             OR industry_expertise_score IS NULL)
        """

        cursor.execute(query)
        profiles = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

        if not profiles:
            logger.info("No profiles found that need AI scoring")
            return
        
        logger.info("Profiles to score: %d", len(profiles))

        processed_count = 0

        for profile_row in profiles:
            try:
                # Convert to dictionary
                profile_dict = dict(zip(column_names, profile_row))


                # Prepare update data
                update_data = {}

                import time
                
                # 1. Past Success Indication Score
                if profile_dict.get("past_success_indication_score") is None:
                    try:
                        score = past_success_indication_score(profile_dict, json=True)
                        update_data["past_success_indication_score"] = score
                        time.sleep(0.5)  # Rate limiting delay
                    except Exception as e:
                        logger.warning("Error scoring past success: %s", e, exc_info=True)

                # 2. Startup Experience Score
                if profile_dict.get("startup_experience_score") is None:
                    try:
                        score = startup_experience_score(profile_dict, json=True)
                        update_data["startup_experience_score"] = score
                        time.sleep(0.5)  # Rate limiting delay
                    except Exception as e:
                        logger.warning("Error scoring startup experience: %s", e, exc_info=True)

                # 3. Company Tech Score
                if profile_dict.get("company_tech_score") is None:
                    try:
                        score = company_tech_score(profile_dict, json=True)
                        update_data["company_tech_score"] = score
                        time.sleep(0.5)  # Rate limiting delay
                    except Exception as e:
                        logger.warning("Error scoring company tech: %s", e, exc_info=True)

                # 4. Industry Expertise Score
                if profile_dict.get("industry_expertise_score") is None:
                    try:
                        score = prompt_industry_score(profile_dict, json=True)
                        update_data["industry_expertise_score"] = score
                        time.sleep(0.5)  # Rate limiting delay
                    except Exception as e:
                        logger.warning("Error scoring industry expertise: %s", e, exc_info=True)

                # Update the profile if we have scores to add
                if update_data:

                    try:
                        # Build the SET clause for update
                        set_clause = ", ".join([f'"{col}" = %s' for col in update_data.keys()])
                        update_query = f"""
                            UPDATE founders
                            SET {set_clause}
                            WHERE id = %s
                        """
                        values = list(update_data.values()) + [profile_dict["id"]]
                        cursor.execute(update_query, values)

                        if cursor.rowcount > 0:
                            conn.commit()
                            processed_count += 1
                        else:
                            logger.warning("No row found for id=%s (url=%s)", profile_dict['id'], profile_dict.get('profile_url'))

                    except Exception as e:
                        conn.rollback()
                        logger.error("Error updating profile %s: %s", profile_dict.get('name', 'Unknown'), e, exc_info=True)
                else:
                    logger.debug("No scores needed for: %s", profile_dict.get('name', 'Unknown'))

            except Exception as e:
                logger.warning("Error processing profile %s: %s", profile_dict.get('name', 'Unknown'), e, exc_info=True)
                continue

        logger.info("Processed %d profiles in add_ai_scoring", processed_count)

    except Exception as e:
        logger.error("Database error in add_ai_scoring: %s", e, exc_info=True)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def add_tree_analysis():
    # Get the new profiles that dont have tree analysis
    from services.tree import test_tree
    test_tree()

def search_aviato_companies(search_filters):
    # Base DSL
    dsl = {
        "offset": 0,
        "limit": 10000,
        "sort": [{"name": "asc"}]
    }

    # Optional: add nameQuery if provided
    if "nameQuery" in search_filters:
        dsl["nameQuery"] = search_filters["nameQuery"]

    # Build filters dynamically
    filter_conditions = []
    if "country" in search_filters:
        filter_conditions.append({"country": {"operation": "eq", "value": search_filters["country"]}})
    if "region" in search_filters:
        # Support both single region and multiple regions
        region_value = search_filters["region"]
        if isinstance(region_value, list):
            filter_conditions.append({"region": {"operation": "in", "value": region_value}})
        else:
            filter_conditions.append({"region": {"operation": "eq", "value": region_value}})
    if "locality" in search_filters:
        # Support both single locality and multiple localities
        locality_value = search_filters["locality"]
        if isinstance(locality_value, list):
            filter_conditions.append({"locality": {"operation": "in", "value": locality_value}})
        else:
            filter_conditions.append({"locality": {"operation": "eq", "value": locality_value}})
    if "locationIDList" in search_filters:
        filter_conditions.append({"locationIDList": {"operation": "in", "value": search_filters["locationIDList"]}})
    if "industryList" in search_filters:
        filter_conditions.append({"industryList": {"operation": "in", "value": search_filters["industryList"]}})
    if "website" in search_filters:
        filter_conditions.append({"website": {"operation": "eq", "value": search_filters["website"]}})
    if "linkedin" in search_filters:
        filter_conditions.append({"linkedin": {"operation": "eq", "value": search_filters["linkedin"]}})
    if "twitter" in search_filters:
        filter_conditions.append({"twitter": {"operation": "eq", "value": search_filters["twitter"]}})
    if "totalFunding" in search_filters:
        filter_conditions.append({"totalFunding": {"operation": "lte", "value": search_filters["totalFunding"]}})
    if "founded" in search_filters:
        # Handle founded date - convert year to ISO datetime format for comparison
        founded_value = search_filters["founded"]
        if isinstance(founded_value, int):
            # If it's a year, convert to end of year datetime for "lte" comparison
            founded_value = f"{founded_value}-12-31T23:59:59Z"
        elif isinstance(founded_value, str) and len(founded_value) == 4 and founded_value.isdigit():
            # If it's a year string, convert to end of year datetime
            founded_value = f"{founded_value}-12-31T23:59:59Z"
        
        filter_conditions.append({"founded": {"operation": "gte", "value": founded_value}})
    
    # Wrap filters in AND structure if any exist
    if filter_conditions:
        dsl["filters"] = [{"AND": filter_conditions}]

    payload = {"dsl": dsl}

    url = "https://data.api.aviato.co/company/search"

    headers = {
        "Authorization": f"Bearer {aviato_api}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        logger.error("Profile search error: %s | %s", response.status_code, response.text[:200])
        return None

def search_aviato_profiles(search_filters):
    # Base DSL
    dsl = {
        "offset": 0,
        "limit": 10
    }

    # Optional: add nameQuery if provided
    if "id" in search_filters:
        dsl["id"] = search_filters["id"]
    if "fullName" in search_filters:
        dsl["fullName"] = search_filters["fullName"]

    # Build filters dynamically
    filters = []
    if "location" in search_filters:
        filters.append({"location": {"operation": "eq", "value": search_filters["location"]}})
    if "website" in search_filters:
        filters.append({"website": {"operation": "eq", "value": search_filters["website"]}})
    if "linkedin" in search_filters:
        filters.append({"linkedin": {"operation": "eq", "value": search_filters["linkedin"]}})
    if "twitter" in search_filters:
        filters.append({"twitter": {"operation": "eq", "value": search_filters["twitter"]}})

    # Attach filters if any
    if filters:
        dsl["filters"] = filters

    payload = {"dsl": dsl}

    url = "https://data.api.aviato.co/person/search"

    headers = {
        "Authorization": f"Bearer {aviato_api}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        logger.error("Profile search error: %s | %s", response.status_code, response.text[:200])


def enrich_companies(company_ids, chunk_size=100, max_retries=2):
    """
    Enrich companies in chunks of up to 100 IDs each.
    Returns a flat list of enriched company objects.
    """
    import time
    
    url = "https://data.api.aviato.co/company/bulk-enrich"
    headers = {
        "Authorization": f"Bearer {aviato_api}",
        "Content-Type": "application/json"
    }

    enriched = []
    # Iterate through company_ids in chunks
    for i in range(0, len(company_ids), chunk_size):
        chunk = company_ids[i:i+chunk_size]
        payload = {"lookups": [{"id": cid} for cid in chunk]}
        
        # Retry logic for 500 errors
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json().get("companies", [])
                    enriched.extend(data)
                    break  # Success, exit retry loop
                elif response.status_code == 500 and attempt < max_retries:
                    logger.warning("500 error on chunk %d, attempt %d/%d. Retrying in 2 seconds...", 
                                 i//chunk_size + 1, attempt + 1, max_retries + 1)
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    logger.warning("Error enriching chunk %d: %s | %s", 
                                 i//chunk_size + 1, response.status_code, response.text[:200])
                    break  # Don't retry for non-500 errors or after max retries
            except requests.exceptions.RequestException as e:
                logger.warning("Request exception on chunk %d, attempt %d: %s", 
                             i//chunk_size + 1, attempt + 1, str(e))
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                else:
                    break

    return enriched

def filter_relevant(companies):
    """Filter out older, well-funded, or irrelevant companies."""
    relevant = []
    for c in companies:
        if not c.get("lookupSuccessful"):
            continue
        co = c["company"]
        # Sample rules — tweak as you like
        if co.get("isAcquired") or co.get("isShutDown"):
            continue
        if co.get("founded") and int(co["founded"][:4]) < 2021:
            continue
        if co.get("totalFunding") and co["totalFunding"] > 5000000:  # too late stage
            continue

        relevant.append(co)
    return relevant

def find_founder(company_id):
    # Respect dedicated RPM for founders endpoint
    _wait_for_founder_rate_limit()

    url = "https://data.api.aviato.co/company/" + company_id + "/founders?perPage=1&page=1"

    headers = {
        "Authorization": f"Bearer {aviato_api}"
    }

    response = request_with_backoff(
        "GET",
        url,
        headers=headers,
    )

    if response is None:
        logger.error("Find founder error: No response for company %s", company_id)
        return None

    if response.status_code == 200:
        try:
            data = response.json()
            return data
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            logger.error("Find founder JSON decode error for %s: %s | Snippet: %s", company_id, e, response.text[:200])
            return None
    else:
        logger.error("Find founder error: %s | %s", response.status_code, response.text[:200])
        return None

def aviato_search_collect(search_filters, source="aviato_search"):
    """
    Collect founder data from Aviato search without inserting to database.
    Returns list of founder dictionaries.
    """
    data = search_aviato_companies(search_filters)
    if not data or "items" not in data:
        logger.warning("No company data returned for search: %s", source)
        return []
        
    ids = [item["id"] for item in data["items"]]
    logger.info("Found %d companies", len(ids))
    companies = enrich_companies(ids)
    relevant = filter_relevant(companies)
    logger.info("Found %d relevant companies", len(relevant))
    founder_data = []
    
    for co in relevant:
        company_id = co["id"]
        company_name = co["name"]
        company_url = co.get("website", "")

        founder = find_founder(company_id)
        if not founder or "founders" not in founder:
            continue
            
        for f in founder.get("founders", []):
            # Skip if no fullName
            if not f.get('fullName'):
                continue
                
            # Get LinkedIn URL safely
            urls = f.get('URLs', {})
            linkedin_url = urls.get('linkedin', '')
            
            # Skip if no LinkedIn URL
            if not linkedin_url:
                continue
                
            founder_data.append({
                'name': f['fullName'],
                'profile_url': linkedin_url,
                'title': f.get('headline', ''),
                'source': source,
                'company_name': company_name,
                'company_url': company_url
            })
    
    logger.info("Collected %d founders from %s", len(founder_data), source)
    return founder_data

def aviato_search(search_filters, source="aviato_search"):
    data = search_aviato_companies(search_filters)
    if not data or "items" not in data:
        logger.warning("No company data returned for search: %s", source)
        return
        
    ids = [item["id"] for item in data["items"]]
    logger.info("Found %d companies", len(ids))
    companies = enrich_companies(ids)
    relevant = filter_relevant(companies)
    logger.info("Found %d relevant companies", len(relevant))
    founder_data = []
    
    for co in relevant:
        company_id = co["id"]
        company_name = co["name"]
        company_url = co.get("website", "")

        founder = find_founder(company_id)
        if not founder or "founders" not in founder:
            continue
            
        for f in founder.get("founders", []):
            # Skip if no fullName
            if not f.get('fullName'):
                continue
                
            # Get LinkedIn URL safely
            urls = f.get('URLs', {})
            linkedin_url = urls.get('linkedin', '')
            
            # Skip if no LinkedIn URL
            if not linkedin_url:
                continue
                
            founder_data.append({
                'name': f['fullName'],
                'profile_url': linkedin_url,
                'title': f.get('headline', ''),
                'source': source,
                'company_name': company_name,
                'company_url': company_url
            })
    
    # Create DataFrame and insert to database
    if founder_data:
        df = pd.DataFrame(founder_data)
        # Remove duplicates within this batch based on profile_url
        df = df.drop_duplicates(subset=['profile_url'], keep='first')
        logger.info("After deduplication: %d unique founders", len(df))
        
        success = insert_search_results(df, table_name="search_list")
        if success:
            logger.info("Inserted %d founders to search_list", len(df))
        else:
            logger.error("Failed to insert founder data")
    else:
        logger.info("No founder data to insert")

def aviato_discover():
    import json
    total_inserted = 0
    file_paths = ["config/aviato_search_fintech.json", "config/aviato_search_healthcare.json", "config/aviato_search_commerce.json", "config/aviato_search_other.json"]

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            for search_config in data.get("search_filters", []):
                search_name = search_config.get("name", "unknown")
                logger.info("Starting search: %s", search_name)
                
                # Collect founder data for this search
                founder_data = aviato_search_collect(search_config.get("filter", {}), search_config.get("source", "aviato_search"))
                
                if founder_data:
                    # Convert to DataFrame and insert immediately
                    df = pd.DataFrame(founder_data)
                    logger.info("Inserting %d founders from %s search", len(df), search_name)
                    
                    success = insert_search_results(df, table_name="search_list")
                    if success:
                        total_inserted += len(df)
                        logger.info("Successfully inserted %d founders from %s. Total so far: %d", len(df), search_name, total_inserted)
                    else:
                        logger.error("Failed to insert founder data from %s search", search_name)
                else:
                    logger.info("No founder data collected from %s search", search_name)
        
        logger.info("Discovery complete. Total founders inserted: %d", total_inserted)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aviato processing pipeline")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run discovery search instead of processing pipeline",
    )

    args = parser.parse_args()

    # Initialize logging using env LOG_LEVEL (INFO by default)
    setup_logging()

    file_paths = ["config/aviato_search_fintech.json", "config/aviato_search_healthcare.json", "config/aviato_search_commerce.json", "config/aviato_search_other.json"]

    if args.discover:
        aviato_discover()
    else:
        process_profiles_aviato(max_profiles=1000)
        add_monty_data()
        add_ai_scoring()
        add_tree_analysis()


    