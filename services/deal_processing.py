import email
import re
import requests
from bs4 import BeautifulSoup
import asyncio
import pandas as pd
import requests
from serpapi import GoogleSearch
import os
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from services.groq_api import get_groq_response
from services.openai_api import web_search
from datetime import datetime

def process_vcnewsdaily(email_text):
    """
    Process VC News Daily email to extract company names, funding round types, sizes, and article links.
    
    Args:
        email_text (str): The parsed email text content
        
    Returns:
        list: List of dictionaries containing deal information
    """
    deals = []
    
    # Find the Recent Articles section
    if "** Recent Articles:" in email_text:
        articles_section = email_text.split("** Recent Articles:")[1]
        # Find the end of the articles section (marked by a line of equal signs)
        if "============================================================" in articles_section:
            articles_section = articles_section.split("============================================================")[0].strip()
        
        # Process each article line
        for line in articles_section.split('\n'):
            line = line.strip()
            # Skip empty lines or separator lines
            if not line or line.startswith("----"):
                continue
                
            # Match the pattern: ** Company Name Verb $X.YM Round Type (URL)
            if line.startswith("**"):
                # Remove the leading **
                line = line.lstrip("* ").strip()
                
                # Extract the URL
                url_match = re.search(r'\((https?://[^)]+)\)', line)
                if url_match:
                    url = url_match.group(1)
                    # Remove the URL part from the line
                    line = line.split("(https")[0].strip()
                    
                    # Extract funding amount using regex
                    amount_match = re.search(r'\$(\d+\.?\d*)M?', line)
                    amount = amount_match.group(0) if amount_match else ""
                    
                    # Identify round type
                    round_types = ["Seed", "Pre-Seed", "Series A", "Series B", "Series C", 
                                  "Series D", "Financing", "Round", "Funding"]
                    
                    found_round_type = None
                    for round_type in round_types:
                        if round_type in line:
                            found_round_type = round_type
                            break
                    
                    # If round type contains just "Round" or "Financing", look for more specific terms
                    if found_round_type in ["Round", "Financing", "Funding"]:
                        if "Seed" in line:
                            found_round_type = "Seed"
                        elif "Pre-Seed" in line or "Pre Seed" in line:
                            found_round_type = "Pre-Seed"
                        elif "Series A" in line:
                            found_round_type = "Series A"
                        elif "Series B" in line:
                            found_round_type = "Series B"
                        elif "Series C" in line:
                            found_round_type = "Series C"
                    
                    # Extract company name - it's typically the first part of the line before the verb
                    # Common verbs in these headlines
                    verbs = ["Completes", "Emerges", "Announces", "Scores", "Lands", "Nabs", 
                            "Secures", "Launches", "Inks", "Scoops", "Raises", "Closes"]
                    
                    company_name = line
                    for verb in verbs:
                        if f" {verb} " in line:
                            company_name = line.split(f" {verb} ")[0].strip()
                            break
                    
                    # Create the deal dictionary
                    deal = {
                        "Company": company_name,
                        "Amount": amount,
                        "Funding Round": found_round_type if found_round_type else "Unspecified",
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Source": "VC News Daily",
                        "Investors": "",  # Not available in the headline
                        "Vertical": "",    # Not available in the headline,-
                        "Category": "",    # Not available in the headline
                        "Link": url
                    }
                    deal = fill_in_the_blanks(deal)
                    deals.append(deal)
    return deals

def process_fortune_termsheet(email_text):
    import re
    """
    Process Fortune Term Sheet email to extract company names, funding round types, sizes, investors, and descriptions.
    Uses Groq to extract structured information from each deal entry.
    
    Args:
        email_text (str): The parsed email text content
        
    Returns:
        list: List of dictionaries containing deal information
    """
    deals = []
    # Find the VENTURE DEALS section
    if "Venture Deals" in email_text:
        print("Found VENTURE DEALS section")
        deals_section = email_text.split("Venture Deals")[1]
        
        # Find the end of the deals section (usually marked by another section like PRIVATE EQUITY)
        end_markers = ["Private Equity"]
        for marker in end_markers:
            if marker in deals_section:
                deals_section = deals_section.split(marker)[0].strip()
                break
        
        # Add debugging to see the full deals section
        print("\n\nDEALS SECTION:")
        print(deals_section[:500] + "..." if len(deals_section) > 500 else deals_section)
        print("\n\n")
        
        # Fortune Term Sheet format has deals that start with a dash
        # We need to split by the dash pattern at the beginning of lines
        
        # First, normalize line breaks to ensure consistent processing
        normalized_section = re.sub(r'\r\n', '\n', deals_section)
        
        # Split by the dash pattern that starts each deal
        # The pattern looks for a dash at the beginning of a line, possibly with whitespace
        entries = re.split(r'\n\s*-\s+', '\n' + normalized_section)
        
        # Skip the first entry if it's just the header
        if not entries[0].strip() or not re.search(r'[A-Za-z]', entries[0]):
            entries = entries[1:]
        
        # Add the dash back to each entry for context
        deal_entries = [f"- {entry.strip()}" for entry in entries if entry.strip()]
        
        print(f"Found {len(deal_entries)} potential deal entries")
        for i, entry in enumerate(deal_entries[:5]):  # Show first 5 for debugging
            print(f"\nEntry {i+1}:\n{entry[:200]}..." if len(entry) > 200 else entry)
        
        for entry in deal_entries:
            if not entry.strip():
                continue
            
            # Use Groq to extract structured information from each deal entry
            prompt = f"""
            Extract the following information from this startup funding announcement:
            
            1. Company name
            2. Funding amount (in USD if available)
            3. Funding round type (be specific: Seed, Pre-Seed, Series A, Series B, etc.)
            4. Vertical (specific business area in 3-7 words)
            5. Investors (list of investors)
            6. Category (classify broadly as one of: fintech, healthcare, commerce, or other)
            
            For the funding amount, always format it as:
            - For millions: $XM (e.g., $50M, $7.5M, $1.2M)
            - For thousands: $XK (e.g., $500K, $750K)
            - Always round to one decimal place if needed
            - Do not spell out "million" or "thousand"
            - Always include the dollar sign
            
            For the category, classify the company as one of: fintech, healthcare, commerce, or other.
            - Fintech includes: AI-powered financial services, financial infrastructure, embedded finance, financial inclusion, next-generation financial tools, payments, lending, investing
            - Healthcare includes: digital therapeutics, AI in healthcare, healthcare infrastructure, techbio, patient experience platforms, medical devices, biotech
            - Commerce includes: AI-powered commerce tools, retail infrastructure, supply chain optimization, commerce enablement, innovative shopping experiences, e-commerce, marketplaces
            - Other: if it truly doesn't fit into any of the above categories
            
            Here's the deal announcement:
            {entry.strip()}
            
            Format your response as a JSON object with these keys:
            {{"company": "...", "amount": "...", "funding_round": "...", "vertical": "...", "investors": "...", "category": "..."}}
            """
            
            try:
                # Get response from Groq
                response = get_groq_response(prompt)
                
                # Extract the JSON part from the response
                import json
                import re
                
                # Find JSON-like content in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    extracted_info = json.loads(json_str)
                    
                    # Create the deal dictionary
                    deal = {
                        "Company": extracted_info.get("company", ""),
                        "Amount": extracted_info.get("amount", ""),
                        "Funding Round": extracted_info.get("funding_round", ""),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Source": "Fortune Term Sheet",
                        "Investors": extracted_info.get("investors", ""),
                        "Vertical": extracted_info.get("vertical", ""),
                        "Category": extracted_info.get("category", ""),
                        "Link": ""  # No direct link in the email
                    }
                    print(deal)
                    deals.append(deal)
                    
            except Exception as e:
                print(f"Error extracting information with Groq: {e}")
                # Create a minimal deal entry with just the raw text
                first_line = entry.strip().split("\n")[0] if entry.strip() else ""
                company_name = first_line.strip() if first_line else "Unknown Company"
                
                deal = {
                    "Company": company_name,
                    "Amount": "",
                    "Funding Round": "",
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Source": "Fortune Term Sheet",
                    "Investors": "",
                    "Vertical": "",
                    "Category": "",
                    "Link": ""
                }
                deals.append(deal)
    else:
        print("No VENTURE DEALS section found")
        print(email_text)
    
    return deals

def process_fresh_funding(email_text):
    """
    Process F6S email to extract company names, funding round types, sizes, investors, and descriptions.
    Uses Groq to extract structured information from each deal entry.
    
    Args:
        email_text (str): The parsed email text content
        
    Returns:
        list: List of dictionaries containing deal information
    """
    from datetime import datetime
    
    deals = []
    
    # Find the end of the deals section (usually marked by "Top new startup product launches")
    deals_section = email_text
    end_markers = ["Top new startup product launches"]
    for marker in end_markers:
        if marker in email_text:
            deals_section = email_text.split(marker)[0].strip()
            break
    
    # Split by triple newlines to get individual deal entries
    entries = deals_section.split("\n\n\n")
    
    for entry in entries:
        entry = entry.strip()
        # Skip non-deal entries
        if not entry or "your profile for better matches" in entry or len(entry) < 20:
            continue
            
        # Check if this looks like a funding announcement
        if "for" in entry and "from" in entry and (
            "$" in entry or "Seed" in entry or "Series" in entry or "Acquisition" in entry):
            
            try:
                # Use Groq to extract structured information from each deal entry
                prompt = f"""
                Extract the following information from this startup funding announcement:
                
                1. Company name
                2. Funding amount (in USD if available)
                3. Funding round type (be specific: Seed, Pre-Seed, Series A, Series B, etc.)
                4. Vertical (specific business area in 3-7 words)
                5. Investors (list of investors)
                6. Category (classify broadly as one of: fintech, healthcare, commerce, or other)
                
                For the funding amount, always format it as:
                - For millions: $XM (e.g., $50M, $7.5M, $1.2M)
                - For thousands: $XK (e.g., $500K, $750K)
                - Always round to one decimal place if needed
                - Do not spell out "million" or "thousand"
                - Always include the dollar sign
                
                For the category, classify the company as one of: fintech, healthcare, commerce, or other.
                - Fintech includes: AI-powered financial services, financial infrastructure, embedded finance, financial inclusion, next-generation financial tools, payments, lending, investing
                - Healthcare includes: digital therapeutics, AI in healthcare, healthcare infrastructure, techbio, patient experience platforms, medical devices, biotech
                - Commerce includes: AI-powered commerce tools, retail infrastructure, supply chain optimization, commerce enablement, innovative shopping experiences, e-commerce, marketplaces
                - Other: if it truly doesn't fit into any of the above categories
                
                Here's the deal announcement:
                {entry.strip()}
                
                Format your response as a JSON object with these keys:
                {{"company": "...", "amount": "...", "funding_round": "...", "vertical": "...", "investors": "...", "category": "..."}}
                """
                
                # Get response from Groq
                response = get_groq_response(prompt)
                
                # Extract the JSON part from the response
                import json
                import re
                
                # Find JSON-like content in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    extracted_info = json.loads(json_str)
                    
                    # Create the deal dictionary
                    deal = {
                        "Company": extracted_info.get("company", ""),
                        "Amount": extracted_info.get("amount", ""),
                        "Funding Round": extracted_info.get("funding_round", ""),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Source": "F6S Fresh Funding",
                        "Investors": extracted_info.get("investors", ""),
                        "Vertical": extracted_info.get("vertical", ""),
                        "Category": extracted_info.get("category", ""),
                        "Link": ""  # No direct link in the email
                    }
                    
                    deals.append(deal)
                    
            except Exception as e:
                print(f"Error extracting information with Groq: {e}")
    
    return deals

def process_daily_digest(email_text):
    print("Processing Daily Digest")
    import re
    start_marker = "💰 Startup funding updates"
    if start_marker not in email_text:
        print("No start marker found")
        return []

    # --- Step 1: Split into lines for better parsing ---
    lines = email_text.splitlines()

    # Step 2: Find start line index
    start_idx = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            break
    if start_idx is None:
        print("Start marker not found in lines")
        return []

    # Step 3: Read until next ALL CAPS heading (new section)
    startups_lines = []
    for j in range(start_idx + 1, len(lines)):
        # Detect a new ALL CAPS heading (like FROM OUR PARTNER, NEW VCs IN THE MARKET)
        if re.match(r"^[A-Z\s&\-]{5,}$", lines[j].strip()):
            break
        startups_lines.append(lines[j])

    # Step 4: Only keep lines that look like funding announcements
    entries = [line.strip() for line in startups_lines 
               if re.search(r"(raised|received|closed|secured|received).*?\$", line, re.IGNORECASE)]

    print(f"Entries found: {len(entries)}")
    deals = []
    
    for entry in entries:
        entry = entry.strip()
        if entry == "" or entry == "\n":
            continue
        try:
                # Use Groq to extract structured information from each deal entry
                prompt = f"""
                Extract the following information from this startup funding announcement:
                
                1. Company name
                2. Funding amount (in USD if available)
                3. Funding round type (be specific: Seed, Pre-Seed, Series A, Series B, etc.)
                4. Vertical (specific business area in 3-7 words)
                5. Investors (list of investors)
                6. Category (classify broadly as one of: fintech, healthcare, commerce, or other)
                
                For the funding amount, always format it as:
                - For millions: $XM (e.g., $50M, $7.5M, $1.2M)
                - For thousands: $XK (e.g., $500K, $750K)
                - Always round to one decimal place if needed
                - Do not spell out "million" or "thousand"
                - Always include the dollar sign
                
                For the category, classify the company as one of: fintech, healthcare, commerce, or other.
                - Fintech includes: AI-powered financial services, financial infrastructure, embedded finance, financial inclusion, next-generation financial tools, payments, lending, investing
                - Healthcare includes: digital therapeutics, AI in healthcare, healthcare infrastructure, techbio, patient experience platforms, medical devices, biotech
                - Commerce includes: AI-powered commerce tools, retail infrastructure, supply chain optimization, commerce enablement, innovative shopping experiences, e-commerce, marketplaces
                - Other: if it truly doesn't fit into any of the above categories
                
                Here's the deal announcement:
                {entry.strip()}
                
                Format your response as a JSON object with these keys:
                {{"company": "...", "amount": "...", "funding_round": "...", "vertical": "...", "investors": "...", "category": "..."}}
                """
                
                # Get response from Groq
                response = get_groq_response(prompt)
                
                # Extract the JSON part from the response
                import json
                import re
                
                # Find JSON-like content in the response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    extracted_info = json.loads(json_str)
                    
                    # Create the deal dictionary
                    deal = {
                        "Company": extracted_info.get("company", ""),
                        "Amount": extracted_info.get("amount", ""),
                        "Funding Round": extracted_info.get("funding_round", ""),
                        "Date": datetime.now().strftime("%Y-%m-%d"),
                        "Source": "F6S Fresh Funding",
                        "Investors": extracted_info.get("investors", ""),
                        "Vertical": extracted_info.get("vertical", ""),
                        "Category": extracted_info.get("category", ""),
                        "Link": ""  # No direct link in the email
                    }
                    non_deals = ["", "Unknown", "Not specified", "Not Disclosed", None, "None", "N/A", "?", "TBD", "Unnamed"]
                    if deal["Company"] not in non_deals or (deal["Funding Round"] is None and deal["Amount"] is None):
                        deals.append(deal)
                        print("Deal found: ")
                        print(deal)
        except Exception as e:
            print(f"Error extracting information with Groq: {e}")
    
    return deals

def fill_in_the_blanks(deal):
    if deal["Amount"] == "" or deal["Funding Round"].lower() not in ["pre-seed", "seed", "series a", "series b", "series c", "series d"] or deal["Investors"] == "" or deal["Category"] == "":
        if deal["Link"] != "":
            try:
                response = requests.get(deal["Link"], timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    article_text = soup.get_text()
                    article_text = '\n'.join([line.strip() for line in article_text.split('\n') if line.strip()])
                    deal["Detail"] = article_text
                    deal = extract_info_from_article(deal, article_text)
            except Exception as e:
                print(f"Error fetching article content: {e}")
    return deal  

def extract_info_from_article(deal, article_text):
    """
    Use Groq to extract missing information from the article text.
    
    Args:
        deal (dict): The deal dictionary with potentially missing information
        article_text (str): The full text of the article
        
    Returns:
        dict: Updated deal dictionary with extracted information
    """
    # Only process if we have article text
    if not article_text:
        return deal
    
    # Determine what information is missing or needs improvement
    missing_info = []
    if not deal["Amount"] or deal["Amount"] == "":
        missing_info.append("funding amount")
    
    # Check if funding round needs improvement (generic terms like "Financing" or "Round")
    generic_rounds = ["financing", "round", "funding", "unspecified"]
    if not deal["Funding Round"] or deal["Funding Round"] == "" or deal["Funding Round"].lower() in generic_rounds:
        missing_info.append("specific funding round type")
    
    if not deal["Investors"] or deal["Investors"] == "":
        missing_info.append("investors")
    
    if not deal["Category"] or deal["Category"] == "":
        missing_info.append("category (fintech, healthcare, commerce, or other)")
        
    if not deal["Vertical"] or deal["Vertical"] == "":
        missing_info.append("vertical (specific business area)")
    
    # If nothing is missing, return the deal as is
    if not missing_info:
        return deal
    
    # Construct a prompt for Groq
    prompt = f"""
    Extract the following information from this startup funding article about {deal['Company']}:
    
    {', '.join(missing_info)}
    
    For the funding round type, be specific (e.g., "Seed", "Pre-Seed", "Series A", "Series B", etc.) - do not use generic terms like "financing" or "round".
    
    For the category, classify the company as one of: fintech, healthcare, commerce, or other.
    - Fintech includes: AI-powered financial services, financial infrastructure, embedded finance, financial inclusion, next-generation financial tools, payments, lending, investing
    - Healthcare includes: digital therapeutics, AI in healthcare, healthcare infrastructure, techbio, patient experience platforms, medical devices, biotech
    - Commerce includes: AI-powered commerce tools, retail infrastructure, supply chain optimization, commerce enablement, innovative shopping experiences, e-commerce, marketplaces
    - Other: if it truly doesn't fit into any of the above categories
    
    For the vertical, provide a brief (3-7 words) description of the specific business area or industry niche the company operates in (e.g., "datacenter cooling technology", "sustainable fashion marketplace", "AI-powered credit underwriting").
    
    Here's the article text:
    {article_text[:4000]}  # Limit text length to avoid token limits
    
    Format your response as a JSON object with these keys (only include keys that were requested):
    {{"amount": "...", "funding_round": "...", "investors": "...", "category": "...", "vertical": "..."}}
    """
    
    try:
        # Get response from Groq
        response = get_groq_response(prompt)
        
        # Extract the JSON part from the response
        import json
        import re
        
        # Find JSON-like content in the response - use a non-greedy match
        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
            # Completely rebuild the JSON to avoid parsing issues
            extracted_info = {}
            
            # Extract key-value pairs using regex
            # This pattern looks for "key":"value" patterns, handling various quote and escape issues
            pattern = r'"(\w+)"\s*:\s*"([^"]*(?:\\"[^"]*)*?)"'
            matches = re.findall(pattern, json_str)
            
            if matches:
                for key, value in matches:
                    # Clean the value - remove problematic backslashes
                    clean_value = value.replace('\\"', '"')  # Replace escaped quotes
                    clean_value = re.sub(r'\\(?!")', '', clean_value)  # Remove other backslashes
                    extracted_info[key] = clean_value
            else:
                # If regex failed, try a more aggressive approach
                try:
                    # Remove all backslashes before quotes
                    clean_json = re.sub(r'\\"', '"', json_str)
                    # Replace all remaining backslashes with nothing
                    clean_json = re.sub(r'\\', '', clean_json)
                    # Fix any double quotes that might cause issues
                    clean_json = re.sub(r'"([^":,{}]+)"\s*"', r'"\1","', clean_json)
                    # Try to parse the cleaned JSON
                    extracted_info = json.loads(clean_json)
                except json.JSONDecodeError as json_err:
                    print(f"Advanced JSON cleaning failed: {json_err}")
                    print(f"Problematic JSON string: {json_str}")
                    
                    # Last resort: extract individual key-value pairs with a more lenient pattern
                    try:
                        # Extract keys
                        keys = re.findall(r'"(\w+)"\s*:', json_str)
                        # Split the string by these keys
                        parts = re.split(r'"\w+"\s*:', json_str)
                        if len(parts) > 1:  # Skip the part before the first key
                            parts = parts[1:]
                        
                        if len(keys) == len(parts):
                            for i, key in enumerate(keys):
                                # Extract value - everything up to the next comma or closing brace
                                value_match = re.search(r'\s*"([^"}]+)"', parts[i])
                                if value_match:
                                    extracted_info[key] = value_match.group(1)
                    except Exception as e:
                        print(f"Last resort parsing failed: {e}")
            
            # Update the deal with extracted information
            if "amount" in extracted_info and extracted_info["amount"] and (not deal["Amount"] or deal["Amount"] == ""):
                deal["Amount"] = extracted_info["amount"]
                
            if "funding_round" in extracted_info and extracted_info["funding_round"]:
                # Only update if current round is generic or missing
                current_round = deal["Funding Round"].lower()
                if current_round in generic_rounds or current_round == "":
                    deal["Funding Round"] = extracted_info["funding_round"]
                    
            if "investors" in extracted_info and extracted_info["investors"]:
                deal["Investors"] = extracted_info["investors"]
                
            if "category" in extracted_info and extracted_info["category"]:
                deal["Category"] = extracted_info["category"]
                
            if "vertical" in extracted_info and extracted_info["vertical"]:
                deal["Vertical"] = extracted_info["vertical"]
    except Exception as e:
        print(f"Error extracting information with Groq: {e}")
    
    return deal

def serpapi_search(deal_dict, num_results=10):
    company = deal_dict["Company"]
    amount = deal_dict["Amount"]
    round_type = deal_dict["Funding Round"]
    investors = deal_dict["Investors"]
    search_query = f"{company} startup funding"
    if investors and len(investors) > 0:
        search_query += f" {investors[0]}"  # Add the first investor to the search
    
    search = GoogleSearch({
        "q": search_query,
        "engine": "google",
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": num_results,
        "tbs": "qdr:w"
    })
    results = search.get_dict()
    links = []
    for r in results.get("organic_results", []):
        links.append({
            "link": r.get("link"),
            "title": r.get("title")
        })
    best_link = pick_best_link(links, company, amount, round_type)
    print("Found links for deal: ", company, "Links: ", links)
    print("Best link: ", best_link)
    return best_link

def score_result(result, company, amount=None, round_type=None):
    RELEVANT_DOMAINS = {
    "techcrunch.com", "venturebeat.com", "businesswire.com",
    "prnewswire.com", "forbes.com", "wsj.com", "bloomberg.com",
    "reuters.com", "fortune.com", "finsmes.com", "vcnewsdaily.com",
    "eu-startups.com", "techfundingnews.com", "siliconcanals.com",
    "pymnts.com", "finextra.com"
    }

    FUNDING_KEYWORDS = [
        "raises", "raised", "funding", "investment", "announces",
        "secures", "bags", "lands", "closes", "series", "seed", "round"
    ]
    score = 0
    title = result["title"].lower()
    link = result["link"].lower()

    # 1. Company name match
    if company.lower() in title or company.lower() in link:
        score += 10

    # 2. Funding keywords in title
    if any(kw in title for kw in FUNDING_KEYWORDS):
        score += 5

    # 3. Amount match (if you have it)
    if amount and amount.lower().replace(" ", "") in title.replace(" ", ""):
        score += 3

    # 4. Round type match
    if round_type and round_type.lower() in title:
        score += 5

    # 5. Preferred domain
    domain = urlparse(link).netloc.replace("www.", "")
    if domain in RELEVANT_DOMAINS:
        score += 5

    return score

def pick_best_link(results, company, amount=None, round_type=None):
    scored = [
        (score_result(r, company, amount, round_type), r)
        for r in results
    ]
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]["link"] if scored else None

def ddg_search(deal_dict):
    
    url = "https://duckduckgo.com/lite/"
    query = f"{deal_dict['Company']} startup funding"
    params = {"q": query}
    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    links = []

    for a in soup.select("a[href^='/l/?']"):  # DuckDuckGo lite link format
        href = a.get("href")
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            real_link = unquote(qs["uddg"][0])
            title = a.get_text(strip=True)
            links.append({"link": real_link, "title": title})

    company = deal_dict["Company"]
    amount = deal_dict["Amount"]
    round_type = deal_dict["Funding Round"]
    best_link = pick_best_link(links, company, amount, round_type)
    print("Found links for deal: ", company, "Links: ", links)
    print("Best link: ", best_link)
    return best_link

def find_link_if_missing(deal):
    """
    Search for a relevant announcement link if one is missing in the deal dictionary.
    Uses Playwright to search DuckDuckGo and find the most relevant link.
    
    Args:
        deal (dict): The deal dictionary
        
    Returns:
        dict: Updated deal dictionary with a link if found
    """
    # Skip if link already exists
    if deal.get("Link") and deal["Link"] != "" and str(deal["Link"]) != "nan":
        print("Link already exists:", deal["Link"])
        return deal
    
    # Extract key information for search and relevance checking
    company = deal["Company"].strip()
    
    # Get funding amount without $ and M/K
    amount = ""
    if deal.get("Amount") and deal["Amount"] != "":
        amount = deal["Amount"].replace("$", "").replace("M", "").replace("K", "").strip()
    
    # Get funding round
    round_type = ""
    if deal.get("Funding Round") and deal["Funding Round"] != "":
        round_type = deal["Funding Round"].lower()
    
    # Get investors list
    investors = []
    if deal.get("Investors") and deal["Investors"] != "":
        if isinstance(deal["Investors"], list):
            investors = deal["Investors"]
        else:
            investors = [inv.strip() for inv in deal["Investors"].split(",")]
    
    # Build search query with company name and funding keywords
    search_query = f"{company} startup funding {round_type}"
    if investors and len(investors) > 0:
        search_query += f" {investors[0]}"  # Add the first investor to the search

    try:
        link = serpapi_search(deal)
        deal["Link"] = link
        return deal

    except Exception as e:
        print(f"Error with serpapi_search: {e}")
    
    duckduckgo_url = f"https://duckduckgo.com/?q={search_query.replace(' ', '+')}"
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  
            page = browser.new_page()
            
            # Navigate to DuckDuckGo
            page.goto(duckduckgo_url)
            
            # Wait longer for search results to load
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(5000)  # Wait 5 seconds
            
            # Try different selectors to find search results
            results = []
            
            # First try the standard selector
            try:
                if page.locator("article.nrn-react-div").count() > 0:
                    result_elements = page.locator("article.nrn-react-div").all()
                    
                    for element in result_elements[:10]:
                        title_el = element.locator("a[data-testid='result-title-a']").first
                        snippet_el = element.locator("div[data-testid='result-snippet']").first
                        
                        if title_el:
                            title = title_el.text_content()
                            link = title_el.get_attribute("href")
                            snippet = snippet_el.text_content() if snippet_el else ""
                            results.append({
                                "title": title,
                                "link": link,
                                "snippet": snippet
                            })
            except Exception as e:
                pass
            # If no results, try alternative selector
            if not results:
                try:
                    # Try a more general selector for results
                    result_links = page.locator("a[href^='https://']").all()
                    
                    for link_el in result_links[:20]:  # Check more links
                        href = link_el.get_attribute("href")
                        # Skip DuckDuckGo internal links
                        if "duckduckgo.com" in href:
                            continue
                        
                        title = link_el.text_content() or "No title"
                        results.append({
                            "title": title,
                            "link": href,
                            "snippet": ""
                        })
                except Exception as e:
                    pass
            
            # If still no results, try one more approach
            if not results:
                try:
                    # Get all links on the page
                    all_links = page.evaluate("""() => {
                        const links = Array.from(document.querySelectorAll('a[href]'));
                        return links.map(link => {
                            return {
                                href: link.href,
                                text: link.textContent
                            };
                        }).filter(link => 
                            link.href.startsWith('https://') && 
                            !link.href.includes('duckduckgo.com')
                        );
                    }""")
                                    
                    for link_data in all_links[:20]:
                        results.append({
                            "title": link_data["text"] or "No title",
                            "link": link_data["href"],
                            "snippet": ""
                        })
                except Exception as e:
                    pass
            
            browser.close()
                    
            # If we have results, score and rank them
            if results:
                # Define relevant domains for funding news
                relevant_domains = [
                    "techcrunch.com", "venturebeat.com", "businesswire.com", 
                    "prnewswire.com", "forbes.com", "wsj.com", "bloomberg.com", "reuters.com",
                    "fortune.com", "siliconangle.com", "fiercebiotech.com", "fiercehealthcare.com",
                    "mobihealthnews.com", "medcitynews.com", "pymnts.com", "finextra.com",
                    "finsmes.com", "vcnewsdaily.com", "eu-startups.com", "world-nuclear-news.org",
                    "techfundingnews.com", "fusionxinvest.com", "siliconcanals.com"
                ]
                
                # Score each result based on relevance
                scored_results = []
                for result in results:
                    score = 0
                    title = result["title"].lower()
                    snippet = result["snippet"].lower()
                    content = title + " " + snippet
                    link = result["link"].lower()
                    
                    # Check for company name (highest priority)
                    if company.lower() in content or company.lower() in link:
                        score += 10
                    
                    # Check for funding amount
                    if amount and amount in content:
                        score += 5
                    
                    # Check for round type
                    if round_type and round_type in content:
                        score += 3
                    
                    # Check for investors
                    for investor in investors:
                        if investor.lower() in content:
                            score += 2
                            break
                    
                    # Check for funding keywords
                    funding_keywords = ["funding", "raises", "raised", "investment", "announces", "million", "round"]
                    for keyword in funding_keywords:
                        if keyword in content:
                            score += 1
                    
                    # Bonus for relevant domains
                    if any(domain in link for domain in relevant_domains):
                        score += 5
                    
                    scored_results.append((score, result))
                
                # Sort by score (descending)
                scored_results.sort(reverse=True, key=lambda x: x[0])
                
                # Choose the highest scoring result
                if scored_results:
                    best_result = scored_results[0][1]
                    deal["Link"] = best_result["link"]
                else:
                    deal["Link"] = duckduckgo_url
            else:
                deal["Link"] = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
    
    except Exception as e:
        deal["Link"] = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
    
    return deal

def analyze_early_stage_deal(deal):
    """
    Analyze an early-stage deal by finding founders and additional information.
    Then add founders to the search_list database table for further processing.
    
    Args:
        deal (dict): Deal information dictionary with company name, category, etc.
        
    Returns:
        dict: Updated deal with founder information added
    """
    category = deal["Category"]
    
    if category != 'nan':
        # First make sure we have a company name
        if not deal.get("Company"):
            print("No company name found for deal:", deal)
            return deal
            
        company_name = deal["Company"]
        
        try:
            # Import the function to find company founders
            from services.scrapy_spiders import find_company_founders
            import pandas as pd
            from services.database import insert_search_results, create_tables_if_not_exist
            
            print(f"Looking for founders of {company_name}...")
            # Run the search in headless mode to avoid UI disruption
            founders = find_company_founders(company_name, headless=True, wait_time=3)
            
            if founders:
                # Add founder information to the deal
                deal["Founders"] = []
                
                # Prepare data for database insertion
                db_records = []
                
                for founder in founders:
                    # Check if the person is actually a founder (not just a director or other employee)
                    title = founder.get("title", "").lower()
                    if any(keyword in title for keyword in ["founder", "co-founder", "ceo", "chief", "cofounder", "cto"]):
                        name = founder.get("name", "")
                        founder_title = founder.get("title", "")
                        linkedin_url = founder.get("linkedin_url", "")
                        
                        # Add to the deal's founder list
                        founder_info = {
                            "Name": name,
                            "Title": founder_title,
                            "LinkedIn": linkedin_url
                        }
                        deal["Founders"].append(founder_info)
                        
                        # Add to database records if we have a LinkedIn URL
                        if linkedin_url:
                            db_records.append({
                                "name": name,
                                "profile_url": linkedin_url,
                                "title": founder_title,
                                "source": f"Deal tracking - {company_name}",
                                "company_name": company_name,
                                "company_url": ""
                            })
                
                print(f"Found {len(deal['Founders'])} founders for {company_name}")
                
                # Insert records into database if we found any valid founders
                if db_records:
                    try:
                        # Make sure the tables exist
                        create_tables_if_not_exist()
                        
                        # Convert to DataFrame for database insertion
                        df = pd.DataFrame(db_records)
                        
                        # Insert into search_list table
                        insert_result = insert_search_results(df, table_name="search_list", stealth_mode=False)
                        if insert_result:
                            print(f"Successfully added {len(db_records)} founders to search_list database")
                        else:
                            print("Failed to add founders to database")
                    except Exception as db_error:
                        print(f"Error adding founders to database: {str(db_error)}")
            else:
                print(f"No founders found for {company_name}")
                deal["Founders"] = []
                
        except Exception as e:
            print(f"Error finding founders for {company_name}: {str(e)}")
            deal["Founders"] = []
    return deal
