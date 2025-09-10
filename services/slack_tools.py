from typing import Any, Dict, List
from agents import function_tool, WebSearchTool
from services.database import get_db_connection
from services.openai_api import ask_monty
import pandas as pd
import json
import asyncio
import logging

logger = logging.getLogger("slack_tools")

def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting from text for Slack chat display.
    
    Args:
        text: Text that may contain markdown formatting
        
    Returns:
        Text with markdown formatting removed and Slack-friendly formatting
    """
    import re
    
    # Remove **bold** formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove *italic* formatting (but preserve single asterisks that aren't formatting)
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', text)
    
    # Remove `code` formatting
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove ```code blocks``` formatting
    text = re.sub(r'```[\s\S]*?```', lambda m: m.group(0).replace('```', ''), text)
    
    # Remove __underline__ formatting
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Remove _italic_ formatting
    text = re.sub(r'(?<!_)_(?!_)([^_]+)_(?!_)', r'\1', text)
    
    # Clean up markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Clean up markdown links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    return text

@function_tool
async def database_query(query: str, user_id: str = "slack_user") -> str:
    """Query the founder database using natural language.
    
    Args:
        query: The natural language query to execute against the database
        user_id: ID of the user making the query
        
    Returns:
        Formatted query results with interpretation
    """
    try:
        logger.info(f"Tool: database_query | Query: '{query}'")
        # Use LLM to convert natural language to SQL
        sql_prompt = f"""
        You are a SQL expert. Convert this natural language query to a PostgreSQL query for a founder database.
        
        IMPORTANT: Always add "LIMIT 50" to your queries to prevent large result sets. If the user asks for counts, use COUNT(*) instead of SELECT *.
        AVOID using SELECT * and rather only select the columns that are needed for the query.
        
        Available tables and detailed column descriptions:
        
        founders table:
        - profile_url: LinkedIn profile URL of the founder
        - name: Full name of the founder
        - company_name: Name of their startup
        - history: Text entry indicating whether we passed, recommended this one, etc. It will be an empty string or NULL if we haven't seen it yet.
        - founder: Boolean whether the person is a founder or not (true or false, lower case)
        - repeat_founder: Text whether the person is a repeat founder or not (true or false, lower case)
        - technical: Text whether the person is a technical founder or not (true or false, lower case)
        - verticals: Comma-separated list of verticals the founder is working in and has experience in, including model confidence. Example: Blockchain (high), Web3 (high), NFT Development (medium)
        - location: Geographic location (city, state, country)
        - fundingamount: Total funding amount raised (A lot of them are empty strings or NULL, indicating no reported funding)
        - tree_path: Decision tree path for founder classification
        - tree_result: Your final recommendation on the company as investment opportunity
        - tree_justification: Reasoning behind the recommendation
        - past_success_indication_score: Score (0-10) indicating likelihood of past entrepreneurial success
        - startup_experience_score: Score (0-10) measuring startup/entrepreneurial expertise
        - all_experiences: Jsonb field containing ALL work experiences and background
        
        investment_theses table:
        - thesis_title: Title/name of the investment thesis (For example, Fighting Financial Fraud)
        - category: Category or sector of the investment thesis (fintech, healthcare, or commerce)
        - thesis_text: Full text description of the investment thesis
        - keywords: Comma-separated keywords related to the thesis
        
        Examples:
        - "Show me founders from San Francisco" → "SELECT * FROM founders WHERE location ILIKE '%San Francisco%' LIMIT 50;"
        - "How many founders are there?" → "SELECT COUNT(*) FROM founders;"
        - "Find AI founders" → "SELECT * FROM founders WHERE verticals ILIKE '%AI%' ORDER BY past_success_indication_score DESC LIMIT 50;"
        - "What are some problems highlighted in our cross-border payments thesis?" → "SELECT * FROM investment_theses WHERE thesis_title LIKE '%Cross-border payments%';"
        
        Query: {query}
        
        Return only the SQL query, no explanation:
        """
        
        sql_query = ask_monty(sql_prompt, "", max_tokens=200)
        
        # Clean up SQL query - remove markdown code blocks if present
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip()
        
        # Debug: Show the generated SQL query
        logger.info(f"Generated SQL: {sql_query}")
        
        # Ensure LIMIT is present in SELECT queries to prevent large result sets
        if sql_query.upper().startswith('SELECT') and 'COUNT(' not in sql_query.upper() and 'LIMIT' not in sql_query.upper():
            sql_query = sql_query.rstrip(';') + ' LIMIT 50;'
        
        # Execute the query safely
        conn = get_db_connection()
        if not conn:
            return "Database connection failed"
            
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        if not results:
            return "No results found for your query."
        
        # Format results for Slack
        df = pd.DataFrame(results, columns=column_names)
        
        # Debug: Show truncated results
        logger.info(f"Query returned {len(results)} rows")
        
        # Cap the data sent to LLM to prevent token limits
        max_rows_for_llm = min(25, len(results))  # Only send first 25 rows to LLM
        df_truncated = df.head(max_rows_for_llm)
        
        # Use LLM to interpret and format the results
        interpretation_prompt = f"""
            The user asked: "{query}"

            Here are the exact database results (don't alter these):
            {df_truncated.to_string(index=False)}

            Task:
            1. Present the results clearly in bullet points or a table.
            2. Then, provide a **short summary** at the end.
            3. Do NOT change or re-write the raw values (titles, names, funding, etc). 
            """
        
        # Limit the data size sent to LLM
        results_text = df_truncated.to_string(max_rows=25)
        if len(results_text) > 3000:  # Further truncate if still too large
            results_text = results_text[:3000] + "... [truncated]"
        
        interpretation = ask_monty(interpretation_prompt, results_text, max_tokens=300)
        
        conn.close()
        return clean_markdown_formatting(interpretation)
        
    except Exception as e:
        logger.error(f"Database query tool error: {e}")
        return f"❌ Error processing query: {str(e)}"

@function_tool
async def notion_pipeline(query: str, user_id: str = "slack_user") -> str:
    """Get insights about companies we've met that are in Notion Pipeline database.
    
    Args:
        query: Natural language query about pipeline companies
        user_id: ID of the user making the query
        
    Returns:
        Pipeline company insights
    """

    logger.info(f"Tool: notion_pipeline | Query: '{query}'")
    from services.notion import import_pipeline
    
    ID = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
    pipeline = import_pipeline(ID)

    # Debug: Show truncated results
    logger.info(f"Query returned {len(pipeline)} rows")
    
    # Use LLM to interpret and format the results
    interpretation_prompt = f"""
    Interpret these Notion Pipeline database results for a user query: "{query}"
    
    Results:
    {pipeline.to_string(max_rows=100)}
    
    Provide a clear, conversational summary. If there are many results, highlight the most interesting findings.
    """
    
    interpretation = ask_monty(interpretation_prompt, pipeline.to_string(max_rows=100), max_tokens=300)
    
    return clean_markdown_formatting(interpretation)

@function_tool
async def api_profile_info(query: str, user_id: str = "slack_user") -> str:
    """
    Get detailed profile information using the Aviato API enrichment service.
    
    Args:
        query: Natural language query specifying the person to look up
        user_id: ID of the user making the query
        
    Returns:
        Detailed profile information from Aviato API
    """
    try:
        logger.info(f"Tool: api_profile_info | Query: '{query}'")
        from workflows.aviato_processing import enrich_profile
        import re
        
        # Use LLM to extract LinkedIn identifier from natural language
        extraction_prompt = f"""
        Extract LinkedIn information from this query: "{query}"
        
        Look for:
        1. LinkedIn URLs (e.g., https://www.linkedin.com/in/username/)
        2. LinkedIn usernames/IDs (e.g., "matthildur", "john-smith")
        3. Names that could be LinkedIn usernames
        
        Return a JSON object with the extracted information:
        - If you find a LinkedIn URL: {{"linkedin_url": "full_url"}}
        - If you find a LinkedIn username/ID: {{"linkedin_id": "username"}}
        - If you find a name that could be a username: {{"linkedin_id": "name_as_username"}}
        - If no LinkedIn info found: {{}}
        
        Examples:
        - "Tell me about https://www.linkedin.com/in/matthildur-arnadottir/" → {{"linkedin_url": "https://www.linkedin.com/in/matthildur-arnadottir/"}}
        - "What do you know about matthildur?" → {{"linkedin_id": "matthildur"}}
        - "Show me John Smith's profile" → {{"linkedin_id": "john-smith"}}
        - "Tell me about this person" → {{}}
        
        Query: {query}
        
        Return only valid JSON:
        """
        
        criteria_json = ask_monty(extraction_prompt, "", max_tokens=150)
        
        # Debug: Show the criteria extraction
        logger.info(f"API Profile criteria extraction: {criteria_json}")
        
        # Parse the JSON response
        linkedin_info = json.loads(criteria_json)
        
        if not linkedin_info:
            return "Please provide a LinkedIn URL or username to get profile information."
        
        # Extract LinkedIn ID and URL
        linkedin_id = linkedin_info.get('linkedin_id')
        linkedin_url = linkedin_info.get('linkedin_url')
        
        # If we have a URL, extract the ID from it
        if linkedin_url and not linkedin_id:
            # Extract username from LinkedIn URL
            match = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url)
            if match:
                linkedin_id = match.group(1)
        
        if not linkedin_id:
            return "Could not extract LinkedIn identifier from your query. Please provide a LinkedIn URL or username."
        
        # Enrich the profile
        profile_data = enrich_profile(linkedin_id, linkedin_url)
        
        # Debug: Show profile results
        logger.info(f"API Profile results: {str(profile_data)[:500] if profile_data else 'None'}...")
        
        if not profile_data:
            return f"No profile information found for LinkedIn ID: {linkedin_id}"
        
        # Use LLM to interpret and format the results
        interpretation_prompt = f"""
        Interpret this profile data and provide a conversational summary.
        Focus on the most relevant professional information.
        
        Original query was: "{query}"
        
        Format the response as a clear, conversational summary mentioning:
        - Person's name and current title/role
        - Current company and location
        - Professional background and experience
        - Education if available
        - Any other notable information
        
        Keep the response informative but conversational.
        """
        
        interpretation = ask_monty(interpretation_prompt, str(profile_data)[:2000], max_tokens=400)
        
        return clean_markdown_formatting(interpretation)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in API profile info: {e}")
        return f"Error parsing LinkedIn information from query: {query}"
    except Exception as e:
        logger.error(f"Error in API profile info: {e}")
        return f"Error getting profile information: {str(e)}"


@function_tool
async def api_company_info(query: str, user_id: str = "slack_user") -> str:
    """
    Get detailed company information using the Aviato API enrichment service.
    
    Args:
        query: Natural language query specifying the company to look up
        user_id: ID of the user making the query
        
    Returns:
        Detailed company information from Aviato API
    """
    try:
        logger.info(f"Tool: api_company_info | Query: '{query}'")
        from workflows.aviato_processing import enrich_company_raw
        import re
        
        # Use LLM to extract company website from natural language
        extraction_prompt = f"""
        Extract company website information from this query: "{query}"
        
        Look for:
        1. Full website URLs (e.g., https://stripe.com, http://openai.com)
        2. Domain names (e.g., "stripe.com", "openai.com")
        3. Company names that could be converted to websites (e.g., "Stripe" → "stripe.com")
        
        Return a JSON object with the extracted information:
        - If you find a full URL: {{"website": "domain_only"}} (extract just the domain)
        - If you find a domain: {{"website": "domain"}}
        - If you find a company name: {{"website": "companyname.com"}} (convert to likely domain)
        - If no website info found: {{}}
        
        Examples:
        - "Tell me about https://stripe.com" → {{"website": "stripe.com"}}
        - "What do you know about openai.com?" → {{"website": "openai.com"}}
        - "Show me Stripe's info" → {{"website": "stripe.com"}}
        - "Tell me about this company" → {{}}
        
        Query: {query}
        
        Return only valid JSON:
        """
        
        criteria_json = ask_monty(extraction_prompt, "", max_tokens=150)
        
        # Debug: Show the criteria extraction
        logger.info(f"API Company criteria extraction: {criteria_json}")
        
        # Parse the JSON response
        company_info = json.loads(criteria_json)
        
        if not company_info or not company_info.get('website'):
            return "Please provide a company website or name to get company information."
        
        # Extract website
        website = company_info.get('website')
        
        # Clean up the website URL (remove protocol, www, trailing slashes)
        if website:
            website = re.sub(r'^https?://', '', website)
            website = re.sub(r'^www\.', '', website)
            website = website.rstrip('/')
        
        if not website:
            return "Could not extract company website from your query. Please provide a company website or name."
        
        # Enrich the company
        company_data = enrich_company_raw(website)
        
        # Debug: Show company results
        logger.info(f"API Company results: {str(company_data)[:500] if company_data else 'None'}...")
        
        if not company_data:
            return f"No company information found for website: {website}"
        
        # Use LLM to interpret and format the results
        interpretation_prompt = f"""
        Interpret this company data and provide a conversational summary.
        Focus on the most relevant business information.
        
        Original query was: "{query}"
        
        Format the response as a clear, conversational summary mentioning:
        - Company name and tagline/description
        - Industry and business model
        - Funding information (total funding, latest round, valuation)
        - Company size (headcount) and founding date
        - Location and key metrics
        - Any other notable information
        
        Keep the response informative but conversational.
        """
        
        interpretation = ask_monty(interpretation_prompt, str(company_data)[:2000], max_tokens=400)
        
        return clean_markdown_formatting(interpretation)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in API company info: {e}")
        return f"Error parsing company information from query: {query}"
    except Exception as e:
        logger.error(f"Error in API company info: {e}")
        return f"Error getting company information: {str(e)}"

@function_tool
async def get_sector_info(query: str, user_id: str = "slack_user") -> str:
    """
Given a sector or category (Like e-commerce, ai drug discovery, payments), return relevant information from Montage's market taxonomy.
    """
    try:
        logger.info(f"Tool: get_sector_info | Query: '{query}'")
        from .tree_tools import find_nodes_by_name
        import re
        import json

        query_prompt = f"""
        Someone at Montage has a request:
        {query}
        
        You can find the relevant information by parsing Montage's market taxonomy tree. To find relevant information, return a short list of node titles that we should search for.
        Some examples of nodes in the system are: Applied robotics, AI drug discovery, Payments,Healthcare, Commerce, Insurance, etc. in varying levels of abstraction.
        
        Return ONLY a valid JSON array with no additional text:
        ["node1", "node2", "node3"]
        """
        
        response = ask_monty(query_prompt, "", max_tokens=150)
        
        # Parse the LLM response to extract node names
        try:
            # Try to parse as JSON first
            nodes = json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: extract content between brackets using regex
            match = re.search(r'\[(.*?)\]', response)
            if match:
                # Split by comma and clean up quotes
                node_text = match.group(1)
                nodes = [node.strip().strip('"\'') for node in node_text.split(',')]
            else:
                # Last resort: split by common delimiters and clean
                nodes = [node.strip().strip('"\'') for node in re.split(r'[,\n]', response) if node.strip()]
        
        if not nodes:
            return f"Could not extract node names from query: {query}"
        
        logger.info(f"Extracted nodes: {nodes}")
        
        # Load the taste tree
        import json
        import os
        
        # Try multiple possible paths for the taste tree
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'taste_tree.json'),
            os.path.join(os.getcwd(), 'data', 'taste_tree.json'),
            'data/taste_tree.json',
            '/Users/matthildur/Desktop/monty2.0/data/taste_tree.json'
        ]
        
        tree = None
        for tree_path in possible_paths:
            try:
                if os.path.exists(tree_path):
                    with open(tree_path, 'r') as f:
                        tree = json.load(f)
                    logger.info(f"Successfully loaded tree from: {tree_path}")
                    break
            except Exception as e:
                logger.debug(f"Failed to load from {tree_path}: {e}")
                continue
        
        if tree is None:
            return f"Error: Could not load taste_tree.json from any expected location"
        
        info = ""
        for node in nodes:
            node_matches = find_nodes_by_name(tree, node.strip())
            if node_matches:
                for match in node_matches:
                    info += f"\n--- {match['name']} ---\n"
                    info += f"Path: {match['path']}\n"
                    if match.get('investment_status'):
                        info += f"Investment Status: {match['investment_status']}\n"
                    if match.get('interest'):
                        info += f"Interest: {match['interest']}\n"
                    if match.get('recent_news'):
                        info += f"Recent News: {match['recent_news'][:500]}...\n" if len(match['recent_news']) > 500 else f"Recent News: {match['recent_news']}\n"
                    if match.get('portfolio_companies') and match['portfolio_companies'] > 0:
                        info += f"Portfolio Companies: {match['portfolio_companies']}\n"
                    if match.get('thesis'):
                        info += f"Thesis: {match['thesis']}\n"
                    if match.get('caution'):
                        info += f"Caution: {match['caution']}\n"
                    if match.get('montage_lead'):
                        info += f"Montage Lead: {match['montage_lead']}\n"
                    info += "\n"
        
        if not info:
            return f"No information found for any nodes related to query: {query}"
            
        # Use LLM to interpret and format the results
        interpretation_prompt = f"""
        Interpret this data and provide a conversational summary about the sector/category.
        Focus on investment thesis, recent news, portfolio companies, and key insights.
        
        Original query was: "{query}"
        
        Data from Montage's taxonomy:
        {info}

        Keep the response informative but conversational.
        """
        
        interpretation = ask_monty(interpretation_prompt, "", max_tokens=400)
        
        return clean_markdown_formatting(interpretation)
        
    except Exception as e:
        logger.error(f"Error in get_sector_info: {e}")
        return f"Error getting sector information: {str(e)}"

# Export all tools for easy import
MONTY_TOOLS = [
    database_query,
    notion_pipeline,
    api_profile_info,
    api_company_info,
    get_sector_info,
    WebSearchTool()
]
