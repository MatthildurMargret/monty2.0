from typing import Any, Dict, List
from agents import function_tool, WebSearchTool
from services.database import get_db_connection
from services.openai_api import ask_monty
import pandas as pd
import json
import asyncio
import logging

logger = logging.getLogger("slack_tools")

def extract_portfolio_from_tree_path(tree, path):
    """
    Extract portfolio companies from a specific tree path.
    
    Args:
        tree: The full taste tree
        path: Path like "Fintech > Insurance & Risk"
        
    Returns:
        List of portfolio company dictionaries
    """
    logger.info(f"🔍 Extracting portfolio from path: {path}")
    path_parts = [part.strip() for part in path.split('>')]
    current_node = tree
    
    # Navigate to the specific node
    for i, part in enumerate(path_parts):
        logger.info(f"🔍 Looking for part '{part}' at level {i}")
        if isinstance(current_node, dict):
            found = False
            # Check both direct keys and children
            for key, value in current_node.items():
                if key.strip() == part and isinstance(value, dict):
                    current_node = value
                    found = True
                    logger.info(f"✅ Found '{part}' as direct key")
                    break
            
            # If not found as direct key, check in children
            if not found and 'children' in current_node:
                children = current_node['children']
                for key, value in children.items():
                    if key.strip() == part and isinstance(value, dict):
                        current_node = value
                        found = True
                        logger.info(f"✅ Found '{part}' in children")
                        break
            
            if not found:
                logger.warning(f"❌ Could not find '{part}' at level {i}")
                logger.info(f"Available keys: {list(current_node.keys())}")
                if 'children' in current_node:
                    logger.info(f"Available children: {list(current_node['children'].keys())}")
                return []
        else:
            logger.warning(f"❌ Current node is not dict at level {i}")
            return []
    
    # Extract portfolio from meta
    if isinstance(current_node, dict) and 'meta' in current_node:
        portfolio = current_node['meta'].get('portfolio', [])
        logger.info(f"📋 Found portfolio with {len(portfolio)} companies")
        if isinstance(portfolio, list):
            return portfolio
    else:
        logger.warning(f"❌ No meta found in final node")
        if isinstance(current_node, dict):
            logger.info(f"Available keys in final node: {list(current_node.keys())}")
    
    return []

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
async def early_stage_founder_query(query: str, user_id: str = "slack_user") -> str:
    """Query the founder database using natural language.
    
    Args:
        query: The natural language query to execute against the database
        user_id: ID of the user making the query
        
    Returns:
        Formatted query results with interpretation
    """
    try:
        logger.info(f"Tool: early_stage_founder_query | Query: '{query}'")
        # Use LLM to convert natural language to SQL
        sql_prompt = f"""
        You are a SQL expert. Convert this natural language query to a PostgreSQL query for Montage's early stage founder database.
        
        IMPORTANT: Always add "LIMIT 50" to your queries to prevent large result sets. If the user asks for counts, use COUNT(*) instead of SELECT *.
        AVOID using SELECT * and rather only select the columns that are needed for the query.
        
        Available tables and detailed column descriptions:
        
        founders table:
        - profile_url: LinkedIn profile URL of the founder
        - name: Full name of the founder
        - company_name: Name of their startup
        - description_1: Description of the startup
        - product: Description of the product the startup offers
        - market: Description of the market the startup sells into or operates in.
        - history: Text entry indicating whether we passed, recommended this one, etc. It will be an empty string or NULL if we haven't seen it yet.
        - founder: Boolean whether the person is a founder or not (true or false, lower case)
        - repeat_founder: Text whether the person is a repeat founder or not (true or false, lower case)
        - technical: Text whether the person is a technical founder or not (true or false, lower case)
        - location: Geographic location (city, state, country)
        - fundingamount: Total funding amount raised (A lot of them are empty strings or NULL, indicating no reported funding)
        - tree_path: Decision tree path for founder classification
        - tree_result: Your final recommendation on the company as investment opportunity
        - tree_justification: Reasoning behind the recommendation
        - past_success_indication_score: Score (0-10) indicating likelihood of past entrepreneurial success
        - startup_experience_score: Score (0-10) measuring startup/entrepreneurial expertise
        - all_experiences: Jsonb field containing ALL work experiences and background
        - company_website: Website of the company
        
        Examples:
        - "Show me founders from San Francisco" → "SELECT * FROM founders WHERE location LIKE '%San Francisco%' LIMIT 50;"
        - "How many founders are there?" → "SELECT COUNT(*) FROM founders;"
        - "Find AI founders" → "SELECT * FROM founders WHERE tree_path LIKE '%AI%' ORDER BY past_success_indication_score DESC LIMIT 50;"
        - "I'm looking for founders that are building in stablecoins" → "SELECT * FROM founders WHERE (tree_path LIKE '%stablecoins%' OR product LIKE '%stablecoins%') ORDER BY past_success_indication_score DESC LIMIT 50;"
        
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
            2. Include as much detail as exists on the results. If user asked for founders, give their profile urls, website urls, funding, location, and brief description.
            3. Do NOT change or re-write the raw values (titles, names, funding, etc). 
            """
        
        # Limit the data size sent to LLM
        results_text = df_truncated.to_string(max_rows=25)
        if len(results_text) > 3000:  # Further truncate if still too large
            results_text = results_text[:3000] + "... [truncated]"
        
        interpretation = ask_monty(interpretation_prompt, results_text, max_tokens=600)
        
        conn.close()
        return clean_markdown_formatting(interpretation)
        
    except Exception as e:
        logger.error(f"Database query tool error: {e}")
        return f"❌ Error processing query: {str(e)}"
    
@function_tool
async def investment_theme_query(query: str, user_id: str = "slack_user") -> str:
    """Query the investment themes database using natural language.
    
    Args:
        query: The natural language query to execute against the database
        user_id: ID of the user making the query
        
    Returns:
        Formatted query results with interpretation
    """
    try:
        logger.info(f"Tool: investment_theme_query | Query: '{query}'")
        # Use LLM to convert natural language to SQL
        sql_prompt = f"""
        You are a SQL expert. Convert this natural language query to a PostgreSQL query for Montage's investment themes database.
        
        Available tables and detailed column descriptions:
    
        investment_theses table:
        - thesis_title: Title/name of the investment thesis (For example, Fighting Financial Fraud)
        - category: Category or sector of the investment thesis (fintech, healthcare, or commerce)
        - thesis_text: Full text description of the investment thesis
        - keywords: Comma-separated keywords related to the thesis
        
        Examples:
        - "What are some problems highlighted in our cross-border payments thesis?" → "SELECT * FROM investment_theses WHERE thesis_title LIKE '%Cross-border payments%';"
        - "What fintech investment themes do we have outlined?" → "SELECT thesis_title FROM investment_theses WHERE category LIKE '%fintech%';"
        - "What are some keywords associated with our AI for drug discovery thesis?" → "SELECT keywords FROM investment_theses WHERE thesis_title LIKE '%drug discovery%';"
        
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
            2. Then, provide a short summary at the end.
            3. Do NOT change or re-write the raw values (titles, keywords, etc.). 
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

    logger.info(f"Pipeline contains {len(pipeline)} total companies")
    
    # Smart chunking approach for 200-500 row datasets
    # Calculate optimal chunk size based on estimated token usage
    
    # Estimate tokens per row (rough approximation)
    sample_row = pipeline.iloc[0:1].to_string() if len(pipeline) > 0 else ""
    estimated_tokens_per_row = len(sample_row.split()) * 1.3  # rough token estimate
    
    # Target ~1500 tokens per chunk to stay well under limits
    target_tokens_per_chunk = 1500
    optimal_chunk_size = max(10, min(100, int(target_tokens_per_chunk / estimated_tokens_per_row)))
    
    logger.info(f"Using chunk size of {optimal_chunk_size} rows (estimated {estimated_tokens_per_row:.1f} tokens per row)")
    
    # Create chunks with the calculated size
    chunks = [pipeline[i:i + optimal_chunk_size] for i in range(0, len(pipeline), optimal_chunk_size)]
    
    # First pass: Analyze each chunk and create summaries
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"""
        Analyze pipeline chunk {i+1}/{len(chunks)} for query: "{query}"
        
        Data ({len(chunk)} companies):
        {chunk.to_string()}
        
        Extract key insights and patterns relevant to the query. Focus on:
        - Companies that match the query criteria
        - Notable trends or patterns
        - Key data points
        
        Be specific and factual.
        """
        
        summary = ask_monty(chunk_prompt, chunk.to_string(), max_tokens=250)
        if summary.strip():
            chunk_summaries.append(f"Chunk {i+1} ({len(chunk)} companies): {summary.strip()}")
    
    # Second pass: Create comprehensive analysis
    # If we have many chunks, do a two-tier synthesis
    if len(chunk_summaries) > 8:
        # Group summaries and create intermediate summaries
        summary_groups = [chunk_summaries[i:i + 4] for i in range(0, len(chunk_summaries), 4)]
        intermediate_summaries = []
        
        for i, group in enumerate(summary_groups):
            group_prompt = f"""
            Synthesize these pipeline analysis summaries for query: "{query}"
            
            Summaries:
            {chr(10).join(group)}
            
            Create a consolidated summary of key findings from this group.
            """
            
            intermediate = ask_monty(group_prompt, "", max_tokens=200)
            if intermediate.strip():
                intermediate_summaries.append(f"Group {i+1}: {intermediate.strip()}")
        
        # Final synthesis from intermediate summaries
        final_prompt = f"""
        Provide final analysis for query: "{query}" based on complete pipeline review.
        
        Dataset: {len(pipeline)} total companies across {len(chunks)} chunks
        
        Consolidated findings:
        {chr(10).join(intermediate_summaries)}
        
        Provide a comprehensive answer to the user's query with specific insights and data.
        """
        
        final_analysis = ask_monty(final_prompt, "", max_tokens=500)
    else:
        # Direct synthesis for smaller number of chunks
        final_prompt = f"""
        Provide comprehensive analysis for query: "{query}" based on complete pipeline review.
        
        Dataset: {len(pipeline)} total companies analyzed across {len(chunks)} chunks
        
        Detailed findings:
        {chr(10).join(chunk_summaries)}
        
        Synthesize these findings into a clear, comprehensive answer to the user's query.
        """
        
        final_analysis = ask_monty(final_prompt, "", max_tokens=500)
    
    return clean_markdown_formatting(final_analysis)

@function_tool
async def get_all_portfolio(query: str, user_id: str = "slack_user") -> str:
    """Get the list of companies we've invested in.
    
    Args:
        query: Natural language query about portfolio companies
        user_id: ID of the user making the query
        
    Returns:
        Portfolio company overview
    """

    logger.info(f"Tool: get_all_portfolio | Query: '{query}'")
    portfolio = pd.read_csv("data/portfolio.csv")

    # Use LLM to interpret and format the results
    interpretation_prompt = f"""
    Interpret these portfolio database results for a user query: "{query}"
    
    Results:
    {portfolio.to_string(max_rows=100)}
    
    Provide a clear, conversational summary. 
    """
    
    interpretation = ask_monty(interpretation_prompt, portfolio.to_string(max_rows=100), max_tokens=500)
    
    logger.info(f"📤 TOOL OUTPUT: get_all_portfolio returned {len(portfolio)} companies: {interpretation[:100]}...")
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
        
        You need to determine:
        1. Which node titles to search for in the market taxonomy tree
        2. Which metadata fields are relevant to extract from those nodes
        
        For node titles, think about common business/industry terms that would appear in a taxonomy:
        - "AI for accounting" should search for "Accounting", "Back-Office Automation", "Financial Technology"
        - "robotics" should search for "Robotics", "Automation" 
        - "insurance" should search for "Insurance", "Risk"
        - Use both specific terms and broader category names
        
        Available metadata fields: 
        portfolio: The companies we have in our portfolio relevant to this node
        recent_news: Recent funding announcements of companies relevant to this node
        interest: Our thoughts and recent IC discussions around this space
        investment_status: High, Low, Medium, Neutral, Exclude (indicates our level of interest in this space)
        caution: Any points of caution or red flags we have around this space
        montage_lead: Who on the team is most interested in this space
        thesis: If we have a particular investment thesis around this space
        description: A short description of the space
        
        Examples:
        - If asking about portfolio companies: extract "portfolio" field
        - If asking about recent news: extract "recent_news" field  
        - If asking about investment thesis: extract "thesis" and "interest" fields
        - If asking general info: extract multiple relevant fields (definitely thesis and interest)
        
        Return ONLY a valid JSON object:
        {{
            "nodes": ["node1", "node2", "node3"],
            "fields": ["field1", "field2"]
        }}
        """
        
        response = ask_monty(query_prompt, "", max_tokens=150)
        
        # Parse the LLM response to extract nodes and fields
        try:
            # Try to parse as JSON first
            query_config = json.loads(response.strip())
            nodes = query_config.get('nodes', [])
            fields = query_config.get('fields', ['portfolio', 'recent_news', 'interest'])  # default fields
        except json.JSONDecodeError:
            # Fallback: assume it's just nodes and use default fields
            match = re.search(r'\[(.*?)\]', response)
            if match:
                node_text = match.group(1)
                nodes = [node.strip().strip('"\'') for node in node_text.split(',')]
            else:
                nodes = [node.strip().strip('"\'') for node in re.split(r'[,\n]', response) if node.strip()]
            fields = ['portfolio', 'recent_news', 'interest']  # default fields
        
        if not nodes:
            return f"Could not extract node names from query: {query}"
        
        logger.info(f"Extracted nodes: {nodes}")
        logger.info(f"Extracting fields: {fields}")
        
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
            logger.info(f"🔍 Searching for node '{node}' - found {len(node_matches)} matches")
            if node_matches:
                for match in node_matches:
                    info += f"\n--- {match['name']} ---\n"
                    info += f"Path: {match['path']}\n"
                    
                    # Only extract requested fields
                    for field in fields:
                        if field == 'portfolio':
                            # Extract portfolio data from the tree node
                            logger.info(f"🏢 Attempting to extract portfolio from path: {match['path']}")
                            portfolio_companies = extract_portfolio_from_tree_path(tree, match['path'])
                            logger.info(f"🏢 Found {len(portfolio_companies)} portfolio companies")
                            if portfolio_companies:
                                info += f"Portfolio Companies:\n"
                                for company in portfolio_companies:
                                    info += f"  - {company.get('company_name', 'Unknown')}: {company.get('brief_description', '')}\n"
                                    info += f"    Status: {company.get('status', 'N/A')}, Stage: {company.get('stage', 'N/A')}\n"
                                    info += f"    Lead: {company.get('montage_lead', 'N/A')}, Fund: {company.get('fund', 'N/A')}\n"
                            else:
                                logger.warning(f"⚠️ No portfolio companies found for path: {match['path']}")
                        elif field == 'recent_news' and match.get('recent_news'):
                            info += f"Recent News: {match['recent_news'][:500]}...\n" if len(match['recent_news']) > 500 else f"Recent News: {match['recent_news']}\n"
                        elif field == 'interest' and match.get('interest'):
                            info += f"Interest: {match['interest']}\n"
                        elif field == 'investment_status' and match.get('investment_status'):
                            info += f"Investment Status: {match['investment_status']}\n"
                        elif field == 'thesis' and match.get('thesis'):
                            info += f"Thesis: {match['thesis']}\n"
                        elif field == 'caution' and match.get('caution'):
                            info += f"Caution: {match['caution']}\n"
                        elif field == 'montage_lead' and match.get('montage_lead'):
                            info += f"Montage Lead: {match['montage_lead']}\n"
                        elif field == 'description' and match.get('description'):
                            info += f"Description: {match['description']}\n"
                    info += "\n"
        
        print("Information gathered:")
        print(info)
        logger.info(f"📋 Collected info length: {len(info)} characters")
        logger.info(f"📋 Info preview: {info}...")
        
        if not info:
            logger.warning("⚠️ No information found for any nodes")
            return f"No information found for any nodes related to query: {query}"
            
        # Use LLM to interpret and format the results
        interpretation_prompt = f"""
        You are Monty, Montage's AI assistant. Interpret the provided taxonomy data and answer the user's query directly.
        If the user asked for portfolio companies, list all the companies shown in the portfolio data.
        If the user asked for recent news, provide the news information.
        If the user asked about general interest in certain areas, it's relevant to pull interest, portfolio, investment_status and especially thesis, which indicates whether we have published a thesis on the space.
        If we have published a thesis on the space, it's definitely of high interest.
        If the user asks about AI for X, you should look for nodes that mention X (not necessarily including the AI part)
        
        Keep the response informative but conversational.
        """
        
        user_data = f"""
        User query: "{query}"
        
        Data from Montage's taxonomy:
        {info}
        """
        
        interpretation = ask_monty(interpretation_prompt, user_data, max_tokens=400)
        
        logger.info(f"📤 TOOL OUTPUT: get_sector_info found {len([n for n in nodes if n])} nodes, returning: {interpretation[:100]}...")
        return clean_markdown_formatting(interpretation)
        
    except Exception as e:
        logger.error(f"Error in get_sector_info: {e}")
        return f"Error getting sector information: {str(e)}"

# Export all tools for easy import
MONTY_TOOLS = [
    #investment_theme_query,
    #early_stage_founder_query,
    #notion_pipeline,
    #get_all_portfolio,
    get_sector_info,
    WebSearchTool()
]
