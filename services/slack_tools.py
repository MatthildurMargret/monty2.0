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
    """Remove markdown formatting (bold/italic) from text while preserving content asterisks.
    
    Args:
        text: Text that may contain markdown formatting
        
    Returns:
        Text with markdown formatting removed
    """
    import re
    # Remove **bold** formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove *italic* formatting
    text = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', text)
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
        - thesis_title: Title/name of the investment thesis
        - category: Category or sector of the investment thesis
        - thesis_text: Full text description of the investment thesis
        - keywords: Comma-separated keywords related to the thesis
        
        Examples:
        - "Show me founders from San Francisco" → "SELECT * FROM founders WHERE location ILIKE '%San Francisco%' LIMIT 50;"
        - "How many founders are there?" → "SELECT COUNT(*) FROM founders;"
        - "Find AI founders" → "SELECT * FROM founders WHERE verticals ILIKE '%AI%' ORDER BY past_success_indication_score DESC LIMIT 50;"
        
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
        logger.info(f"Generated SQL Query: {sql_query}")
        
        # Ensure LIMIT is present in SELECT queries to prevent large result sets
        if sql_query.upper().startswith('SELECT') and 'COUNT(' not in sql_query.upper() and 'LIMIT' not in sql_query.upper():
            sql_query = sql_query.rstrip(';') + ' LIMIT 50;'
        
        # Execute the query safely
        conn = get_db_connection()
        if not conn:
            return "❌ Database connection failed"
            
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
        Interpret these database results for a user query: "{query}"
        
        Results (showing {max_rows_for_llm} of {len(results)} total rows):
        {df_truncated.to_string(max_rows=25)}
        
        Provide a clear, conversational summary. If showing a subset of results, mention the total count and highlight key patterns or interesting findings.
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
async def profile_search(query: str, user_id: str = "slack_user") -> str:
    """Search for founder profiles and companies based on criteria.
    
    Args:
        query: Natural language search query for profiles (location, industry, role, company type, etc.)
        user_id: ID of the user making the search
        
    Returns:
        Formatted search results with profile information
    """
    try:
        conn = get_db_connection()
        if not conn:
            return "❌ Database connection failed"
        
        # Extract search criteria from natural language
        search_prompt = f"""
        Extract search criteria from this query: "{query}"
        
        Return a JSON object with these possible fields (only include if mentioned):
        - "company": company name or type
        - "location": city, state, or country  
        - "role": job title or role
        - "funding": funding stage or amount
        - "industry": industry or sector
        
        Example: {{"company": "fintech", "location": "San Francisco", "role": "CEO"}}
        """
        
        criteria_json = ask_monty(search_prompt, "", max_tokens=150)
        
        # Debug: Show the criteria extraction
        logger.info(f"Profile Search criteria extraction: {criteria_json}")
        
        criteria = json.loads(criteria_json)
        
        # Build SQL query based on criteria
        where_conditions = []
        params = []
        
        if criteria.get("company"):
            where_conditions.append("company_name ILIKE %s")
            params.append(f"%{criteria['company']}%")
            
        if criteria.get("location"):
            where_conditions.append("location ILIKE %s")
            params.append(f"%{criteria['location']}%")
            
        if criteria.get("role"):
            where_conditions.append("headline ILIKE %s")
            params.append(f"%{criteria['role']}%")
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        sql_query = f"""
        SELECT name, company_name, location, funding
        FROM founders 
        WHERE {where_clause}
        ORDER BY funding DESC NULLS LAST
        LIMIT 10
        """
        
        cursor = conn.cursor()
        cursor.execute(sql_query, params)
        results = cursor.fetchall()
        
        # Debug: Show query results
        logger.info(f"Profile Search SQL: {sql_query}")
        logger.info(f"Profile Search returned {len(results)} rows")
        if results:
            logger.info(f"Sample results (first 3 rows): {results[:3]}")
        
        if not results:
            return f"🔍 No profiles found matching: {query}"
        
        # Format results
        response = f"🚀 **Found {len(results)} profiles matching '{query}':**\n\n"
        
        for i, (name, company, location, funding) in enumerate(results[:5], 1):
            funding_info = f" | ${funding:,.0f}" if funding else ""
            
            response += f"{i}. **{name}**\n"
            response += f"   Company: {company or 'Unknown'}\n"
            response += f"   📍 {location}{funding_info}\n\n"
        
        conn.close()
        return response
        
    except Exception as e:
        logger.error(f"Profile search tool error: {e}")
        return f"❌ Error searching profiles: {str(e)}"

@function_tool
async def deal_analysis(query: str, user_id: str = "slack_user") -> str:
    """Analyze funding deals, investment rounds, and financial data.
    
    Args:
        query: Natural language query about funding deals, investment trends, or financial analysis
        user_id: ID of the user making the query
        
    Returns:
        Formatted analysis of funding deals and investment data
    """
    try:
        conn = get_db_connection()
        if not conn:
            return "❌ Database connection failed"
        
        # Analyze funding data
        sql_query = """
        SELECT 
            'Funding' as deal_type,
            AVG(funding) as avg_funding,
            COUNT(*) as deal_count
        FROM founders 
        WHERE funding IS NOT NULL 
        GROUP BY 'Funding'
        ORDER BY avg_funding DESC
        """
        
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        # Debug: Show query results
        logger.info(f"Deal Analysis SQL: {sql_query}")
        logger.info(f"Deal Analysis returned {len(results)} rows")
        if results:
            logger.info(f"Sample results: {results[:3]}")
        
        if not results:
            return "📈 No funding data available"
        
        response = "💰 **Funding Analysis:**\n\n"
        
        for deal_type, avg_funding, count in results:
            response += f"**{deal_type}:**\n"
            response += f"  • Average: ${avg_funding:,.0f}\n"
            response += f"  • Count: {count} deals\n\n"
        
        # Add recent high-value deals
        recent_deals_query = """
        SELECT name, company_name, funding
        FROM founders 
        WHERE funding > 1000000
        ORDER BY funding DESC
        LIMIT 5
        """
        
        cursor.execute(recent_deals_query)
        recent_results = cursor.fetchall()
        
        if recent_results:
            response += "🔥 **Top Funded Companies:**\n"
            for name, company, amount in recent_results:
                response += f"• **{name}** ({company}): ${amount:,.0f}\n"
        
        conn.close()
        return response
        
    except Exception as e:
        logger.error(f"Deal analysis tool error: {e}")
        return f"❌ Error analyzing deals: {str(e)}"

@function_tool
async def company_insights(query: str, user_id: str = "slack_user") -> str:
    """Get insights about companies, market trends, and industry analysis.
    
    Args:
        query: Natural language query about companies, market trends, or strategic analysis
        user_id: ID of the user making the query
        
    Returns:
        Strategic insights and market analysis with actionable recommendations
    """
    try:
        # Enhanced insights using database query with strategic context
        enhanced_query = f"Provide strategic insights and analysis about: {query}"
        
        # Use the same logic as DatabaseQueryTool but with strategic focus
        sql_prompt = f"""
        You are a SQL expert. Convert this strategic analysis query to a PostgreSQL query for a founder/startup database.
        Focus on trends, patterns, and insights rather than just raw data.
        
        Available tables and key columns:
        - founders: profile_url, name, company_name, founder, repeat_founder, technical, verticals, location, funding, tree_path, tree_result, tree_justification, past_success_indication_score, startup_expertise_score, allexperiences
        - investment_theses: thesis_title, category, thesis_text, keywords
        
        Query: {enhanced_query}
        
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
        logger.info(f"Company Insights SQL Query: {sql_query}")
        
        conn = get_db_connection()
        if not conn:
            return "❌ Database connection failed"
            
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
        
        if not results:
            return "No data available for this analysis."
        
        df = pd.DataFrame(results, columns=column_names)
        
        # Debug: Show truncated results
        logger.info(f"Company Insights returned {len(results)} rows")
        logger.info(f"Sample results (first 3 rows): {df.head(3).to_dict('records')}")
        
        # Strategic interpretation focused on insights and trends
        interpretation_prompt = f"""
        Provide strategic insights and market analysis for: "{query}"
        
        Data:
        {df.to_string(max_rows=15)}
        
        Focus on:
        - Market trends and patterns
        - Strategic opportunities
        - Key insights for decision making
        - Actionable recommendations
        
        Format as a strategic brief with clear insights and recommendations.
        """
        
        interpretation = ask_monty(interpretation_prompt, df.to_string(max_rows=15), max_tokens=400)
        
        conn.close()
        return f"🎯 **Strategic Insights:**\n{interpretation}"
        
    except Exception as e:
        logger.error(f"Company insights tool error: {e}")
        return f"❌ Error generating insights: {str(e)}"

@function_tool
async def notion_pipeline(query: str, user_id: str = "slack_user") -> str:
    """Get insights about companies we've met that are in Notion Pipeline database.
    
    Args:
        query: Natural language query about companies, past interactions, or strategic analysis
        user_id: ID of the user making the query
        
    Returns:
        Information from Notion Pipeline database
    """
    from services.notion import import_pipeline
    try:
        ID = "15e30f29-5556-4fe1-89f6-76d477a79bf8"
        pipeline = import_pipeline(ID)

        # Debug: Show truncated results
        logger.info(f"Query returned {len(pipeline)} rows")
        
        # Use LLM to interpret and format the results
        interpretation_prompt = f"""
        Interpret these database results for a user query: "{query}"
        
        Results:
        {pipeline.to_string(max_rows=100)}
        
        Provide a clear, conversational summary. If there are many results, highlight the most interesting findings.
        """
        
        interpretation = ask_monty(interpretation_prompt, pipeline.to_string(max_rows=100), max_tokens=300)
        
        return clean_markdown_formatting(interpretation)
        
    except Exception as e:
        logger.error(f"Company insights tool error: {e}")
        return f"❌ Error generating insights: {str(e)}"

@function_tool
async def api_profile_info(query: str, user_id: str = "slack_user") -> str:
    """
    Get detailed profile information using the Aviato API enrichment service.
    
    Args:
        query: Natural language query containing LinkedIn ID or URL (e.g., "Tell me about matthildur", "What do you know about https://www.linkedin.com/in/matthildur-arnadottir/")
        user_id: User ID for logging purposes
        
    Returns:
        Formatted profile information from Aviato API
    """
    from workflows.aviato_processing import enrich_profile
    import re
    
    try:
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
        query: Natural language query containing company website or name (e.g., "Tell me about stripe.com", "What do you know about https://openai.com")
        user_id: User ID for logging purposes
        
    Returns:
        Formatted company information from Aviato API
    """
    from workflows.aviato_processing import enrich_company_raw
    import re
    
    try:
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


# Export all tools for easy import
MONTY_TOOLS = [
    database_query,
    notion_pipeline,
    api_profile_info,
    api_company_info,
    WebSearchTool()
]
