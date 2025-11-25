#!/usr/bin/env python3
"""
Direct tool testing - test individual tools without the agent
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.slack_tools import MONTY_TOOLS

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_tools")

async def test_database_tool():
    """Test database_query function directly"""
    logger.info("Testing database_query...")
    
    # Get the database_query tool
    database_query_tool = None
    for tool in MONTY_TOOLS:
        if tool.name == "database_query":
            database_query_tool = tool
            break
    
    if not database_query_tool:
        print("ERROR: database_query tool not found")
        return
    
    test_queries = [
        "Show me the top 5 founders with the highest funding amounts",
        "How many technical founders are in the database?",
        "What are the most common verticals?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 30)
        
        try:
            # Use the tool's on_invoke_tool method with proper arguments
            import json
            args = json.dumps({"query": query, "user_id": "test_user"})
            result = await database_query_tool.on_invoke_tool(None, args)
            print("Result:")
            print(result)
            print("-" * 30)
            
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Tool test failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)

async def test_pipeline_tool():
    """Test database_query function directly"""
    logger.info("Testing notion_pipeline...")
    
    # Get the database_query tool
    database_query_tool = None
    for tool in MONTY_TOOLS:
        if tool.name == "notion_pipeline":
            database_query_tool = tool
            break
    
    if not database_query_tool:
        print("ERROR: notion_pipeline tool not found")
        return
    
    test_queries = [
        "What companies are currently in the pipeline?",
        "What sectors are most of the current companies in?",
        "What are some themes of companies we've passed on?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 30)
        
        try:
            # Use the tool's on_invoke_tool method with proper arguments
            import json
            args = json.dumps({"query": query, "user_id": "test_user"})
            result = await database_query_tool.on_invoke_tool(None, args)
            print("Result:")
            print(result)
            print("-" * 30)
            
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Tool test failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)

async def test_api_profile_info_tool():
    """Test api_profile_info function directly"""
    logger.info("Testing api_profile_info...")
    
    # Get the api_profile_info tool
    api_profile_tool = None
    for tool in MONTY_TOOLS:
        if tool.name == "api_profile_info":
            api_profile_tool = tool
            break
    
    if not api_profile_tool:
        print("ERROR: api_profile_info tool not found")
        return
    
    test_queries = [
        "What do you know about https://www.linkedin.com/in/matthildur-arnadottir/"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nAPI Profile Info Test {i}: {query}")
        print("-" * 30)
        
        try:
            # Use the tool's on_invoke_tool method with proper arguments
            import json
            args = json.dumps({"query": query, "user_id": "test_user"})
            result = await api_profile_tool.on_invoke_tool(None, args)
            print("Result:")
            print(result)
            print("-" * 30)
            
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"API profile info test failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)

async def test_api_company_info_tool():
    """Test api_company_info function directly"""
    logger.info("Testing api_company_info...")
    
    # Get the api_company_info tool
    api_company_tool = None
    for tool in MONTY_TOOLS:
        if tool.name == "api_company_info":
            api_company_tool = tool
            break
    
    if not api_company_tool:
        print("ERROR: api_company_info tool not found")
        return
    
    test_queries = [
        "Tell me about stripe.com",
        "What do you know about https://openai.com",
        "Show me Anthropic's company info",
        "Get info for montageventures.com"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nAPI Company Info Test {i}: {query}")
        print("-" * 30)
        
        try:
            # Use the tool's on_invoke_tool method with proper arguments
            import json
            args = json.dumps({"query": query, "user_id": "test_user"})
            result = await api_company_tool.on_invoke_tool(None, args)
            print("Result:")
            print(result)
            print("-" * 30)
            
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"API company info test failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)

async def test_get_sector_info_tool():
    """Test get_sector_info function directly"""
    logger.info("Testing get_sector_info...")
    
    # Get the get_sector_info tool
    sector_info_tool = None
    for tool in MONTY_TOOLS:
        if tool.name == "get_sector_info":
            sector_info_tool = tool
            break
    
    if not sector_info_tool:
        print("ERROR: get_sector_info tool not found")
        return
    
    test_queries = [
        "Tell me about AI infrastructure",
        "What's Montage's view on payments?",
        "Show me information about healthcare AI",
        "What do we know about fintech?",
        "Give me details on robotics investments"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nSector Info Test {i}: {query}")
        print("-" * 30)
        
        try:
            # Use the tool's on_invoke_tool method with proper arguments
            import json
            args = json.dumps({"query": query, "user_id": "test_user"})
            result = await sector_info_tool.on_invoke_tool(None, args)
            print("Result:")
            print(result)
            print("-" * 30)
            
        except Exception as e:
            print(f"ERROR: {e}")
            logger.error(f"Sector info test failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)


async def main():
    """Run direct tool tests"""
    print("🧪 Direct Tool Testing")
    await test_get_sector_info_tool()
    #await test_api_company_info_tool()
    #await test_api_profile_info_tool()
    #await test_pipeline_tool()
    #await test_database_tool()
    #await test_profile_search()
    #await test_deal_analysis()
    #await test_company_insights()
    print("\n✅ Direct tool testing complete!")

if __name__ == "__main__":
    asyncio.run(main())
