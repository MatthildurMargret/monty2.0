#!/usr/bin/env python3
"""
Test script for Monty Agent and Tools
Run this to test the agent functionality locally without Slack
"""

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path to import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import Agent, Runner, trace
from services.slack_tools import MONTY_TOOLS

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("test_agent")

class MontyTester:
    def __init__(self):
        self.agent = Agent(
            name="Monty",
            instructions="""You are Monty, an intelligent assistant for startup and founder data analysis. 
            
            You have access to a database of founders, startups, recent news, and funding deals. 
            You can help users:
            - Search for founders and companies by various criteria
            - Analyze funding deals and investment trends  
            - Query the database with natural language
            - Provide insights about the startup ecosystem
            
            Always be helpful and give specific answers. Users frequently ask for founders and specific companies, so it's important to 
            always give details on people and companies and not abstract away. They want details!
            
            Use the appropriate tool based on the user's request:
            - Use database_query for general data questions and statistics
            """,
            tools=MONTY_TOOLS,
        )
    
    async def test_query(self, query: str, description: str = ""):
        """Test a single query against the agent"""
        print(f"\n{'='*60}")
        print(f"TEST: {description or query}")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)
        
        try:
            with trace(f"Test query: {description}"):
                result = await Runner.run(
                    self.agent,
                    query,
                    context={"test_mode": True}
                )
            
            print("Response:")
            print(result.final_output)
            print("-" * 60)
            return True
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            logger.error(f"Test failed for '{query}': {e}")
            print("-" * 60)
            return False
    
    async def run_test_suite(self):
        """Run a comprehensive test suite"""
        print("🚀 Starting Monty Agent Test Suite")
        print(f"Available tools: {[tool.name for tool in MONTY_TOOLS]}")
        
        test_queries = [
            {
                "query": "Whats the latest news on AI in healthcare?",
                "description": "Company search check"
            }
        ]
        
        passed = 0
        total = len(test_queries)
        
        for test in test_queries:
            success = await self.test_query(test["query"], test["description"])
            if success:
                passed += 1
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {passed}/{total} tests passed")
        print(f"{'='*60}")
        
        return passed == total

async def interactive_mode():
    """Interactive testing mode"""
    tester = MontyTester()
    
    print("\n🤖 Monty Agent Interactive Test Mode")
    print("Type your queries to test the agent. Type 'quit' to exit.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nYour query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue
            
            await tester.test_query(query, "Interactive Query")
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_mode()
    else:
        tester = MontyTester()
        success = await tester.run_test_suite()
        
        if success:
            print("\n✅ All tests passed! Agent is working correctly.")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
