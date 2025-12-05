"""
Parallel API Client - Wrapper to match the Parallel SDK interface
"""
import os
import requests
from typing import List, Dict, Any, Optional


class Parallel:
    """Wrapper class that mimics the Parallel SDK interface"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.parallel.ai"
        self.beta = self.Beta(self)
    
    class Beta:
        """Beta API methods"""
        
        def __init__(self, parent):
            self.parent = parent
        
        def search(
            self, 
            mode: str = "one-shot", 
            search_queries: Optional[List[str]] = None, 
            max_results: int = 10,
            objective: Optional[str] = None,
            max_chars_per_result: int = 10000
        ) -> Dict[str, Any]:
            """
            Search using Parallel API
            
            Args:
                mode: Search mode (e.g., "one-shot")
                search_queries: List of search query strings (can be None)
                max_results: Maximum number of results to return
                objective: Optional objective string for the search
                max_chars_per_result: Maximum characters per result excerpt (default: 10000)
                
            Returns:
                Dictionary with search results
            """
            # Use the correct API endpoint based on working example
            api_url = "https://api.parallel.ai/v1beta/search"
            
            # Headers match the working format
            headers = {
                "x-api-key": self.parent.api_key,
                "Content-Type": "application/json",
                "parallel-beta": "search-extract-2025-10-10"
            }
            
            # Build payload - can use either search_queries or objective
            payload = {
                "mode": mode,
                "max_results": max_results,
                "excerpts": {
                    "max_chars_per_result": max_chars_per_result
                }
            }
            
            # Add search_queries if provided, otherwise use None
            if search_queries:
                payload["search_queries"] = search_queries
            else:
                payload["search_queries"] = None
            
            # Add objective if provided
            if objective:
                payload["objective"] = objective
            
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(
                        f"Parallel API request failed with status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
            except requests.exceptions.Timeout:
                raise Exception("Parallel API request timed out")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Parallel API request failed: {str(e)}")
        
        def extract(
            self,
            urls: List[str],
            objective: Optional[str] = None,
            excerpts: bool = True,
            full_content: bool = False
        ) -> Any:
            """
            Extract information from URLs using Parallel API extract endpoint.
            
            Args:
                urls: List of URLs to extract information from
                objective: Objective/question to extract information for
                excerpts: Whether to return excerpts (default: True)
                full_content: Whether to return full content (default: False)
                
            Returns:
                Object with results, errors, and usage information
            """
            api_url = "https://api.parallel.ai/v1beta/extract"
            
            headers = {
                "x-api-key": self.parent.api_key,
                "Content-Type": "application/json",
                "parallel-beta": "search-extract-2025-10-10"
            }
            
            payload = {
                "urls": urls,
                "excerpts": excerpts,
                "full_content": full_content
            }
            
            if objective:
                payload["objective"] = objective
            
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    # Return an object that mimics the Parallel SDK response structure
                    response_data = response.json()
                    return ExtractResponse(response_data)
                else:
                    raise Exception(
                        f"Parallel API extract request failed with status {response.status_code}: "
                        f"{response.text[:200]}"
                    )
            except requests.exceptions.Timeout:
                raise Exception("Parallel API extract request timed out")
            except requests.exceptions.RequestException as e:
                raise Exception(f"Parallel API extract request failed: {str(e)}")


class ExtractResponse:
    """Response object that mimics the Parallel SDK extract response structure"""
    
    def __init__(self, response_data: Dict[str, Any]):
        self._data = response_data
        self.extract_id = response_data.get("extract_id", "")
        self.results = response_data.get("results", [])
        self.errors = response_data.get("errors", [])
        self.warnings = response_data.get("warnings")
        self.usage = response_data.get("usage", [])

