"""
Simple web search and content extraction utilities.
"""

import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse

def extract_article_content(url: str, timeout: int = 10) -> str:
    """
    Extract text content from a web page.
    
    Args:
        url: URL to extract content from
        timeout: Request timeout in seconds
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to avoid overwhelming the analysis
        if len(text) > 5000:
            text = text[:5000] + "..."
        
        return text
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""
