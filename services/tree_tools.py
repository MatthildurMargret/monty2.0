from typing import List, Dict, Any, Optional
import re
from datetime import datetime

def find_nodes_by_name(root_node, name):
    """
    Search for nodes by name in the taste tree and return matches with their paths.
    
    Args:
        root_node: The root of the taste tree (dict)
        name: The name to search for (case-insensitive)
        
    Returns:
        List of matches with node summary and path
    """
    target = name.strip().lower()
    matches = []
    
    def _walk(node, path=[]):
        # Check if this is a dict with children
        if isinstance(node, dict):
            # Check top-level keys (like "Commerce", "Health", etc.)
            for key, value in node.items():
                if key != "meta" and isinstance(value, dict):
                    current_path = path + [key]
                    if target in key.strip().lower():
                        # Create a summary instead of returning the full node
                        summary = {
                            "name": key,
                            "path": " > ".join(current_path),
                            "has_children": len(value.get("children", {})) > 0,
                            "child_count": len(value.get("children", {})),
                            "meta_fields": list(value.get("meta", {}).keys()) if "meta" in value else []
                        }
                        
                        # Add key meta info if available
                        if "meta" in value:
                            meta = value["meta"]
                            summary["interest"] = meta.get("interest", "")[:100] + "..." if len(meta.get("interest", "")) > 100 else meta.get("interest", "")
                            summary["investment_status"] = meta.get("investment_status", "")
                            summary["portfolio_companies"] = len([p for p in meta.get("portfolio", []) if isinstance(p, dict) and "company_name" in p])
                            summary["recent_news"] = meta.get("recent_news", "")
                            summary["caution"] = meta.get("caution", "")
                            summary["montage_lead"] = meta.get("montage_lead", "")
                            summary["last_updated"] = meta.get("last_updated", "")
                            summary["description"] = meta.get("description", "")
                            summary["thesis"] = meta.get("thesis", "")
                        
                        matches.append(summary)
                    # Recursively search children
                    if "children" in value:
                        _walk(value["children"], current_path)
    
    _walk(root_node)
    return matches

def find_companies_in_node(node):
    """
    Extract all companies from a node's portfolio.
    
    Args:
        node: A tree node (dict) that may contain portfolio companies
        
    Returns:
        List of company dictionaries
    """
    companies = []
    if isinstance(node, dict) and "meta" in node:
        portfolio = node["meta"].get("portfolio", [])
        for item in portfolio:
            if isinstance(item, dict) and "company_name" in item:
                companies.append(item)
    return companies

def search_tree_content(root_node, search_term):
    """
    Search for content within the tree (descriptions, interests, etc.)
    
    Args:
        root_node: The root of the taste tree
        search_term: Term to search for (case-insensitive)
        
    Returns:
        List of matches with context
    """
    search_term = search_term.strip().lower()
    matches = []
    
    def _search_node(node, path=[]):
        if isinstance(node, dict):
            for key, value in node.items():
                if key != "meta" and isinstance(value, dict):
                    current_path = path + [key]
                    
                    # Search in meta content
                    if "meta" in value:
                        meta = value["meta"]
                        for field, content in meta.items():
                            if isinstance(content, str) and search_term in content.lower():
                                matches.append({
                                    "path": " > ".join(current_path),
                                    "field": field,
                                    "content": content[:200] + "..." if len(content) > 200 else content
                                })
                    
                    # Recursively search children
                    if "children" in value:
                        _search_node(value["children"], current_path)
    
    _search_node(root_node)
    return matches

def list_all_nodes(root_node):
    """
    List all nodes in the tree with their paths.
    
    Args:
        root_node: The root of the taste tree
        
    Returns:
        List of all nodes with their paths
    """
    all_nodes = []
    
    def _walk(node, path=[]):
        if isinstance(node, dict):
            for key, value in node.items():
                if key != "meta" and isinstance(value, dict):
                    current_path = path + [key]
                    all_nodes.append({
                        "name": key,
                        "path": " > ".join(current_path),
                        "has_children": len(value.get("children", {})) > 0,
                        "child_count": len(value.get("children", {}))
                    })
                    # Recursively search children
                    if "children" in value:
                        _walk(value["children"], current_path)
    
    _walk(root_node)
    return all_nodes

# New function: find all pipeline companies in the tree
def find_pipeline_companies(root_node, filter_date: Optional[str] = None) -> List[Dict[str, Any]]:
    
    """
    Traverse the tree and collect all companies listed in any node's pipeline.

    The tree stores pipeline items as a newline-separated string in `node['meta']['pipeline']`.
    Each line commonly follows patterns like:
      - "[YYYY-MM-DD] CompanyName, Description, Tags | Status: Qualifying"
      - "CompanyName, Description"

    This function parses those lines into structured entries and returns a list of:
      {
        'company_name': str,
        'description': str,
        'status': str,
        'date': str,
        'raw': str,            # the original pipeline line
        'node': str,           # node name
        'path': str            # full path e.g. "Commerce > Retail & consumer"
      }
    """

    results: List[Dict[str, Any]] = []

    def parse_pipeline_line(line: str) -> Optional[Dict[str, Any]]:
        data: Dict[str, Any] = {
            'date': '',
            'company_name': '',
            'description': '',
            'status': '',
            'raw': line,
        }

        s = (line or '').strip()
        if not s:
            return None

        # Extract optional [date]
        m = re.match(r"^\[(?P<date>[^\]]+)\]\s*(?P<rest>.*)$", s)
        if m:
            data['date'] = (m.group('date') or '').strip()
            s = (m.group('rest') or '').strip()

        # Extract optional trailing "| Status: ..."
        status_match = re.search(r"\|\s*Status\s*:\s*(.+)$", s, flags=re.IGNORECASE)
        if status_match:
            data['status'] = status_match.group(1).strip()
            s = s[: status_match.start()].strip()

        # Split remaining into company_name and description (first comma)
        if ',' in s:
            name_part, desc_part = s.split(',', 1)
            data['company_name'] = name_part.strip()
            data['description'] = desc_part.strip()
        else:
            data['company_name'] = s

        return data

    def _walk(node: Dict[str, Any], path: List[str] = []):
        if isinstance(node, dict):
            for key, value in node.items():
                if key != 'meta' and isinstance(value, dict):
                    current_path = path + [key]

                    meta = value.get('meta', {}) if isinstance(value.get('meta', {}), dict) else {}
                    pipeline_str = meta.get('pipeline', '')
                    if isinstance(pipeline_str, str) and pipeline_str.strip():
                        for line in pipeline_str.splitlines():
                            parsed = parse_pipeline_line(line)
                            if parsed:
                                results.append({
                                    'node': current_path[-1],
                                    'path': ' > '.join(current_path),
                                    **parsed,
                                })

                    if 'children' in value:
                        _walk(value['children'], current_path)

    _walk(root_node)

    # Filter out companies with "Pass" or "Passed" status
    results = [
        r for r in results
        if r['status'].lower() not in ['pass', 'passed']
    ]

    if filter_date:
        cutoff = datetime.fromisoformat(filter_date).date()
        results = [
            r for r in results
            if r['date'] and datetime.fromisoformat(r['date']).date() >= cutoff
        ]
    return results

# Test functions (uncomment to run manually)
# if __name__ == "__main__":
#     import json
#     
#     # Load your tree
#     with open('data/taste_tree.json', 'r') as f:
#         tree = json.load(f)
#     
#     # Test the improved functions
#     print("=== Find nodes by name ===")
#     results = find_nodes_by_name(tree, "commerce")
#     for result in results:
#         print(f"Name: {result['name']}")
#         print(f"Path: {result['path']}")
#         print(f"Has children: {result['has_children']} ({result['child_count']} children)")
#         print(f"Investment status: {result.get('investment_status', 'N/A')}")
#         print(f"Portfolio companies: {result.get('portfolio_companies', 0)}")
#         print(f"Interest: {result.get('interest', 'N/A')}")
#         print(f"Recent news: {result.get('recent_news', 'N/A')}")
#         print(f"Caution: {result.get('caution', 'N/A')}")
#         print(f"Montage lead: {result.get('montage_lead', 'N/A')}")
#         print(f"Last updated: {result.get('last_updated', 'N/A')}")
#         print(f"Description: {result.get('description', 'N/A')}")
#         print(f"Thesis: {result.get('thesis', 'N/A')}")
#         print()
#     
#     
#     print("\n=== Search for AI content ===")
#     ai_matches = search_tree_content(tree, "MCP")
#     print("Found ", len(ai_matches), " nodes with mention of ", "MCP")
#     for match in ai_matches[:5]:  # Show first 5 matches
#         print(f"Path: {match['path']}")
#         print(f"Field: {match['field']}")
#         print(f"Content: {match['content']}")
#         print()