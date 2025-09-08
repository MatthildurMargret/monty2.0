import json
import pandas as pd
from collections import defaultdict
from anytree import Node, RenderTree

# -------------------
# Load the JSON tree
# -------------------
def load_tree(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

# -------------------
# Flatten the tree
# -------------------
def flatten_tree(node_dict, path=None, depth=0, results=None):
    if results is None:
        results = []
    if path is None:
        path = []

    # Handle the case where node_dict is the root dictionary
    for key, value in node_dict.items():
        if key == 'meta':
            continue
            
        current_path = path + [key]
        meta = value.get('meta', {})
        children = value.get('children', {})

        results.append({
            "path": " > ".join(current_path),
            "node_name": key,
            "depth": depth,
            "interest": meta.get("interest"),
            "investment_status": meta.get("investment_status"),
            "num_children": len(children),
            "thesis": meta.get("thesis", None),
            "montage_lead": meta.get("montage_lead"),
            "caution": meta.get("caution"),
            "description": meta.get("description")
        })

        # Recursively process children
        if children:
            flatten_tree(children, current_path, depth + 1, results)

    return results

# -------------------
# Sample path exploration
# -------------------
import random

def get_all_paths(node_dict, current_path=None, all_paths=None):
    """Get all possible paths through the tree."""
    if current_path is None:
        current_path = []
    if all_paths is None:
        all_paths = []
    
    for key, value in node_dict.items():
        if key == 'meta':
            continue
            
        new_path = current_path + [key]
        children = value.get('children', {})
        
        if children:
            get_all_paths(children, new_path, all_paths)
        else:
            # This is a leaf node
            all_paths.append(new_path)
    
    return all_paths

def show_path_options(node_dict, path):
    """Show the available options at each step of a given path."""
    current_node = node_dict
    
    print(f"\n🔍 Exploring path: {' > '.join(path)}")
    print("=" * 60)
    
    for i, step in enumerate(path):
        # Show current level options
        options = [key for key in current_node.keys() if key != 'meta']
        
        print(f"\nStep {i+1}: Choose from {len(options)} options:")
        for j, option in enumerate(options, 1):
            marker = "👉" if option == step else "  "
            print(f"  {marker} {j}. {option}")
        
        if step in current_node:
            current_node = current_node[step].get('children', {})
            
            # Show metadata for the chosen option
            meta = current_node.get('meta', {}) if step in current_node else current_node.get(step, {}).get('meta', {})
            if not meta:
                # Try to get meta from the parent level
                meta = node_dict
                for s in path[:i+1]:
                    if s in meta:
                        meta = meta[s].get('meta', {})
                        break
            
            if meta:
                interest = meta.get('interest', '')
                investment_status = meta.get('investment_status', '')
                if interest and len(interest) > 100:
                    interest = interest[:100] + "..."
                if interest or investment_status:
                    print(f"     📊 Interest: {interest or 'Not specified'}")
                    print(f"     💰 Investment Status: {investment_status or 'Not specified'}")
        else:
            break
    
    print("\n" + "="*60)

def show_sample_paths(tree_json, num_samples=5):
    """Show a random sample of paths through the decision tree."""
    all_paths = get_all_paths(tree_json)
    
    print(f"\n🌳 DECISION TREE EXPLORATION")
    print(f"Found {len(all_paths)} total leaf paths in the tree")
    
    # Sample random paths
    sample_paths = random.sample(all_paths, min(num_samples, len(all_paths)))
    
    for path in sample_paths:
        show_path_options(tree_json, path)

# -------------------
# Validation checks
# -------------------
def validate_tree(flat_df):
    issues = defaultdict(list)

    # Missing node names (shouldn't happen with new structure)
    missing_names = flat_df[flat_df["node_name"].isnull()]
    if not missing_names.empty:
        issues["missing_names"] = missing_names["path"].tolist()

    # Duplicate names at same depth
    dups = flat_df.groupby(["depth", "node_name"]).size().reset_index(name="count")
    dups = dups[dups["count"] > 1]
    if not dups.empty:
        issues["duplicate_nodes"] = dups.to_dict(orient="records")

    # Leaf nodes with no metadata (no interest, investment_status, or thesis)
    leaf_nodes = flat_df[flat_df["num_children"] == 0]
    empty_leaves = leaf_nodes[
        (leaf_nodes["interest"].isnull() | (leaf_nodes["interest"] == "")) &
        (leaf_nodes["investment_status"].isnull() | (leaf_nodes["investment_status"] == "")) &
        (leaf_nodes["thesis"].isnull() | (leaf_nodes["thesis"] == ""))
    ]
    if not empty_leaves.empty:
        issues["empty_leaf_nodes"] = empty_leaves["path"].tolist()

    # Nodes with high interest but no investment status
    high_interest_no_status = flat_df[
        (flat_df["interest"].str.contains("excited|high|priority", case=False, na=False)) &
        (flat_df["investment_status"].isnull() | (flat_df["investment_status"] == ""))
    ]
    if not high_interest_no_status.empty:
        issues["high_interest_no_status"] = high_interest_no_status["path"].tolist()

    return issues


# -------------------
# Interest field duplicate checker
# -------------------
def check_interest_duplicates(tree_json):
    """
    Check for duplicate thought entries within each individual node.
    
    Returns:
        dict: Dictionary with duplicate analysis results per node
    """
    import re
    from difflib import SequenceMatcher
    
    def extract_interest_entries_by_node(node_dict, path=None, nodes_with_entries=None):
        """Extract interest field entries grouped by node path."""
        if path is None:
            path = []
        if nodes_with_entries is None:
            nodes_with_entries = {}
        
        for key, value in node_dict.items():
            if key == 'meta':
                continue
                
            current_path = path + [key]
            path_str = ' > '.join(current_path)
            meta = value.get('meta', {})
            interest = meta.get('interest', '')
            
            if interest and interest.strip():
                # Split interest field into individual thoughts/entries
                # Assuming thoughts are separated by timestamps or double newlines
                thoughts = split_interest_into_thoughts(interest.strip())
                
                if len(thoughts) > 1:  # Only store nodes that have multiple thoughts
                    nodes_with_entries[path_str] = {
                        'path': path_str,
                        'thoughts': thoughts,
                        'total_thoughts': len(thoughts)
                    }
            
            # Recurse into children
            children = value.get('children', {})
            if children:
                extract_interest_entries_by_node(children, current_path, nodes_with_entries)
        
        return nodes_with_entries
    
    def split_interest_into_thoughts(interest_text):
        """Split interest text into individual thoughts/entries."""
        # Split by timestamp patterns like [2025-08-26]
        timestamp_pattern = r'\[\d{4}-\d{2}-\d{2}\]'
        
        # First try splitting by timestamps
        thoughts = re.split(timestamp_pattern, interest_text)
        
        # Clean up thoughts and remove empty ones
        cleaned_thoughts = []
        for thought in thoughts:
            cleaned = thought.strip()
            if cleaned and len(cleaned) > 10:  # Ignore very short entries
                cleaned_thoughts.append(cleaned)
        
        # If no timestamp splits found, try splitting by double newlines or periods
        if len(cleaned_thoughts) <= 1:
            # Try double newlines
            thoughts = interest_text.split('\n\n')
            if len(thoughts) <= 1:
                # Try splitting by sentences ending with periods
                thoughts = [s.strip() + '.' for s in interest_text.split('.') if s.strip()]
            
            cleaned_thoughts = []
            for thought in thoughts:
                cleaned = thought.strip()
                if cleaned and len(cleaned) > 10:
                    cleaned_thoughts.append(cleaned)
        
        return cleaned_thoughts
    
    def clean_interest_text(text):
        """Clean interest text by removing timestamps and normalizing whitespace."""
        # Remove timestamp patterns like [2025-08-26]
        cleaned = re.sub(r'\[\d{4}-\d{2}-\d{2}\]', '', text)
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove trailing dots and periods
        cleaned = cleaned.strip().rstrip('.').strip()
        return cleaned
    
    def find_duplicates_in_node(thoughts, similarity_threshold=0.8):
        """Find duplicate thoughts within a single node."""
        exact_duplicates = []
        similar_groups = []
        processed = set()
        
        # Find exact duplicates first
        content_map = {}
        for i, thought in enumerate(thoughts):
            cleaned = clean_interest_text(thought)
            if cleaned in content_map:
                content_map[cleaned].append((i, thought))
            else:
                content_map[cleaned] = [(i, thought)]
        
        for content, thought_list in content_map.items():
            if len(thought_list) > 1:
                exact_duplicates.append({
                    'content': content[:100] + '...' if len(content) > 100 else content,
                    'count': len(thought_list),
                    'indices': [idx for idx, _ in thought_list],
                    'full_thoughts': [thought for _, thought in thought_list]
                })
                # Mark these as processed for similarity check
                for idx, _ in thought_list:
                    processed.add(idx)
        
        # Find similar (but not exact) duplicates
        for i, thought1 in enumerate(thoughts):
            if i in processed:
                continue
                
            group = [{'index': i, 'thought': thought1}]
            processed.add(i)
            
            cleaned1 = clean_interest_text(thought1)
            
            for j, thought2 in enumerate(thoughts[i+1:], i+1):
                if j in processed:
                    continue
                    
                cleaned2 = clean_interest_text(thought2)
                
                # Calculate similarity
                similarity = SequenceMatcher(None, cleaned1, cleaned2).ratio()
                
                if similarity >= similarity_threshold:
                    group.append({'index': j, 'thought': thought2})
                    processed.add(j)
            
            if len(group) > 1:
                similar_groups.append({
                    'similarity_score': max([SequenceMatcher(None, 
                                                           clean_interest_text(group[0]['thought']), 
                                                           clean_interest_text(entry['thought'])).ratio() 
                                           for entry in group[1:]]),
                    'group': group
                })
        
        return exact_duplicates, similar_groups
    
    # Extract interest entries grouped by node
    nodes_with_entries = extract_interest_entries_by_node(tree_json)
    
    # Analyze duplicates within each node
    node_duplicate_results = {}
    total_nodes_analyzed = 0
    total_exact_duplicates = 0
    total_similar_groups = 0
    
    for node_path, node_data in nodes_with_entries.items():
        thoughts = node_data['thoughts']
        
        # Find duplicates within this node
        exact_dups, similar_groups = find_duplicates_in_node(thoughts, similarity_threshold=0.8)
        
        if exact_dups or similar_groups:
            node_duplicate_results[node_path] = {
                'path': node_path,
                'total_thoughts': len(thoughts),
                'exact_duplicates': exact_dups,
                'similar_groups': similar_groups,
                'all_thoughts': thoughts  # For reference
            }
            
            total_exact_duplicates += len(exact_dups)
            total_similar_groups += len(similar_groups)
        
        total_nodes_analyzed += 1
    
    # Overall statistics
    nodes_with_duplicates = len(node_duplicate_results)
    
    return {
        'statistics': {
            'total_nodes_analyzed': total_nodes_analyzed,
            'nodes_with_duplicates': nodes_with_duplicates,
            'total_exact_duplicate_groups': total_exact_duplicates,
            'total_similar_groups': total_similar_groups
        },
        'nodes_with_duplicates': node_duplicate_results
    }


def remove_duplicate_thoughts(tree_json, similarity_threshold=0.8, dry_run=True):
    """
    Remove duplicate thoughts from the tree, keeping only the most recent entry.
    
    Args:
        tree_json (dict): The tree JSON to clean
        similarity_threshold (float): Threshold for considering thoughts similar (0.8 = 80% similar)
        dry_run (bool): If True, only report what would be removed without making changes
        
    Returns:
        dict: Results of the cleanup operation
    """
    import re
    from difflib import SequenceMatcher
    import copy
    
    # Create a copy for modification if not dry run
    if not dry_run:
        tree_json = copy.deepcopy(tree_json)
    
    def extract_timestamp_from_thought(thought):
        """Extract timestamp from thought text, return None if not found."""
        timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2})\]'
        match = re.search(timestamp_pattern, thought)
        if match:
            return match.group(1)
        return None
    
    def clean_interest_text(text):
        """Clean interest text by removing timestamps and normalizing whitespace."""
        # Remove timestamp patterns like [2025-08-26]
        cleaned = re.sub(r'\[\d{4}-\d{2}-\d{2}\]', '', text)
        # Remove extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove trailing dots and periods
        cleaned = cleaned.strip().rstrip('.').strip()
        return cleaned
    
    def split_interest_into_thoughts(interest_text):
        """Split interest text into individual thoughts/entries."""
        # Split by timestamp patterns like [2025-08-26]
        timestamp_pattern = r'\[\d{4}-\d{2}-\d{2}\]'
        
        # First try splitting by timestamps
        thoughts = re.split(timestamp_pattern, interest_text)
        
        # Clean up thoughts and remove empty ones
        cleaned_thoughts = []
        for thought in thoughts:
            cleaned = thought.strip()
            if cleaned and len(cleaned) > 10:  # Ignore very short entries
                cleaned_thoughts.append(cleaned)
        
        # If no timestamp splits found, try splitting by double newlines or periods
        if len(cleaned_thoughts) <= 1:
            # Try double newlines
            thoughts = interest_text.split('\n\n')
            if len(thoughts) <= 1:
                # Try splitting by sentences ending with periods
                thoughts = [s.strip() + '.' for s in interest_text.split('.') if s.strip()]
            
            cleaned_thoughts = []
            for thought in thoughts:
                cleaned = thought.strip()
                if cleaned and len(cleaned) > 10:
                    cleaned_thoughts.append(cleaned)
        
        return cleaned_thoughts
    
    def reconstruct_interest_with_timestamps(original_text, kept_thoughts):
        """Reconstruct interest field with original timestamps for kept thoughts."""
        # Find all timestamp + thought pairs in original text
        timestamp_pattern = r'(\[\d{4}-\d{2}-\d{2}\])([^\[]*?)(?=\[\d{4}-\d{2}-\d{2}\]|$)'
        matches = re.findall(timestamp_pattern, original_text, re.DOTALL)
        
        if not matches:
            # No timestamps found, just join thoughts
            return '\n\n'.join(kept_thoughts)
        
        # Build mapping of cleaned thought to timestamp + original
        thought_to_timestamp = {}
        for timestamp, thought_text in matches:
            cleaned = thought_text.strip()
            if cleaned and len(cleaned) > 10:
                thought_to_timestamp[clean_interest_text(cleaned)] = (timestamp, cleaned)
        
        # Reconstruct with timestamps for kept thoughts
        reconstructed_parts = []
        for thought in kept_thoughts:
            cleaned_thought = clean_interest_text(thought)
            if cleaned_thought in thought_to_timestamp:
                timestamp, original = thought_to_timestamp[cleaned_thought]
                reconstructed_parts.append(f"{timestamp} {original}")
            else:
                # Fallback: add without timestamp
                reconstructed_parts.append(thought)
        
        return '\n\n'.join(reconstructed_parts)
    
    def remove_duplicates_from_node(node_dict, path=None, cleanup_results=None):
        """Remove duplicates from a single node and recurse."""
        if path is None:
            path = []
        if cleanup_results is None:
            cleanup_results = {'nodes_cleaned': 0, 'thoughts_removed': 0, 'details': []}
        
        for key, value in node_dict.items():
            if key == 'meta':
                continue
                
            current_path = path + [key]
            path_str = ' > '.join(current_path)
            meta = value.get('meta', {})
            interest = meta.get('interest', '')
            
            if interest and interest.strip():
                # Split interest field into individual thoughts
                thoughts = split_interest_with_timestamps(interest.strip())
                
                if len(thoughts) > 1:
                    # Find duplicates within this node
                    kept_thoughts, removed_count = deduplicate_thoughts(thoughts, similarity_threshold)
                    
                    if removed_count > 0:
                        cleanup_results['nodes_cleaned'] += 1
                        cleanup_results['thoughts_removed'] += removed_count
                        
                        # Reconstruct the interest field
                        if not dry_run:
                            new_interest = reconstruct_interest_with_timestamps(interest, kept_thoughts)
                            meta['interest'] = new_interest
                        
                        cleanup_results['details'].append({
                            'path': path_str,
                            'original_count': len(thoughts),
                            'final_count': len(kept_thoughts),
                            'removed_count': removed_count
                        })
            
            # Recurse into children
            children = value.get('children', {})
            if children:
                remove_duplicates_from_node(children, current_path, cleanup_results)
        
        return cleanup_results
    
    def split_interest_with_timestamps(interest_text):
        """Split interest preserving timestamp information."""
        # Find all timestamp + thought pairs
        timestamp_pattern = r'(\[\d{4}-\d{2}-\d{2}\])([^\[]*?)(?=\[\d{4}-\d{2}-\d{2}\]|$)'
        matches = re.findall(timestamp_pattern, interest_text, re.DOTALL)
        
        if matches:
            thoughts = []
            for timestamp, thought_text in matches:
                cleaned = thought_text.strip()
                if cleaned and len(cleaned) > 10:
                    thoughts.append({
                        'timestamp': timestamp,
                        'text': cleaned,
                        'full': f"{timestamp} {cleaned}"
                    })
            return thoughts
        else:
            # Fallback to simple splitting
            simple_thoughts = split_interest_into_thoughts(interest_text)
            return [{'timestamp': None, 'text': t, 'full': t} for t in simple_thoughts]
    
    def deduplicate_thoughts(thoughts, threshold):
        """Remove duplicate thoughts, keeping the most recent."""
        if not thoughts:
            return [], 0
        
        # Sort by timestamp (most recent first), handle None timestamps
        def sort_key(thought):
            if thought['timestamp']:
                return thought['timestamp']
            return '1900-01-01'  # Put non-timestamped items at the end
        
        sorted_thoughts = sorted(thoughts, key=sort_key, reverse=True)
        
        kept_thoughts = []
        removed_count = 0
        processed_indices = set()
        
        for i, thought1 in enumerate(sorted_thoughts):
            if i in processed_indices:
                continue
            
            # This thought will be kept (it's the most recent of its group)
            kept_thoughts.append(thought1['text'])
            processed_indices.add(i)
            
            cleaned1 = clean_interest_text(thought1['text'])
            
            # Find and mark similar thoughts for removal
            for j, thought2 in enumerate(sorted_thoughts[i+1:], i+1):
                if j in processed_indices:
                    continue
                
                cleaned2 = clean_interest_text(thought2['text'])
                
                # Check for exact match or high similarity
                if cleaned1 == cleaned2 or SequenceMatcher(None, cleaned1, cleaned2).ratio() >= threshold:
                    processed_indices.add(j)
                    removed_count += 1
        
        return kept_thoughts, removed_count
    
    # Perform the cleanup
    results = remove_duplicates_from_node(tree_json)
    
    return {
        'tree_json': tree_json if not dry_run else None,
        'dry_run': dry_run,
        'nodes_cleaned': results['nodes_cleaned'],
        'thoughts_removed': results['thoughts_removed'],
        'details': results['details']
    }


def print_duplicate_analysis(duplicate_results):
    """
    Print a formatted report of duplicate analysis results for individual nodes.
    """
    print("\n🔍 NODE-LEVEL DUPLICATE THOUGHT ANALYSIS")
    print("=" * 60)
    
    stats = duplicate_results['statistics']
    print(f"\n📊 Statistics:")
    print(f"  • Total nodes analyzed: {stats['total_nodes_analyzed']}")
    print(f"  • Nodes with duplicates: {stats['nodes_with_duplicates']}")
    print(f"  • Total exact duplicate groups: {stats['total_exact_duplicate_groups']}")
    print(f"  • Total similar groups: {stats['total_similar_groups']}")
    
    nodes_with_dups = duplicate_results['nodes_with_duplicates']
    
    if not nodes_with_dups:
        print("\n✅ No duplicate thoughts found within any nodes!")
        return
    
    print(f"\n🎯 Nodes with Duplicate Thoughts: {len(nodes_with_dups)}")
    print("-" * 50)
    
    for node_path, node_data in nodes_with_dups.items():
        print(f"\n📍 Node: {node_path}")
        print(f"   Total thoughts: {node_data['total_thoughts']}")
        
        # Show exact duplicates in this node
        exact_dups = node_data['exact_duplicates']
        if exact_dups:
            print(f"\n   🎯 Exact Duplicates: {len(exact_dups)} groups")
            for i, dup in enumerate(exact_dups, 1):
                print(f"     {i}. Appears {dup['count']} times (indices: {dup['indices']}):")
                print(f"        Content: {dup['content']}")
                # Show first few full thoughts
                for j, thought in enumerate(dup['full_thoughts'][:2], 1):
                    print(f"        [{j}] {thought[:100]}{'...' if len(thought) > 100 else ''}")
                if len(dup['full_thoughts']) > 2:
                    print(f"        ... and {len(dup['full_thoughts']) - 2} more")
        
        # Show similar groups in this node
        similar_groups = node_data['similar_groups']
        if similar_groups:
            print(f"\n   🔄 Similar Groups: {len(similar_groups)}")
            for i, group in enumerate(similar_groups, 1):
                print(f"     Group {i} (similarity: {group['similarity_score']:.2f}):")
                for entry in group['group']:
                    thought_preview = entry['thought'][:80] + ('...' if len(entry['thought']) > 80 else '')
                    print(f"       [{entry['index']}] {thought_preview}")
        
        print()  # Add spacing between nodes
    
    # Summary recommendations
    print("\n💡 Recommendations:")
    print("  • Review nodes with exact duplicates - these are likely copy-paste errors")
    print("  • Consider consolidating similar thoughts within nodes")
    print("  • Use the indices shown to locate specific duplicate thoughts for removal")


def print_cleanup_results(cleanup_results):
    """
    Print a formatted report of duplicate cleanup results.
    """
    print(f"\n🧹 DUPLICATE CLEANUP RESULTS")
    print("=" * 50)
    
    if cleanup_results['dry_run']:
        print("🔍 DRY RUN - No changes were made")
    else:
        print("✅ CLEANUP COMPLETED - Changes applied to tree")
    
    print(f"\n📊 Summary:")
    print(f"  • Nodes cleaned: {cleanup_results['nodes_cleaned']}")
    print(f"  • Duplicate thoughts removed: {cleanup_results['thoughts_removed']}")
    
    if cleanup_results['details']:
        print(f"\n📝 Details:")
        for detail in cleanup_results['details']:
            print(f"\n📍 {detail['path']}")
            print(f"   Before: {detail['original_count']} thoughts")
            print(f"   After:  {detail['final_count']} thoughts")
            print(f"   Removed: {detail['removed_count']} duplicates")
    
    if cleanup_results['dry_run'] and cleanup_results['thoughts_removed'] > 0:
        print(f"\n💡 To apply these changes, run with dry_run=False")
    elif not cleanup_results['dry_run'] and cleanup_results['thoughts_removed'] > 0:
        print(f"\n✅ Tree has been cleaned! Consider saving the updated tree to a file.")
    elif cleanup_results['thoughts_removed'] == 0:
        print(f"\n✨ No duplicates found - tree is already clean!")


# -------------------
# Recent deals integration
# -------------------
def update_tree_with_recent_deals(tree_json, deals_csv_path="data/deal_data/all_deals.csv", days_back=7):
    """
    Update the tree's recent_news metadata fields with deals from the past N days.
    Uses LLM to find the most appropriate node for each deal, similar to thought insertion.
    
    Args:
        tree_json (dict): The loaded tree JSON
        deals_csv_path (str): Path to the deals CSV file
        days_back (int): Number of days back to look for deals
        
    Returns:
        tuple: (updated_tree, deals_added_count, deals_processed)
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import re
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    
    # Load OpenAI client
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    
    # Load deals data
    try:
        deals_df = pd.read_csv(deals_csv_path)
        print(f"Loaded {len(deals_df)} deals from {deals_csv_path}")
    except Exception as e:
        print(f"Error loading deals CSV: {e}")
        return tree_json, 0, []
    
    # Convert date column to datetime
    deals_df['Date'] = pd.to_datetime(deals_df['Date'], errors='coerce')
    
    # Filter for recent deals
    cutoff_date = datetime.now() - timedelta(days=days_back)
    recent_deals = deals_df[deals_df['Date'] >= cutoff_date].copy()
    
    print(f"Found {len(recent_deals)} deals from the past {days_back} days")
    
    if len(recent_deals) == 0:
        return tree_json, 0, []
    
    def format_multi_level_context(children, deal_description):
        """Format children and their sub-categories for multi-level context window."""
        lines = []
        for name, child in children.items():
            meta = child.get("meta", {})
            line = f"**{name}**"
            if "description" in meta:
                line += f"\n  Description: {meta['description']}"
            
            # Show sub-categories if they exist
            if "children" in child and child["children"]:
                subcats = list(child["children"].keys())[:5]  # Show first 5 subcategories
                line += f"\n  Sub-categories: {', '.join(subcats)}"
                if len(child["children"]) > 5:
                    line += f" (and {len(child['children']) - 5} more)"
            
            lines.append(line)
        return "\n\n".join(lines)
    
    def find_best_node_for_deal(node, deal_description, path=None):
        """Find the best node for a deal using LLM analysis."""
        if path is None:
            path = []
        
        if "children" not in node or not node["children"]:
            return path  # Leaf node, stop here

        multi_level_context = format_multi_level_context(node["children"], deal_description)
        
        prompt = f"""
You are an investment analyst categorizing a new deal into a thematic investment tree.

Deal Description:
---
{deal_description}
---

Available categories and sub-categories:
---
{multi_level_context}
---

Instructions:
1. Pick the most relevant top-level category (among children), or say "STOP" if you've reached the right level.
2. Choose the category that best matches the deal's industry, technology, or business model.

Respond with just the category name, or "STOP" if the current level is appropriate.

Examples:
Fintech
Healthcare  
STOP
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an investment analyst deciding the best category for a new deal."},
                    {"role": "user", "content": prompt.strip()}
                ],
                temperature=0.2,
                max_tokens=30
            )
            
            choice = response.choices[0].message.content.strip()
            
            if choice == "STOP" or choice not in node["children"]:
                return path
            
            return find_best_node_for_deal(node["children"][choice], deal_description, path + [choice])
        except Exception as e:
            print(f"  Warning: LLM call failed for deal categorization: {e}")
            return path  # Return current path as fallback
    
    def format_deal_entry(deal_row):
        """Format a deal into a timestamped news entry."""
        company = deal_row.get('Company', 'Unknown Company')
        amount = deal_row.get('Amount', 'Undisclosed')
        round_type = deal_row.get('Funding Round', 'Unknown Round')
        description = deal_row.get('Vertical', 'No description')
        investors = deal_row.get('Investors', 'Undisclosed investors')
        date = deal_row.get('Date')
        
        # Format amount
        if pd.notna(amount) and str(amount) != 'nan':
            amount_str = f"${amount}" if not str(amount).startswith('$') else str(amount)
        else:
            amount_str = "Undisclosed"
        
        # Format date
        if pd.notna(date):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = 'Unknown date'
        
        return f"[{date_str}] {company} raised {amount_str} in {round_type} for {description}. Investors: {investors}"
    
    def create_deal_description(deal_row):
        """Create a comprehensive description of the deal for LLM analysis."""
        company = deal_row.get('Company', 'Unknown Company')
        vertical = deal_row.get('Vertical', '')
        category = deal_row.get('Category', '')
        description = deal_row.get('Description', '')
        
        parts = [f"Company: {company}"]
        if vertical and str(vertical) != 'nan':
            parts.append(f"Vertical: {vertical}")
        if category and str(category) != 'nan':
            parts.append(f"Category: {category}")
        if description and str(description) != 'nan':
            parts.append(f"Description: {description}")
        
        return ". ".join(parts)
    
    def add_deal_to_node_by_path(tree_dict, path, deal_entry):
        """Add a deal entry to a specific node's recent_news field using path."""
        current_node = tree_dict
        
        # Navigate to the correct node
        for step in path:
            if step in current_node:
                current_node = current_node[step]
                # If this is not the final step and we have children, go into children
                if step != path[-1] and 'children' in current_node:
                    current_node = current_node['children']
            else:
                return False
        
        # Ensure meta exists
        if 'meta' not in current_node:
            current_node['meta'] = {}
        
        # Get existing recent_news or create empty string
        existing_news = current_node['meta'].get('recent_news', '')
        
        # Add the new deal entry
        if existing_news:
            current_node['meta']['recent_news'] = f"{existing_news}\n{deal_entry}"
        else:
            current_node['meta']['recent_news'] = deal_entry
        
        return True
    
    # Create a root node structure to match the expected format for LLM analysis
    root_node = {
        "meta": {},
        "children": tree_json
    }
    
    # Process each recent deal
    deals_added = 0
    deals_processed = []
    
    for idx, deal in recent_deals.iterrows():
        deal_description = create_deal_description(deal)
        deal_entry = format_deal_entry(deal)
        
        # Find the best node path for this deal
        best_path = find_best_node_for_deal(root_node, deal_description)
        
        if not best_path:
            # If no specific path found, add to root level (shouldn't happen often)
            print(f"  ⚠️  No specific category found for {deal.get('Company', 'Unknown')}, skipping")
            continue
        
        # Try to add to the determined path
        success = add_deal_to_node_by_path(tree_json, best_path, deal_entry)
        
        if success:
            deals_added += 1
            deals_processed.append({
                'company': deal.get('Company', ''),
                'path': ' > '.join(best_path),
                'amount': deal.get('Amount', ''),
                'date': deal.get('Date', '')
            })
            print(f"  ✓ Added {deal.get('Company', 'Unknown')} to {' > '.join(best_path)}")
        else:
            print(f"  ✗ Failed to add {deal.get('Company', 'Unknown')} to {' > '.join(best_path)}")
    
    return tree_json, deals_added, deals_processed


def save_updated_tree(tree_json, output_path="data/taste_tree.json"):
    """
    Save the updated tree to a new JSON file.
    
    Args:
        tree_json (dict): Updated tree JSON
        output_path (str): Path to save the updated tree
    """
    import json
    
    try:
        with open(output_path, 'w') as f:
            json.dump(tree_json, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Updated tree saved to {output_path}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving updated tree: {e}")
        return False


def show_recent_news_summary(tree_json):
    """
    Show a summary of recent news entries across all nodes.
    
    Args:
        tree_json (dict): The tree JSON
    """
    def extract_recent_news(node_dict, path=None, news_entries=None):
        if path is None:
            path = []
        if news_entries is None:
            news_entries = []
        
        for key, value in node_dict.items():
            if key == 'meta':
                continue
                
            current_path = path + [key]
            meta = value.get('meta', {})
            recent_news = meta.get('recent_news', '')
            
            if recent_news and recent_news.strip():
                # Count number of entries (by counting date patterns)
                import re
                date_pattern = r'\[\d{4}-\d{2}-\d{2}\]'
                entry_count = len(re.findall(date_pattern, recent_news))
                
                news_entries.append({
                    'path': ' > '.join(current_path),
                    'entry_count': entry_count,
                    'content': recent_news[:200] + '...' if len(recent_news) > 200 else recent_news
                })
            
            # Recurse into children
            children = value.get('children', {})
            if children:
                extract_recent_news(children, current_path, news_entries)
        
        return news_entries
    
    news_entries = extract_recent_news(tree_json)
    
    print(f"\n📰 RECENT NEWS SUMMARY")
    print("=" * 50)
    print(f"Found recent news in {len(news_entries)} nodes:")
    
    for entry in news_entries:
        print(f"\n📍 {entry['path']} ({entry['entry_count']} entries)")
        print(f"   {entry['content']}")

def similarity_experiment():
    # Load tree
    from services.tree import find_best_node_for_company, find_similar_nodes
    import pandas as pd
    import random
    from collections import Counter
    
    json_path = "data/taste_tree.json"
    tree_json = load_tree(json_path)
    root = {"meta": {}, "children": tree_json}
    
    # Load the 20 most recent deals from data/deal_data/all_deals.csv
    deals_df = pd.read_csv("data/deal_data/all_deals.csv")
    
    # Convert Date column to datetime and sort by most recent
    deals_df['Date'] = pd.to_datetime(deals_df['Date'])
    recent_deals = deals_df.sort_values('Date', ascending=False).head(20)
    
    print(f"Testing {len(recent_deals)} most recent deals")
    print(f"Date range: {recent_deals['Date'].min().strftime('%Y-%m-%d')} to {recent_deals['Date'].max().strftime('%Y-%m-%d')}")
    print(f"{'='*60}\n")
    
    # Debug: Check tree structure
    print(f"Tree has {len(tree_json)} top-level categories: {list(tree_json.keys())}")
    wrapped_tree = {'children': tree_json}
    
    # Track overall results
    all_results = []
    deals_with_similar_nodes = 0
    total_consistency_scores = []
    
    # Loop through each recent deal
    for idx, deal in recent_deals.iterrows():
        company_name = deal['Company']
        company_description = f"{company_name} - {deal['Vertical']} - {deal['Amount']} {deal['Funding Round']}"
        
        print(f"\n{'='*40}")
        print(f"DEAL {len(all_results) + 1}/20: {company_name}")
        print(f"{'='*40}")
        print(f"Description: {company_description}")
        print(f"Date: {deal['Date'].strftime('%Y-%m-%d')}")
        
        # Find the best node for this deal
        initial_path = find_best_node_for_company(root, company_description)
        initial_node_name = initial_path[-1] if initial_path else "ROOT"
        
        print(f"Best node: {' > '.join(initial_path) if initial_path else 'ROOT'}")
        print(f"Node name for similarity search: '{initial_node_name}'")
        
        # Find all the similar nodes in the tree - use root structure
        similar_nodes = find_similar_nodes(root, initial_node_name)
        num_similar_nodes = len(similar_nodes)
        
        print(f"Similar nodes: {num_similar_nodes}")
        if similar_nodes and num_similar_nodes <= 5:  # Only show if not too many
            for node in similar_nodes:
                print(f"  - {node['path']}")
        elif similar_nodes:
            print(f"  (Too many to display - showing count only)")
    
        # Check if we should skip iterations when there are no similar nodes
        if num_similar_nodes == 0:
            print("   ⚠️  No similar nodes - skipping consistency trials")
            
            # Store result for this deal
            deal_result = {
                'company': company_name,
                'description': company_description,
                'similar_nodes_count': num_similar_nodes,
                'trials': 0,
                'most_common_node': initial_node_name,
                'consistency_rate': 'N/A',
                'placement': ' > '.join(initial_path) if initial_path else 'ROOT'
            }
            all_results.append(deal_result)
            continue
        
        # Count deals with similar nodes for summary
        deals_with_similar_nodes += 1
        
        # Run consistency trials for this deal
        num_trials = 5  # Reduced from 20 since we're testing 20 deals
        node_results = []
        
        print(f"   Running {num_trials} consistency trials...")
        
        for i in range(num_trials):
            path = find_best_node_for_company(root, company_description)
            final_node = path[-1] if path else "ROOT"
            full_path = " > ".join(path) if path else "ROOT"
            node_results.append((final_node, full_path))
        
        # Count occurrences for this deal
        from collections import Counter
        node_counter = Counter([result[0] for result in node_results])
        most_common_node = node_counter.most_common(1)[0]
        consistency_percentage = (most_common_node[1] / num_trials) * 100
        
        print(f"   Consistency: {consistency_percentage:.1f}% ({most_common_node[1]}/{num_trials} times to {most_common_node[0]})")
        
        # Store result for this deal
        deal_result = {
            'company': company_name,
            'description': company_description,
            'similar_nodes_count': num_similar_nodes,
            'trials': num_trials,
            'most_common_node': most_common_node[0],
            'consistency_rate': consistency_percentage,
            'placement': ' > '.join(initial_path) if initial_path else 'ROOT'
        }
        all_results.append(deal_result)
        total_consistency_scores.append(consistency_percentage)
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL CONSISTENCY ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Total deals tested: {len(all_results)}")
    print(f"Deals with similar nodes: {deals_with_similar_nodes}")
    print(f"Deals without similar nodes: {len(all_results) - deals_with_similar_nodes}")
    
    if total_consistency_scores:
        avg_consistency = sum(total_consistency_scores) / len(total_consistency_scores)
        print(f"\nAverage consistency rate: {avg_consistency:.1f}%")
        
        # Categorize consistency levels
        high_consistency = sum(1 for score in total_consistency_scores if score >= 80)
        moderate_consistency = sum(1 for score in total_consistency_scores if 60 <= score < 80)
        low_consistency = sum(1 for score in total_consistency_scores if score < 60)
        
        print(f"\nConsistency breakdown:")
        print(f"  High (≥80%): {high_consistency}/{len(total_consistency_scores)} deals")
        print(f"  Moderate (60-79%): {moderate_consistency}/{len(total_consistency_scores)} deals")
        print(f"  Low (<60%): {low_consistency}/{len(total_consistency_scores)} deals")
        
        # Overall assessment
        if avg_consistency >= 80:
            assessment = "HIGH - LLM routing is very stable across deals"
        elif avg_consistency >= 60:
            assessment = "MODERATE - LLM routing is somewhat stable"
        else:
            assessment = "LOW - LLM routing is unstable"
        
        print(f"\nOverall assessment: {assessment}")
    else:
        print("\nNo consistency trials run (all deals had no similar nodes)")
    
    # Show most problematic deals (lowest consistency)
    if total_consistency_scores:
        print(f"\n{'='*40}")
        print("MOST INCONSISTENT DEALS")
        print(f"{'='*40}")
        
        # Sort deals by consistency rate (lowest first)
        deals_with_scores = [(result, result['consistency_rate']) for result in all_results if result['consistency_rate'] != 'N/A']
        deals_with_scores.sort(key=lambda x: x[1])
        
        for i, (deal, score) in enumerate(deals_with_scores[:5]):  # Show top 5 most inconsistent
            print(f"{i+1}. {deal['company']} - {score:.1f}% consistency ({deal['similar_nodes_count']} similar nodes)")
    
    return {
        'total_deals': len(all_results),
        'deals_with_similar_nodes': deals_with_similar_nodes,
        'average_consistency': sum(total_consistency_scores) / len(total_consistency_scores) if total_consistency_scores else 'N/A',
        'all_deal_results': all_results,
        'consistency_scores': total_consistency_scores
    }


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tree analysis and testing tools')
    parser.add_argument('--similarity', action='store_true', help='Run similarity experiment')
    parser.add_argument('--deals', action='store_true', help='Run recent deals integration')
    parser.add_argument('--duplicates', action='store_true', help='Run duplicate detection and cleanup')
    parser.add_argument('--flatten', action='store_true', help='Flatten tree to CSV')
    parser.add_argument('--validation', action='store_true', help='Run validation checks')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--tree-path', default='data/taste_tree.json', help='Path to tree JSON file')
    
    args = parser.parse_args()
    
    # If no specific test is selected, show help
    if not any([args.similarity, args.deals, args.duplicates, args.flatten, args.validation, args.all]):
        parser.print_help()
        print("\nExample usage:")
        print("  python test_tree.py --similarity")
        print("  python test_tree.py --deals --duplicates")
        print("  python test_tree.py --all")
        exit(0)
    
    json_path = args.tree_path
    tree_json = load_tree(json_path)
    
    print(f"Loaded tree from: {json_path}")
    print(f"Running tests: {', '.join([name for name, value in vars(args).items() if value and name not in ['tree_path', 'all']])}")
    
    # Flatten tree to CSV
    if args.flatten or args.all:
        print("\n" + "="*50)
        print("TREE FLATTENING")
        print("="*50)
        flat = flatten_tree(tree_json)
        df = pd.DataFrame(flat)
        df.to_csv("tree_flattened.csv", index=False)
        print("Flattened tree saved to tree_flattened.csv")

    # Update tree with recent deals
    if args.deals or args.all:
        print("\n" + "="*50)
        print("RECENT DEALS INTEGRATION")
        print("="*50)

        updated_tree, deals_added, deals_processed = update_tree_with_recent_deals(tree_json, days_back=30)

        if deals_added > 0:
            print(f"\nSuccessfully added {deals_added} recent deals to the taxonomy tree")

            # Show summary of deals processed
            print("\n📈 DEALS PROCESSED:")
            for deal in deals_processed:
                print(f"  • {deal['company']} ({deal['path']}) - {deal['amount']} on {deal['date']}")

            # Show recent news summary
            show_recent_news_summary(updated_tree)

            # Save updated tree
            save_updated_tree(updated_tree)
        else:
            print("\nNo recent deals found in the past 7 days")

    # Check for duplicate thoughts within nodes
    if args.duplicates or args.all:
        print("\n" + "="*50)
        print("DUPLICATE THOUGHT DETECTION")
        print("="*50)
        
        duplicate_results = check_interest_duplicates(tree_json)
        print_duplicate_analysis(duplicate_results)
        
        if duplicate_results['statistics']['nodes_with_duplicates'] > 0:
            print("\n" + "="*50)
            print("DUPLICATE CLEANUP")
            print("="*50)
            
            # First do a dry run to show what would be removed
            print("\n🔍 Performing dry run to preview changes...")
            dry_run_results = remove_duplicate_thoughts(tree_json, similarity_threshold=0.8, dry_run=True)
            print_cleanup_results(dry_run_results)
            
            if dry_run_results['thoughts_removed'] > 0:
                # Actually perform the cleanup
                print("\n🧹 Applying cleanup...")
                cleanup_results = remove_duplicate_thoughts(tree_json, similarity_threshold=0.8, dry_run=False)
                print_cleanup_results(cleanup_results)
                
                # Update tree_json with cleaned version
                tree_json = cleanup_results['tree_json']
                
                # Optionally save the cleaned tree
                save_cleaned_tree = True  # Set to False to skip saving
                if save_cleaned_tree:
                    import json
                    output_path = "data/taste_tree.json"
                    with open(output_path, 'w') as f:
                        json.dump(tree_json, f, indent=2)
                    print(f"\n💾 Cleaned tree saved to: {output_path}")
        else:
            print("\n✨ No duplicates found - tree is already clean!")

    # Run similarity experiment
    if args.similarity or args.all:
        print("\n" + "="*50)
        print("SIMILARITY EXPERIMENT")
        print("="*50)
        
        similarity_experiment()

    # Validation
    if args.validation or args.all:
        print("\n" + "="*50)
        print("VALIDATION REPORT")
        print("="*50)
        
        flat = flatten_tree(tree_json)
        df = pd.DataFrame(flat)
        issues = validate_tree(df)
        if issues:
            for k, v in issues.items():
                print(f"- {k}: {v}")
        else:
            print("No issues found 🚀")
