"""
Path Mapper - Translates old tree paths to new tree paths.

After the tree restructuring, the database contains tree_path values with old node names,
but the current tree uses new names. This module provides translation between them.
"""

import json
import os

_mapping_cache = None

def load_mapping():
    """Load the node name mapping from JSON file."""
    global _mapping_cache
    
    if _mapping_cache is not None:
        return _mapping_cache
    
    mapping_file = 'data/node_name_mapping.json'
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(
            f"Node mapping file not found: {mapping_file}\n"
            "Run: python3 workflows/create_node_mapping.py"
        )
    
    with open(mapping_file, 'r') as f:
        _mapping_cache = json.load(f)
    
    return _mapping_cache

def translate_old_path_to_new(old_path):
    """Translate an old tree path to the new tree path.
    
    Args:
        old_path: Path from database (e.g., "AI > AI Applications > AI Agents")
    
    Returns:
        tuple: (new_path, confidence, match_type)
        - new_path: Translated path in new tree structure
        - confidence: Confidence score (0.0 to 1.0)
        - match_type: Type of match ('unchanged', 'exact_match', 'similarity_match', etc.)
    
    Example:
        >>> translate_old_path_to_new("AI > AI Applications > AI Agents")
        ("AI > Agent & Automation Infrastructure", 0.85, "similarity_match")
    """
    mapping = load_mapping()
    
    if old_path in mapping:
        info = mapping[old_path]
        return info['new_path'], info['confidence'], info['match_type']
    
    # If not in mapping, return as-is (might be a new path)
    return old_path, 1.0, 'not_in_mapping'

def translate_new_path_to_old(new_path):
    """Translate a new tree path back to old path(s).
    
    This is useful for reverse lookups. Note that multiple old paths
    might map to the same new path.
    
    Args:
        new_path: Path in current tree structure
    
    Returns:
        list: List of old paths that map to this new path
    """
    mapping = load_mapping()
    
    old_paths = []
    for old_path, info in mapping.items():
        if info['new_path'] == new_path:
            old_paths.append(old_path)
    
    return old_paths

def get_all_matching_old_paths(new_path_pattern):
    """Get all old paths that match a new path pattern (substring match).
    
    This is useful for finding all database entries that should match
    a given node in the new tree structure.
    
    Args:
        new_path_pattern: Pattern to match (e.g., "AI > Agent")
    
    Returns:
        list: List of old paths that map to paths containing the pattern
    
    Example:
        >>> get_all_matching_old_paths("AI > Agent")
        ["AI > AI Applications > AI Agents", "AI > AI Applications > AI Agents > AI Personal Assistants", ...]
    """
    mapping = load_mapping()
    
    matching_old_paths = []
    for old_path, info in mapping.items():
        new_path = info['new_path']
        if new_path and new_path_pattern in new_path:
            matching_old_paths.append(old_path)
    
    return matching_old_paths

def get_mapping_stats():
    """Get statistics about the mapping."""
    mapping = load_mapping()
    
    stats = {
        'total': len(mapping),
        'unchanged': 0,
        'exact_match': 0,
        'similarity_match': 0,
        'top_level_fallback': 0,
        'no_match': 0,
        'not_in_mapping': 0
    }
    
    for info in mapping.values():
        match_type = info['match_type']
        if match_type in stats:
            stats[match_type] += 1
    
    return stats

if __name__ == '__main__':
    # Test the mapping
    print("Testing path mapper...")
    print("=" * 70)
    
    test_paths = [
        "AI > AI Applications > AI Agents",
        "Commerce > Retail & consumer > DTC Brands",
        "Fintech > Payments",
        "Healthcare > Digital Health"
    ]
    
    for old_path in test_paths:
        new_path, confidence, match_type = translate_old_path_to_new(old_path)
        print(f"\nOLD: {old_path}")
        print(f"NEW: {new_path}")
        print(f"Confidence: {confidence:.2f} ({match_type})")
    
    print("\n" + "=" * 70)
    print("Mapping statistics:")
    stats = get_mapping_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
