"""
Create a mapping from old tree node names to new tree node names.
This is needed because the database has tree_path values with old node names,
but the current tree has been restructured with new names.
"""

import json
from difflib import SequenceMatcher

def similarity(a, b):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def collect_node_names(tree, path=[]):
    """Collect all node names with their full paths."""
    nodes = {}
    for key, value in tree.items():
        current_path = path + [key]
        full_path = ' > '.join(current_path)
        nodes[full_path] = {
            'name': key,
            'full_path': full_path,
            'depth': len(current_path)
        }
        if 'children' in value and value['children']:
            nodes.update(collect_node_names(value['children'], current_path))
    return nodes

def find_best_match(old_path, old_name, current_nodes):
    """Find the best matching new path for an old path."""
    
    # Extract components from old path
    old_parts = old_path.split(' > ')
    old_top_level = old_parts[0] if old_parts else ''
    
    # Strategy 1: Exact name match at same depth
    for new_path, info in current_nodes.items():
        if info['name'] == old_name and info['depth'] == len(old_parts):
            new_parts = new_path.split(' > ')
            if new_parts[0] == old_top_level:  # Same top-level category
                return new_path, 1.0, 'exact_match'
    
    # Strategy 2: High similarity name match at same top-level
    best_match = None
    best_score = 0.0
    
    for new_path, info in current_nodes.items():
        new_parts = new_path.split(' > ')
        
        # Must be same top-level category
        if new_parts[0] != old_top_level:
            continue
        
        # Calculate similarity
        name_sim = similarity(old_name, info['name'])
        path_sim = similarity(old_path, new_path)
        
        # Weighted score (name is more important)
        score = (name_sim * 0.7) + (path_sim * 0.3)
        
        if score > best_score and score > 0.5:  # Threshold
            best_score = score
            best_match = new_path
    
    if best_match:
        return best_match, best_score, 'similarity_match'
    
    # Strategy 3: Map to top-level category if no good match
    if old_top_level in current_nodes:
        return old_top_level, 0.3, 'top_level_fallback'
    
    return None, 0.0, 'no_match'

def create_mapping():
    """Create mapping from old paths to new paths."""
    
    # Load both trees
    with open('data/taste_tree.json', 'r') as f:
        current_tree = json.load(f)
    
    with open('data/taste_tree_backup.json', 'r') as f:
        old_tree = json.load(f)
    
    # Collect all nodes
    current_nodes = {}
    for branch_name, branch_data in current_tree.items():
        if 'children' in branch_data:
            current_nodes.update(collect_node_names(branch_data['children'], [branch_name]))
        # Add top-level too
        current_nodes[branch_name] = {
            'name': branch_name,
            'full_path': branch_name,
            'depth': 1
        }
    
    old_nodes = {}
    for branch_name, branch_data in old_tree.items():
        if 'children' in branch_data:
            old_nodes.update(collect_node_names(old_tree[branch_name]['children'], [branch_name]))
        old_nodes[branch_name] = {
            'name': branch_name,
            'full_path': branch_name,
            'depth': 1
        }
    
    # Create mapping
    mapping = {}
    stats = {
        'exact_match': 0,
        'similarity_match': 0,
        'top_level_fallback': 0,
        'no_match': 0
    }
    
    for old_path, old_info in old_nodes.items():
        # Check if path exists in current tree (unchanged)
        if old_path in current_nodes:
            mapping[old_path] = {
                'new_path': old_path,
                'confidence': 1.0,
                'match_type': 'unchanged'
            }
            stats['exact_match'] += 1
        else:
            # Find best match
            new_path, confidence, match_type = find_best_match(
                old_path, 
                old_info['name'], 
                current_nodes
            )
            
            if new_path:
                mapping[old_path] = {
                    'new_path': new_path,
                    'confidence': confidence,
                    'match_type': match_type
                }
                stats[match_type] += 1
            else:
                mapping[old_path] = {
                    'new_path': None,
                    'confidence': 0.0,
                    'match_type': 'no_match'
                }
                stats['no_match'] += 1
    
    return mapping, stats

def main():
    print("Creating node name mapping...")
    print("=" * 70)
    
    mapping, stats = create_mapping()
    
    # Save mapping
    output_file = 'data/node_name_mapping.json'
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✅ Mapping created: {output_file}")
    print(f"\nStatistics:")
    print(f"  Unchanged paths: {stats['exact_match']}")
    print(f"  Similarity matches: {stats['similarity_match']}")
    print(f"  Top-level fallbacks: {stats['top_level_fallback']}")
    print(f"  No matches: {stats['no_match']}")
    print(f"  Total: {sum(stats.values())}")
    
    # Show some examples
    print(f"\n📋 Sample mappings:")
    print("-" * 70)
    
    count = 0
    for old_path, info in mapping.items():
        if info['match_type'] == 'similarity_match' and count < 10:
            print(f"OLD: {old_path}")
            print(f"NEW: {info['new_path']}")
            print(f"Confidence: {info['confidence']:.2f} ({info['match_type']})")
            print()
            count += 1
    
    # Show no matches
    no_matches = [old for old, info in mapping.items() if info['match_type'] == 'no_match']
    if no_matches:
        print(f"\n⚠️  Paths with no match ({len(no_matches)}):")
        for path in no_matches[:10]:
            print(f"  - {path}")

if __name__ == '__main__':
    main()
