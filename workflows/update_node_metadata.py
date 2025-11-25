"""
Interactive script to review and update node metadata.
Helps you:
- Review all nodes in the tree
- Identify new auto-generated nodes
- Update montage_lead and investment_status
- Add descriptions and thesis
"""

import json
from datetime import datetime

def load_tree(filepath: str):
    """Load the taste tree JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_tree(tree, filepath: str):
    """Save the taste tree JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)

def collect_all_nodes(tree_dict, path=None, nodes=None):
    """Collect all nodes with their paths and metadata."""
    if path is None:
        path = []
    if nodes is None:
        nodes = []
    
    for key, value in tree_dict.items():
        if key == 'meta':
            continue
        
        current_path = path + [key]
        path_str = ' > '.join(current_path)
        meta = value.get('meta', {})
        children = value.get('children', {})
        
        nodes.append({
            'path': path_str,
            'path_list': current_path,
            'name': key,
            'depth': len(current_path),
            'meta': meta,
            'has_children': len(children) > 0,
            'num_children': len(children)
        })
        
        # Recurse into children
        if children:
            collect_all_nodes(children, current_path, nodes)
    
    return nodes

def filter_nodes(nodes, filter_type='all'):
    """Filter nodes based on criteria."""
    if filter_type == 'new':
        return [n for n in nodes if n['meta'].get('investment_status') == 'New']
    elif filter_type == 'no_lead':
        return [n for n in nodes if not n['meta'].get('montage_lead')]
    elif filter_type == 'no_status':
        return [n for n in nodes if not n['meta'].get('investment_status')]
    elif filter_type == 'leaf':
        return [n for n in nodes if not n['has_children']]
    elif filter_type == 'branch':
        return [n for n in nodes if n['has_children']]
    else:
        return nodes

def display_node_summary(node, index=None):
    """Display a formatted summary of a node."""
    prefix = f"[{index}] " if index is not None else ""
    
    print(f"\n{prefix}{'='*70}")
    print(f"📍 Path: {node['path']}")
    print(f"   Depth: {node['depth']} | Children: {node['num_children']}")
    
    meta = node['meta']
    
    # Show key metadata
    status = meta.get('investment_status', '❌ Not set')
    lead = meta.get('montage_lead', '❌ Not set')
    description = meta.get('description', '❌ Not set')
    
    # Highlight new nodes
    if status == 'New':
        print(f"   🆕 Investment Status: {status} (AUTO-GENERATED)")
    else:
        print(f"   💼 Investment Status: {status}")
    
    print(f"   👤 Montage Lead: {lead}")
    
    if description and description != '❌ Not set':
        desc_preview = description[:100] + '...' if len(description) > 100 else description
        print(f"   📝 Description: {desc_preview}")
    else:
        print(f"   📝 Description: {description}")
    
    # Show other metadata if present
    if meta.get('interest'):
        interest_preview = meta['interest'][:80] + '...' if len(meta['interest']) > 80 else meta['interest']
        print(f"   💡 Interest: {interest_preview}")
    
    if meta.get('portfolio'):
        portfolio = meta['portfolio']
        if isinstance(portfolio, list):
            print(f"   📊 Portfolio: {len(portfolio)} companies")
        else:
            print(f"   📊 Portfolio: Present")
    
    if meta.get('pipeline'):
        pipeline_preview = str(meta['pipeline'])[:80] + '...' if len(str(meta['pipeline'])) > 80 else str(meta['pipeline'])
        print(f"   🔄 Pipeline: {pipeline_preview}")
    
    if meta.get('recent_news'):
        news_preview = meta['recent_news'][:80] + '...' if len(meta['recent_news']) > 80 else meta['recent_news']
        print(f"   📰 Recent News: {news_preview}")

def update_node_metadata(tree_dict, path_list, updates):
    """Update metadata for a specific node."""
    current = tree_dict
    
    # Navigate to the node
    for i, step in enumerate(path_list):
        if i == 0:
            if step not in current:
                return False
            current = current[step]
        else:
            if 'children' not in current or step not in current['children']:
                return False
            current = current['children'][step]
    
    # Update metadata
    if 'meta' not in current:
        current['meta'] = {}
    
    for key, value in updates.items():
        if value is not None and value != '':
            current['meta'][key] = value
    
    # Update last_updated timestamp
    current['meta']['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    
    return True

def interactive_update_session(tree_dict, nodes, filter_type='all'):
    """Run an interactive session to update node metadata."""
    filtered_nodes = filter_nodes(nodes, filter_type)
    
    if not filtered_nodes:
        print(f"\n✅ No nodes found matching filter: {filter_type}")
        return tree_dict, 0
    
    print(f"\n{'='*70}")
    print(f"📋 INTERACTIVE METADATA UPDATE SESSION")
    print(f"{'='*70}")
    print(f"Filter: {filter_type}")
    print(f"Nodes to review: {len(filtered_nodes)}")
    print(f"\nCommands:")
    print(f"  - Press ENTER to skip")
    print(f"  - Type 'q' to quit")
    print(f"  - Type 's' to show full metadata")
    print(f"  - Type values to update")
    
    updates_made = 0
    
    for i, node in enumerate(filtered_nodes, 1):
        display_node_summary(node, i)
        
        print(f"\n{'─'*70}")
        
        # Get current values
        current_status = node['meta'].get('investment_status', '')
        current_lead = node['meta'].get('montage_lead', '')
        current_desc = node['meta'].get('description', '')
        
        # Prompt for updates
        print(f"\n🔧 Update this node? (Enter to skip, 'q' to quit, 's' to show full meta)")
        action = input("   Action: ").strip().lower()
        
        if action == 'q':
            print("\n👋 Exiting update session...")
            break
        elif action == 's':
            print(f"\n📄 Full metadata:")
            for key, value in node['meta'].items():
                print(f"   {key}: {value}")
            print()
            action = input("   Continue with update? (y/n): ").strip().lower()
            if action != 'y':
                continue
        elif action == '':
            continue
        
        updates = {}
        
        # Update investment status
        print(f"\n💼 Investment Status (current: {current_status})")
        print(f"   Options: High, Moderate, Medium, Low, New, EXCLUDE")
        new_status = input(f"   New value (or Enter to keep): ").strip()
        if new_status:
            updates['investment_status'] = new_status
        
        # Update montage lead
        print(f"\n👤 Montage Lead (current: {current_lead})")
        new_lead = input(f"   New value (or Enter to keep): ").strip()
        if new_lead:
            updates['montage_lead'] = new_lead
        
        # Update description (optional)
        print(f"\n📝 Description (current: {current_desc[:50]}...)")
        update_desc = input(f"   Update description? (y/n): ").strip().lower()
        if update_desc == 'y':
            new_desc = input(f"   New description: ").strip()
            if new_desc:
                updates['description'] = new_desc
        
        # Apply updates
        if updates:
            success = update_node_metadata(tree_dict, node['path_list'], updates)
            if success:
                print(f"   ✅ Updated successfully!")
                updates_made += 1
            else:
                print(f"   ❌ Failed to update node")
        else:
            print(f"   ⏭️  No changes made")
    
    return tree_dict, updates_made

def bulk_update_session(tree_dict, nodes):
    """Bulk update session for common operations."""
    print(f"\n{'='*70}")
    print(f"🔄 BULK UPDATE SESSION")
    print(f"{'='*70}")
    
    print(f"\nAvailable bulk operations:")
    print(f"  1. Set montage_lead for all nodes without one")
    print(f"  2. Change investment_status for all 'New' nodes")
    print(f"  3. Set default investment_status for nodes without one")
    print(f"  4. Add description to all leaf nodes without one")
    print(f"  5. Cancel")
    
    choice = input(f"\nSelect operation (1-5): ").strip()
    
    updates_made = 0
    
    if choice == '1':
        lead_name = input("Enter montage_lead name: ").strip()
        if lead_name:
            nodes_to_update = filter_nodes(nodes, 'no_lead')
            print(f"\nWill update {len(nodes_to_update)} nodes. Confirm? (y/n): ", end='')
            if input().strip().lower() == 'y':
                for node in nodes_to_update:
                    if update_node_metadata(tree_dict, node['path_list'], {'montage_lead': lead_name}):
                        updates_made += 1
                print(f"✅ Updated {updates_made} nodes")
    
    elif choice == '2':
        new_nodes = filter_nodes(nodes, 'new')
        print(f"\nFound {len(new_nodes)} nodes with status 'New'")
        new_status = input("Set new investment_status for these nodes: ").strip()
        if new_status:
            print(f"Will update {len(new_nodes)} nodes to '{new_status}'. Confirm? (y/n): ", end='')
            if input().strip().lower() == 'y':
                for node in new_nodes:
                    if update_node_metadata(tree_dict, node['path_list'], {'investment_status': new_status}):
                        updates_made += 1
                print(f"✅ Updated {updates_made} nodes")
    
    elif choice == '3':
        nodes_no_status = filter_nodes(nodes, 'no_status')
        print(f"\nFound {len(nodes_no_status)} nodes without investment_status")
        default_status = input("Set default investment_status: ").strip()
        if default_status:
            print(f"Will update {len(nodes_no_status)} nodes to '{default_status}'. Confirm? (y/n): ", end='')
            if input().strip().lower() == 'y':
                for node in nodes_no_status:
                    if update_node_metadata(tree_dict, node['path_list'], {'investment_status': default_status}):
                        updates_made += 1
                print(f"✅ Updated {updates_made} nodes")
    
    elif choice == '4':
        leaf_nodes = filter_nodes(nodes, 'leaf')
        nodes_no_desc = [n for n in leaf_nodes if not n['meta'].get('description')]
        print(f"\nFound {len(nodes_no_desc)} leaf nodes without description")
        print("This will set a generic description. Continue? (y/n): ", end='')
        if input().strip().lower() == 'y':
            for node in nodes_no_desc:
                desc = f"Investment opportunities in {node['name']}"
                if update_node_metadata(tree_dict, node['path_list'], {'description': desc}):
                    updates_made += 1
            print(f"✅ Updated {updates_made} nodes")
    
    return tree_dict, updates_made

def main():
    input_file = '/Users/matthildur/Desktop/monty2.0/data/taste_tree.json'
    backup_file = f'/Users/matthildur/Desktop/monty2.0/data/taste_tree_backup_metadata_update_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    print("Loading taste tree...")
    tree = load_tree(input_file)
    
    # Create backup
    print(f"Creating backup at {backup_file}...")
    save_tree(tree, backup_file)
    
    # Collect all nodes
    print("Collecting nodes...")
    all_nodes = []
    for top_level_key, top_level_value in tree.items():
        if isinstance(top_level_value, dict) and 'children' in top_level_value:
            collect_all_nodes(top_level_value['children'], [top_level_key], all_nodes)
    
    print(f"Found {len(all_nodes)} total nodes")
    
    # Show summary statistics
    new_nodes = filter_nodes(all_nodes, 'new')
    no_lead = filter_nodes(all_nodes, 'no_lead')
    no_status = filter_nodes(all_nodes, 'no_status')
    
    print(f"\n{'='*70}")
    print(f"📊 TREE METADATA SUMMARY")
    print(f"{'='*70}")
    print(f"Total nodes: {len(all_nodes)}")
    print(f"🆕 New auto-generated nodes: {len(new_nodes)}")
    print(f"👤 Nodes without montage_lead: {len(no_lead)}")
    print(f"💼 Nodes without investment_status: {len(no_status)}")
    
    # Main menu
    total_updates = 0  # Move outside the loop
    while True:
        print(f"\n{'='*70}")
        print(f"MAIN MENU")
        print(f"{'='*70}")
        print(f"1. Review and update NEW nodes")
        print(f"2. Review and update nodes without montage_lead")
        print(f"3. Review and update nodes without investment_status")
        print(f"4. Review and update ALL nodes")
        print(f"5. Review leaf nodes only")
        print(f"6. Bulk update operations")
        print(f"7. Save and exit")
        print(f"8. Exit without saving")
        
        choice = input(f"\nSelect option (1-8): ").strip()
        
        if choice == '1':
            tree, updates = interactive_update_session(tree, all_nodes, 'new')
            total_updates += updates
        elif choice == '2':
            tree, updates = interactive_update_session(tree, all_nodes, 'no_lead')
            total_updates += updates
        elif choice == '3':
            tree, updates = interactive_update_session(tree, all_nodes, 'no_status')
            total_updates += updates
        elif choice == '4':
            tree, updates = interactive_update_session(tree, all_nodes, 'all')
            total_updates += updates
        elif choice == '5':
            tree, updates = interactive_update_session(tree, all_nodes, 'leaf')
            total_updates += updates
        elif choice == '6':
            tree, updates = bulk_update_session(tree, all_nodes)
            total_updates += updates
        elif choice == '7':
            print(f"\n💾 Saving changes...")
            save_tree(tree, input_file)
            print(f"✅ Saved successfully!")
            print(f"📊 Total updates made: {total_updates}")
            print(f"💾 Backup saved at: {backup_file}")
            break
        elif choice == '8':
            print(f"\n👋 Exiting without saving...")
            break
        else:
            print(f"❌ Invalid option")

if __name__ == '__main__':
    main()
