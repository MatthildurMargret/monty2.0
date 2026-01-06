import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import shutil

load_dotenv()

# Try to import supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  supabase package not installed. Install with: pip install supabase")


def get_supabase_client():
    """Create and return a Supabase client."""
    if not SUPABASE_AVAILABLE:
        raise ImportError(
            "supabase package is not installed. Install it with: pip install supabase"
        )
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url:
        raise ValueError("SUPABASE_URL must be set in your .env file")
    
    if not supabase_key:
        raise ValueError("SUPABASE_KEY must be set in your .env file")
    
    return create_client(supabase_url, supabase_key)


def load_taste_tree_from_supabase():
    """Load the latest version of taste_tree from Supabase."""
    supabase = get_supabase_client()
    
    # Get the latest version of the tree
    response = supabase.table("taste_tree")\
        .select("data, version, created_at")\
        .order("created_at", desc=True)\
        .limit(1)\
        .execute()
    
    if not response.data:
        raise ValueError("No taste_tree data found in Supabase")
    
    record = response.data[0]
    return record['data'], record.get('version'), record.get('created_at')


def load_local_taste_tree(file_path=None):
    """Load taste tree from local JSON file."""
    if file_path is None:
        file_path = os.getenv("TASTE_TREE_PATH", "data/taste_tree.json")
    
    if not os.path.isabs(file_path):
        project_root = Path(__file__).parent.parent
        file_path = project_root / file_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def deep_merge_dict(base_dict, update_dict, path="root"):
    """
    Deep merge two dictionaries, with update_dict taking precedence for conflicts.
    
    Strategy:
    - If both are dicts, recursively merge
    - If both have 'meta' and 'children', merge meta fields intelligently
    - For meta fields: prefer non-empty values from update_dict, but keep base if update is empty
    - For children: merge recursively
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key not in result:
            # New key from update - add it
            result[key] = value
        elif isinstance(result[key], dict) and isinstance(value, dict):
            # Both are dicts - merge recursively
            if key == "meta":
                # Special handling for meta: merge fields, prefer non-empty from update
                merged_meta = result[key].copy()
                for meta_key, meta_value in value.items():
                    if meta_key in merged_meta:
                        # If update has a non-empty value, use it; otherwise keep base
                        if meta_value and (not isinstance(meta_value, str) or meta_value.strip()):
                            merged_meta[meta_key] = meta_value
                        # If base is empty and update is empty, keep base (no change)
                    else:
                        # New meta field from update
                        merged_meta[meta_key] = meta_value
                result[key] = merged_meta
            elif key == "children":
                # Merge children recursively
                merged_children = {}
                # Start with base children
                for child_key, child_value in result[key].items():
                    merged_children[child_key] = child_value.copy() if isinstance(child_value, dict) else child_value
                
                # Merge in update children
                for child_key, child_value in value.items():
                    if child_key in merged_children:
                        # Recursively merge existing child
                        merged_children[child_key] = deep_merge_dict(
                            merged_children[child_key], 
                            child_value, 
                            f"{path}.{key}.{child_key}"
                        )
                    else:
                        # New child from update
                        merged_children[child_key] = child_value
                
                result[key] = merged_children
            else:
                # Regular dict merge
                result[key] = deep_merge_dict(result[key], value, f"{path}.{key}")
        else:
            # Different types or update is not a dict - update takes precedence
            # But only if update value is not empty/None
            if value is not None and (not isinstance(value, str) or value.strip()):
                result[key] = value
    
    return result


def merge_taste_trees(local_tree, supabase_tree):
    """
    Merge local and Supabase taste trees.
    
    Returns merged tree with Supabase changes taking precedence for conflicts.
    """
    merged = {}
    
    # Get all top-level categories from both trees
    all_categories = set(local_tree.keys()) | set(supabase_tree.keys())
    
    print(f"\n📊 Merging {len(all_categories)} top-level categories...")
    
    for category in all_categories:
        if category in local_tree and category in supabase_tree:
            # Both have this category - merge recursively
            print(f"  🔀 Merging: {category}")
            merged[category] = deep_merge_dict(local_tree[category], supabase_tree[category], category)
        elif category in supabase_tree:
            # Only in Supabase - add it
            print(f"  ➕ Adding from Supabase: {category}")
            merged[category] = supabase_tree[category]
        else:
            # Only in local - keep it
            print(f"  📌 Keeping from local: {category}")
            merged[category] = local_tree[category]
    
    return merged


def save_taste_tree(data, file_path=None, create_backup=True):
    """Save taste tree to file, optionally creating a backup first."""
    if file_path is None:
        file_path = os.getenv("TASTE_TREE_PATH", "data/taste_tree.json")
    
    if not os.path.isabs(file_path):
        project_root = Path(__file__).parent.parent
        file_path = project_root / file_path
    
    # Create backup if requested
    if create_backup and file_path.exists():
        backup_path = file_path.parent / f"{file_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(file_path, backup_path)
        print(f"💾 Created backup: {backup_path}")
    
    # Save merged tree
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved merged tree to: {file_path}")


def compare_trees(local_tree, supabase_tree):
    """Compare two trees and report differences."""
    differences = {
        "new_in_supabase": [],
        "new_in_local": [],
        "modified": []
    }
    
    all_categories = set(local_tree.keys()) | set(supabase_tree.keys())
    
    for category in all_categories:
        if category not in local_tree:
            differences["new_in_supabase"].append(category)
        elif category not in supabase_tree:
            differences["new_in_local"].append(category)
        else:
            # Both exist - check if they differ
            local_json = json.dumps(local_tree[category], sort_keys=True)
            supabase_json = json.dumps(supabase_tree[category], sort_keys=True)
            if local_json != supabase_json:
                differences["modified"].append(category)
    
    return differences


def collect_all_nodes(tree, path="", nodes=None):
    """Recursively collect all nodes from the tree with their paths."""
    if nodes is None:
        nodes = []
    
    for key, value in tree.items():
        current_path = f"{path} > {key}" if path else key
        if isinstance(value, dict) and "meta" in value:
            nodes.append({
                "path": current_path,
                "node": value,
                "meta": value.get("meta", {}),
                "children": value.get("children", {})
            })
            # Recursively collect children
            if "children" in value and value["children"]:
                collect_all_nodes(value["children"], current_path, nodes)
    
    return nodes


def verify_merge(merged_tree, supabase_tree, local_tree):
    """
    Verify that all Supabase metadata was correctly merged.
    
    Returns a verification report with:
    - Total nodes checked
    - Nodes with metadata updates
    - Nodes that match Supabase exactly
    - Nodes with missing Supabase metadata
    - Detailed field-level changes
    """
    print("\n" + "=" * 70)
    print("🔍 VERIFICATION: Checking merge completeness...")
    print("=" * 70)
    
    # Collect all nodes from each tree
    supabase_nodes = collect_all_nodes(supabase_tree)
    merged_nodes = collect_all_nodes(merged_tree)
    
    # Create path lookup for merged nodes
    merged_by_path = {node["path"]: node for node in merged_nodes}
    
    verification_report = {
        "total_supabase_nodes": len(supabase_nodes),
        "total_merged_nodes": len(merged_nodes),
        "nodes_verified": 0,
        "nodes_with_updates": [],
        "nodes_missing_metadata": [],
        "field_changes": {}
    }
    
    # Check each Supabase node
    for supabase_node in supabase_nodes:
        path = supabase_node["path"]
        supabase_meta = supabase_node["meta"]
        
        if path not in merged_by_path:
            verification_report["nodes_missing_metadata"].append({
                "path": path,
                "reason": "Node not found in merged tree"
            })
            continue
        
        merged_node = merged_by_path[path]
        merged_meta = merged_node["meta"]
        
        # Check each metadata field from Supabase
        updates = []
        for field, supabase_value in supabase_meta.items():
            merged_value = merged_meta.get(field)
            
            # Normalize for comparison (handle empty strings, None, etc.)
            supabase_normalized = supabase_value if supabase_value and (not isinstance(supabase_value, str) or supabase_value.strip()) else None
            merged_normalized = merged_value if merged_value and (not isinstance(merged_value, str) or merged_value.strip()) else None
            
            # Check if Supabase value was properly merged
            if supabase_normalized is not None:
                if merged_normalized != supabase_normalized:
                    # Field should have been updated but wasn't
                    updates.append({
                        "field": field,
                        "supabase_value": str(supabase_value)[:100] + "..." if len(str(supabase_value)) > 100 else str(supabase_value),
                        "merged_value": str(merged_value)[:100] + "..." if merged_value and len(str(merged_value)) > 100 else str(merged_value),
                        "status": "❌ NOT MERGED"
                    })
                else:
                    # Field was correctly merged
                    if field not in verification_report["field_changes"]:
                        verification_report["field_changes"][field] = {"updated": 0, "unchanged": 0}
                    verification_report["field_changes"][field]["updated"] += 1
            elif merged_normalized is not None:
                # Supabase had empty/None, merged kept local value (this is correct)
                if field not in verification_report["field_changes"]:
                    verification_report["field_changes"][field] = {"updated": 0, "unchanged": 0}
                verification_report["field_changes"][field]["unchanged"] += 1
        
        if updates:
            verification_report["nodes_with_updates"].append({
                "path": path,
                "updates": updates
            })
        else:
            verification_report["nodes_verified"] += 1
    
    # Print verification summary
    print(f"\n📊 Verification Summary:")
    print(f"   Total nodes in Supabase: {verification_report['total_supabase_nodes']}")
    print(f"   Total nodes in merged: {verification_report['total_merged_nodes']}")
    print(f"   ✅ Nodes verified (metadata matches): {verification_report['nodes_verified']}")
    print(f"   ⚠️  Nodes with potential issues: {len(verification_report['nodes_with_updates'])}")
    print(f"   ❌ Nodes missing from merged: {len(verification_report['nodes_missing_metadata'])}")
    
    # Show field-level statistics
    if verification_report["field_changes"]:
        print(f"\n📝 Metadata Field Updates:")
        for field, stats in sorted(verification_report["field_changes"].items()):
            total = stats["updated"] + stats["unchanged"]
            print(f"   {field}:")
            print(f"      - Updated from Supabase: {stats['updated']}")
            print(f"      - Kept local (Supabase empty): {stats['unchanged']}")
            print(f"      - Total nodes with this field: {total}")
    
    # Show nodes with issues (limit to first 10)
    if verification_report["nodes_with_updates"]:
        print(f"\n⚠️  Nodes with metadata merge issues (showing first 10):")
        for node_info in verification_report["nodes_with_updates"][:10]:
            print(f"\n   Path: {node_info['path']}")
            for update in node_info["updates"]:
                print(f"      {update['field']}: {update['status']}")
                if update.get("supabase_value"):
                    print(f"         Supabase: {update['supabase_value'][:80]}...")
                if update.get("merged_value"):
                    print(f"         Merged:   {update['merged_value'][:80]}...")
        
        if len(verification_report["nodes_with_updates"]) > 10:
            print(f"\n   ... and {len(verification_report['nodes_with_updates']) - 10} more nodes with issues")
    
    # Show missing nodes
    if verification_report["nodes_missing_metadata"]:
        print(f"\n❌ Nodes from Supabase not found in merged tree:")
        for node_info in verification_report["nodes_missing_metadata"][:10]:
            print(f"   - {node_info['path']}: {node_info['reason']}")
    
    # Final validation: Check that all non-empty Supabase metadata fields are in merged
    print(f"\n🔬 Final Validation: Checking metadata field completeness...")
    validation_passed = True
    validation_issues = []
    
    for supabase_node in supabase_nodes:
        path = supabase_node["path"]
        if path not in merged_by_path:
            continue
        
        merged_node = merged_by_path[path]
        supabase_meta = supabase_node["meta"]
        merged_meta = merged_node["meta"]
        
        for field, supabase_value in supabase_meta.items():
            # Only check non-empty values from Supabase
            if supabase_value and (not isinstance(supabase_value, str) or supabase_value.strip()):
                merged_value = merged_meta.get(field)
                # Check if values match (allowing for type differences in JSON serialization)
                if merged_value != supabase_value:
                    # Try JSON comparison for complex types
                    try:
                        if json.dumps(merged_value, sort_keys=True) != json.dumps(supabase_value, sort_keys=True):
                            validation_passed = False
                            validation_issues.append({
                                "path": path,
                                "field": field,
                                "issue": "Value mismatch"
                            })
                    except (TypeError, ValueError):
                        # If JSON comparison fails, do string comparison
                        if str(merged_value) != str(supabase_value):
                            validation_passed = False
                            validation_issues.append({
                                "path": path,
                                "field": field,
                                "issue": "Value mismatch"
                            })
    
    if validation_issues:
        print(f"   ⚠️  Found {len(validation_issues)} validation issues")
        if len(validation_issues) <= 5:
            for issue in validation_issues:
                print(f"      - {issue['path']}: {issue['field']} - {issue['issue']}")
        else:
            print(f"      (showing first 5 of {len(validation_issues)} issues)")
            for issue in validation_issues[:5]:
                print(f"      - {issue['path']}: {issue['field']} - {issue['issue']}")
    else:
        print(f"   ✅ All Supabase metadata fields correctly merged")
    
    # Overall status
    print("\n" + "=" * 70)
    if (verification_report["nodes_verified"] == verification_report["total_supabase_nodes"] and 
        len(verification_report["nodes_with_updates"]) == 0 and
        len(verification_report["nodes_missing_metadata"]) == 0 and
        validation_passed):
        print("✅ VERIFICATION PASSED: All Supabase metadata correctly merged!")
    else:
        print("⚠️  VERIFICATION WARNING: Some issues detected (see details above)")
        if not validation_passed:
            print("   ⚠️  Final validation found mismatches - please review")
    print("=" * 70)
    
    return verification_report


def main(dry_run=False):
    print("=" * 70)
    print("Merge Taste Tree from Supabase")
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be saved")
    print("=" * 70)
    
    # Load local tree
    print("\n📖 Loading local taste_tree.json...")
    try:
        local_tree = load_local_taste_tree()
        print(f"✅ Loaded local tree with {len(local_tree)} top-level categories")
    except Exception as e:
        print(f"❌ Error loading local tree: {e}")
        return 1
    
    # Load from Supabase
    print("\n📥 Fetching latest version from Supabase...")
    try:
        supabase_tree, version, created_at = load_taste_tree_from_supabase()
        print(f"✅ Loaded Supabase tree (version: {version}, created: {created_at})")
        print(f"   Supabase tree has {len(supabase_tree)} top-level categories")
    except Exception as e:
        print(f"❌ Error loading from Supabase: {e}")
        return 1
    
    # Compare trees
    print("\n🔍 Comparing trees...")
    differences = compare_trees(local_tree, supabase_tree)
    
    if differences["new_in_supabase"]:
        print(f"\n➕ New categories in Supabase ({len(differences['new_in_supabase'])}):")
        for cat in differences["new_in_supabase"]:
            print(f"   - {cat}")
    
    if differences["new_in_local"]:
        print(f"\n📌 Categories only in local ({len(differences['new_in_local'])}):")
        for cat in differences["new_in_local"]:
            print(f"   - {cat}")
    
    if differences["modified"]:
        print(f"\n🔀 Modified categories ({len(differences['modified'])}):")
        for cat in differences["modified"]:
            print(f"   - {cat}")
    
    if not any(differences.values()):
        print("\n✅ Trees are identical - no merge needed!")
        return 0
    
    # Merge trees
    print("\n🔀 Merging trees (Supabase changes take precedence)...")
    merged_tree = merge_taste_trees(local_tree, supabase_tree)
    print(f"✅ Merged tree has {len(merged_tree)} top-level categories")
    
    # Verify the merge
    verification_report = verify_merge(merged_tree, supabase_tree, local_tree)
    
    if dry_run:
        print("\n🔍 DRY RUN: Would save merged tree (use --apply to actually save)")
        return 0
    
    # Save merged tree
    print("\n💾 Saving merged tree...")
    try:
        save_taste_tree(merged_tree, create_backup=True)
        print("\n✨ Merge complete!")
        print("\n💡 Tip: Review the merged file and backup before committing.")
    except Exception as e:
        print(f"\n❌ Error saving merged tree: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    dry_run = "--apply" not in sys.argv
    if dry_run and len(sys.argv) > 1:
        print("💡 Use --apply flag to actually save the merged tree")
    exit(main(dry_run=dry_run))
