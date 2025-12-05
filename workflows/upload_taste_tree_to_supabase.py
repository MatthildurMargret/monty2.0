import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Try to import supabase client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  supabase package not installed. Install with: pip install supabase")


def load_taste_tree(file_path=None):
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


def upload_taste_tree(taste_tree_data: dict):
    """Upload taste tree using Supabase client."""
    if not SUPABASE_AVAILABLE:
        raise ImportError("supabase package is required. Install with: pip install supabase")
    
    supabase = get_supabase_client()
    
    # Check data size
    import sys
    data_size_mb = sys.getsizeof(json.dumps(taste_tree_data)) / (1024 * 1024)
    print(f"📊 Data size: {data_size_mb:.2f} MB")
    
    # Check current row count
    try:
        response = supabase.table("taste_tree").select("id", count="exact").execute()
        rows_before = response.count if hasattr(response, 'count') else len(response.data) if response.data else 0
        print(f"📊 Rows before insert: {rows_before}")
    except Exception as e:
        if "relation" in str(e).lower() or "does not exist" in str(e).lower():
            print("⚠️  Table 'taste_tree' does not exist yet.")
            print("   Please create it first using the SQL setup script in Supabase SQL Editor.")
            raise
        rows_before = 0
        print(f"⚠️  Could not get row count (table might be empty): {e}")
    
    # Prepare the data for insertion
    version = datetime.utcnow().isoformat()
    upload_data = {
        "data": taste_tree_data,
        "version": version,
        "updated_at": datetime.utcnow().isoformat()
    }
    
    print(f"📤 Inserting data into taste_tree table...")
    
    try:
        # Insert the data
        response = supabase.table("taste_tree").insert(upload_data).execute()
        
        print(f"✅ Insert successful!")
        
        # Verify the insert
        try:
            # Get updated count
            verify_response = supabase.table("taste_tree").select("id", count="exact").execute()
            rows_after = verify_response.count if hasattr(verify_response, 'count') else len(verify_response.data) if verify_response.data else 0
            print(f"📊 Total rows in table after insert: {rows_after}")
            
            if rows_after <= rows_before:
                print("⚠️  WARNING: Row count did not increase!")
                print("   This might indicate:")
                print("   1. Row Level Security (RLS) is blocking the insert")
                print("   2. The insert was rolled back")
                print("   3. A trigger prevented the insert")
            else:
                print(f"✅ Row count increased from {rows_before} to {rows_after}")
            
            # Get the latest record
            latest_response = supabase.table("taste_tree")\
                .select("id, version, created_at")\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if latest_response.data:
                latest = latest_response.data[0]
                print(f"✅ Latest record: ID={latest['id']}, Version={latest['version']}, Created={latest['created_at']}")
            else:
                print("⚠️  Warning: Could not retrieve latest record")
                
        except Exception as e:
            print(f"⚠️  Could not verify insert: {e}")
        
        print(f"✅ Uploaded taste tree. Version: {version}")
        return response
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error uploading: {error_msg}")
        
        # Check for common errors
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower():
            print("\n💡 Table 'taste_tree' does not exist.")
            print("   Run the SQL setup script in Supabase SQL Editor:")
            print("   workflows/supabase_taste_tree_setup.sql")
        elif "permission" in error_msg.lower() or "policy" in error_msg.lower() or "row level security" in error_msg.lower():
            print("\n💡 Row Level Security (RLS) might be blocking the insert.")
            print("   Options:")
            print("   1. Disable RLS on the table (temporarily)")
            print("   2. Create an INSERT policy")
            print("   3. Use service_role key instead of anon key")
        elif "payload too large" in error_msg.lower() or "too large" in error_msg.lower():
            print("\n💡 The data might be too large.")
            print("   Consider splitting the data or checking Supabase limits.")
        
        raise


def main():
    print("=" * 70)
    print("Taste Tree Upload to Supabase")
    print("=" * 70)
    print("\n📖 Loading JSON...")
    data = load_taste_tree()
    print(f"✅ Loaded JSON with {len(data)} top-level categories")

    print("\n📤 Uploading to Supabase...")
    try:
        upload_taste_tree(data)
        print("\n✨ Upload complete!")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()
