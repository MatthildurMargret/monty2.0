import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_connection():
    """
    Create a connection to the PostgreSQL database using environment variables.
    
    Returns:
        connection: A PostgreSQL database connection
    """
    try:
        db_url = os.getenv("DATABASE_URL")
        connection = psycopg2.connect(db_url)
        return connection
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None


def insert_search_results(df, table_name="search_list", stealth_mode=False):
    """
    Insert search results from a pandas DataFrame into the PostgreSQL database.
    If a profile_url already exists, update the record with the new data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing search results
        table_name (str): Name of the table to insert into (default: "search_list")
        stealth_mode (bool): If True, insert into the stealth version of the table
        
    Returns:
        bool: True if insertion was successful, False otherwise
    """
    # If stealth mode is enabled, use stealth_search_list table
    if stealth_mode:
        table_name = "stealth_search_list"
    
    # Get database connection
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Ensure all required columns exist in the dataframe
        required_columns = ['name', 'title', 'company_name', 'profile_url', 'source', 'company_url']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None  # Add missing columns with NULL values
        
        # Remove duplicates within the batch to avoid ON CONFLICT issues
        original_count = len(df)
        df = df.drop_duplicates(subset=['profile_url'], keep='first')
        if len(df) < original_count:
            print(f"Removed {original_count - len(df)} duplicate profile_urls from batch")
        
        # Prepare data for insertion
        data = [
            (
                row.get('name', None),
                row.get('profile_url', None),
                row.get('title', None),
                row.get('source', None),
                row.get('company_name', None),
                row.get('company_url', None)
            )
            for _, row in df.to_dict('index').items()
        ]
        
        # Define the SQL query with ON CONFLICT to update existing records
        insert_query = f"""
        INSERT INTO {table_name} 
        (name, profile_url, title, source, company_name, company_url)
        VALUES %s
        ON CONFLICT (profile_url) 
        DO UPDATE SET
            name = EXCLUDED.name,
            title = EXCLUDED.title,
            source = EXCLUDED.source,
            company_name = EXCLUDED.company_name,
            company_url = EXCLUDED.company_url
        """
        
        # Execute the query
        execute_values(cursor, insert_query, data)
        
        # Commit the transaction
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error inserting data into {table_name}: {e}")
        conn.rollback()
        return False
        
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def insert_producthunt_products(products):
    """
    Inserts a list of Product Hunt products into the database.
    Avoids inserting duplicates based on the product URL.
    """
    conn = get_db_connection()
    if not conn:
        print("DB connection failed.")
        return

    cursor = conn.cursor()

    # Convert the data into the correct format
    records = []
    for p in products:
        try:
            record = (
                str(p['name']),
                str(p['tagline']),
                str(p['url']),
                int(p['votes']),
                str(p['image_url']),
                str(p['scrape_date'])
            )
            records.append(record)
        except Exception as e:
            print(f"Error processing record: {e}")
            print(f"Problematic record: {p}")
            continue

    query = """
    INSERT INTO product_hunt_products (name, tagline, url, votes, image_url, scrape_date)
    VALUES %s
    ON CONFLICT (url) DO NOTHING;
    """

    try:
        execute_values(cursor, query, records)
        conn.commit()
    except Exception as e:
        print(f"Error inserting products: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def insert_producthunt_founder(founder_dict):
    """
    Inserts a founder profile into product_hunt_founders table.
    Avoids duplicates using profile_url + product_id.
    """
    conn = get_db_connection()
    if not conn:
        print("DB connection failed.")
        return

    cursor = conn.cursor()

    try:
        # Find the matching product_id
        cursor.execute(
            "SELECT id FROM product_hunt_products WHERE name = %s",
            (founder_dict['company_name'],)
        )
        result = cursor.fetchone()

        if not result:
            return

        product_id = result[0]

        cursor.execute(
            """
            INSERT INTO product_hunt_founders (name, profile_url, product_id, source)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
            """,
            (
                founder_dict['name'],
                founder_dict['profile_url'],
                product_id,
                founder_dict.get('source', 'Product Hunt')
            )
        )

        conn.commit()
    except Exception as e:
        print(f"Error inserting founder: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def create_tables_if_not_exist(table_name="search_list"):
    """
    Create the necessary tables if they don't already exist in the database.
    Ensures that profile_url has a unique constraint for the ON CONFLICT clause to work.
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Create search_list table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            name TEXT,
            profile_url TEXT UNIQUE,
            title TEXT,
            source TEXT,
            company_name TEXT,
            company_url TEXT
        )
        """)
        
        # Create stealth_search_list table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stealth_search_list (
            id SERIAL PRIMARY KEY,
            name TEXT,
            profile_url TEXT UNIQUE,
            title TEXT,
            source TEXT,
            company_name TEXT,
            company_url TEXT
        )
        """)
        
        # Add unique constraint on profile_url if it doesn't exist (for older tables)
        try:
            cursor.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = '{table_name}_profile_url_key' 
                    AND conrelid = '{table_name}'::regclass
                ) THEN
                    ALTER TABLE {table_name} ADD CONSTRAINT {table_name}_profile_url_key UNIQUE (profile_url);
                END IF;
                
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = 'stealth_search_list_profile_url_key' 
                    AND conrelid = 'stealth_search_list'::regclass
                ) THEN
                    ALTER TABLE stealth_search_list ADD CONSTRAINT stealth_search_list_profile_url_key UNIQUE (profile_url);
                END IF;
            END
            $$;
            """)
        except Exception as e:
            print(f"Warning: Could not add unique constraints: {e}")
            # Continue anyway, as the CREATE TABLE statements already include UNIQUE
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
        return False
        
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def clean_linkedin_url(url):
    """
    Clean LinkedIn profile URLs by removing tracking parameters.
    
    Args:
        url (str): LinkedIn profile URL, potentially with tracking parameters
        
    Returns:
        str: Cleaned LinkedIn profile URL
    """
    if not url:
        return url
    
    # Remove tracking parameters
    url = re.sub(r'\?.*$', '', url)  # Remove query parameters
    
    # Convert Sales Navigator URLs to standard LinkedIn profile URLs
    if 'linkedin.com/sales/people' in url:
        # Extract the profile ID from the URL
        match = re.search(r'/people/([^/]+)', url)
        if match:
            profile_id = match.group(1)
            url = f"https://www.linkedin.com/in/{profile_id}"
    
    return url

def clean_column_name(column_name):
    """
    Clean column names to be compatible with PostgreSQL.
    Converts to lowercase and replaces spaces with underscores.
    """
    if not column_name:
        return "unnamed_column"
    
    # Convert to lowercase and replace spaces with underscores
    cleaned = str(column_name).lower().replace(' ', '_')
    
    # Remove special characters except underscores
    cleaned = re.sub(r'[^a-z0-9_]', '', cleaned)
    
    # Ensure it doesn't start with a number
    if cleaned and cleaned[0].isdigit():
        cleaned = 'col_' + cleaned
    
    # Handle empty strings or strings that become empty after cleaning
    if not cleaned:
        return "unnamed_column"
    
    return cleaned

def clean_value(value, column_name=None):
    """
    Clean values to be compatible with PostgreSQL.
    Handles None values, converts objects to strings, and truncates long strings.
    For problematic fields like post_data with complex Unicode, returns None instead of trying to fix.
    """
    if value is None:
        return None
    
    # Handle empty strings for numeric fields
    if value == "" and column_name and any(numeric_suffix in column_name.lower() for numeric_suffix in 
                                         ['_score', 'likelihood', 'years_of_experience']):
        return None
    
    # Skip problematic fields entirely - just return None for post_data field
    # This is the safest approach to avoid database errors with complex Unicode
    if column_name == 'post_data':
        return None
    
    # Convert to string if not already
    if not isinstance(value, (str, int, float, bool, list, dict)):
        value = str(value)
    
    # Handle lists and dictionaries by converting to JSON string
    if isinstance(value, (list, dict)):
        import json
        try:
            # For fields containing text content that might have problematic Unicode
            if isinstance(value, list) and any('content' in item for item in value if isinstance(item, dict)):
                # Skip complex content fields that might have problematic Unicode
                for item in value:
                    if isinstance(item, dict) and 'content' in item:
                        # Check if content has non-ASCII characters
                        if isinstance(item['content'], str) and any(ord(c) > 127 for c in item['content']):
                            # Skip complex Unicode content
                            item['content'] = ''
            
            value = json.dumps(value, ensure_ascii=True)
        except Exception as e:
            print(f"JSON encoding error: {e}")
            # If we can't encode as JSON, just return None instead of trying to fix
            return None
    
    # Sanitize string values that might contain problematic Unicode
    if isinstance(value, str):
        value = value.replace('\x00', '')
        # Truncate very long strings to prevent database errors
        if len(value) > 10000:
            value = value[:10000] + "..."
        
        # For fields that might contain problematic content, check for non-ASCII
        if 'content' in column_name.lower() if column_name else False:
            # If the string has complex Unicode characters, just return None
            if any(ord(c) > 127 for c in value):
                return None
    
    return value

def sanitize_unicode(text):
    """
    Sanitize text with problematic Unicode characters to make it safe for JSON encoding.
    
    Args:
        text (str): The text to sanitize
        
    Returns:
        str: Sanitized text
    """
    if not isinstance(text, str):
        return text
        
    # Replace problematic Unicode characters with their Unicode names or safe alternatives
    import re
    
    # Replace incomplete Unicode escape sequences
    text = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', text)
    
    # Handle mathematical alphanumeric symbols (common in fancy text)
    # These often cause problems in JSON encoding
    try:
        # Attempt to decode and re-encode to handle escape sequences properly
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    except Exception:
        # If that fails, use a more aggressive approach
        text = ''.join(c if ord(c) < 0x10000 else '?' for c in text)
    
    return text

def fetch_profiles_from_db(table_name):
    """Fetch all data from a given table in the PostgreSQL database."""

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = f"SELECT * FROM {table_name};"
        cur.execute(query)
        rows = cur.fetchall()
        column_names = [desc[0] for desc in cur.description]
        cur.close()
        conn.close()

        df = pd.DataFrame(rows, columns=column_names)

        #rename_map = {"profile_url": "Profile URL"}

        #df.rename(columns=rename_map, inplace=True)
        
        # Clean LinkedIn URLs if the Profile URL column exists
        if "Profile URL" in df.columns:
            df["Profile URL"] = df["Profile URL"].apply(clean_linkedin_url)
            
        return df  # Return the full DataFrame
    except Exception as e:
        print(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()

def insert_profile_to_db(profile_data, stealth, yc=False):
    """Insert a new profile into the PostgreSQL database, avoiding duplicates."""
    # Initialize variables that might be referenced in the exception handler
    placeholders = ""
    cleaned_columns = {}
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Clean column names to match the SQL table format
        cleaned_columns = {}
        for col, val in profile_data.items():
            # Skip empty values for numeric fields
            if val == "" and any(numeric_suffix in col.lower() for numeric_suffix in 
                               ['_score', 'likelihood', 'years_of_experience']):
                continue
            if col.lower() == 'id':
                continue

            cleaned_key = clean_column_name(col)
            cleaned_val = clean_value(val, col)
            
            # Only include non-None values
            if cleaned_val is not None:
                cleaned_columns[cleaned_key] = cleaned_val

        # Extract column names dynamically
        columns = ', '.join(f'"{col}"' for col in cleaned_columns.keys())
        placeholders = ', '.join(['%s'] * len(cleaned_columns))
        
        # Skip if there are no columns to insert
        if not columns or not placeholders:
            print("No valid columns to insert for profile:", profile_data.get("name", "Unknown"))
            return

        # Insert query (uses ON CONFLICT DO NOTHING to avoid duplicates)
        if stealth:
            insert_query = f"""
                        INSERT INTO stealth_founders ({columns}) 
                        VALUES ({placeholders})
                    """
        elif yc:
            insert_query = f"""
                        INSERT INTO yc ({columns}) 
                        VALUES ({placeholders})
                    """
        else:
            insert_query = f"""
                INSERT INTO founders ({columns}) 
                VALUES ({placeholders})
            """

        cur.execute(insert_query, list(cleaned_columns.values()))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting profile into database: {e}")
        print("Placeholders: ", placeholders)
        print("Cleaned columns: ", cleaned_columns)
        print("Profile data: ", profile_data)

def delete_profile_from_table(profile_url, profile_id, table="stealth_founders"):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = f"DELETE FROM {table} WHERE profile_url = %s OR id = %s;"
        cur.execute(query, (profile_url, profile_id))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error deleting {profile_url} from {table}: {e}")


def add_profile_columns_to_table(table_name="interesting_people"):
    """
    Add all profile dictionary columns to an existing table.
    This ensures the table can store all the data from initialize_profile_dict().
    
    Args:
        table_name (str): Name of the table to modify
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Import the initialize_profile_dict function to get all column names
        from services.profile_analysis import initialize_profile_dict
        
        # Get all column names from the profile dictionary
        profile_dict = initialize_profile_dict()
        
        # For each column in the profile dictionary, add it to the table if it doesn't exist
        for column_name in profile_dict.keys():
            # Clean the column name to match SQL naming conventions
            clean_col = clean_column_name(column_name)
            
            # Determine column type based on the column name
            if any(score_suffix in clean_col.lower() for score_suffix in 
                  ['_score', 'likelihood', 'years_of_experience']):
                col_type = "FLOAT"
            else:
                col_type = "TEXT"
            
            # Add the column if it doesn't exist
            cursor.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = '{table_name}' AND column_name = '{clean_col}'
                ) THEN
                    ALTER TABLE {table_name} ADD COLUMN "{clean_col}" {col_type};
                END IF;
            END
            $$;
            """)
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error adding profile columns to {table_name}: {e}")
        conn.rollback()
        return False
        
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_profile_in_db(profile_url, profile_data, table_name="interesting_people"):
    """
    Update an existing profile in the database with new profile data.
    
    Args:
        profile_url (str): The LinkedIn profile URL to update
        profile_data (dict): Dictionary containing profile information
        table_name (str): Name of the table to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Clean column names and values
        cleaned_columns = {}
        for col, val in profile_data.items():
            # Skip empty values for numeric fields
            if val == "" and any(numeric_suffix in col.lower() for numeric_suffix in 
                               ['_score', 'likelihood', 'years_of_experience']):
                continue
                
            cleaned_key = clean_column_name(col)
            cleaned_val = clean_value(val, col)
            
            # Only include non-None values
            if cleaned_val is not None:
                cleaned_columns[cleaned_key] = cleaned_val
        
        # Build the SET clause for the UPDATE statement
        set_clause = ", ".join([f'"{col}" = %s' for col in cleaned_columns.keys()])
        
        # Skip if there are no columns to update
        if not set_clause:
            print(f"No valid columns to update for profile: {profile_url}")
            return False
        
        # Update query
        update_query = f"""
            UPDATE {table_name}
            SET {set_clause}
            WHERE profile_url = %s
        """
        
        # Execute the update query with values
        values = list(cleaned_columns.values()) + [profile_url]
        cursor.execute(update_query, values)
        
        # Check if any rows were affected
        if cursor.rowcount == 0:
            print(f"No profile found with URL: {profile_url}")
            return False
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error updating profile in {table_name}: {e}")
        conn.rollback()
        return False
        
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def combine_previous_scrapes():
    """Fetch and combine already scraped profile URLs from the SQL database."""

    # Fetch full DataFrames from the database instead of CSV files
    all_df = fetch_profiles_from_db("founders")  
    yc_df = fetch_profiles_from_db("yc")
    stealth_df = fetch_profiles_from_db("stealth_founders")  

    # Ensure 'profile_url' column exists and drop NaN values
    all_df = all_df.dropna(subset=['profile_url'])
    yc_df = yc_df.dropna(subset=['profile_url'])
    stealth_df = stealth_df.dropna(subset=['profile_url'])

    # Extract Profile URLs as a list
    already_scraped_urls = (
            all_df['profile_url'].tolist() +
            yc_df['profile_url'].tolist() +
            stealth_df['profile_url'].tolist()
    )

    return already_scraped_urls

def heavy_clean(url):
    if not url:
        return None
    if "https://www." not in url:
        return url
    return url.split("https://www.")[1]

def is_linkedin_url(url):
    """
    Check if a URL is a LinkedIn profile URL.
    
    Args:
        url (str): URL to check
        
    Returns:
        bool: True if the URL is a LinkedIn profile URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
        
    # Check if the URL contains LinkedIn profile indicators
    
    return '/in/' in url and ('linkedin.com' in url.lower())

def cleanup_search_list():
    """
    Clean up the search_list and stealth_search_list tables by removing entries that have already been processed.
    
    This function:
    1. Fetches all profiles from founders, stealth_founders, and yc tables
    2. Gets all entries from search_list and stealth_search_list tables
    3. Removes entries from both tables that already exist in the processed tables
    4. Prints statistics about the cleanup process
    
    Returns:
        tuple: (remaining_count, removed_count) - total number of entries remaining and removed
    """
    import psycopg2
    from services.database import get_db_connection
        
    # Get all processed profiles (URLs only)
    already_scraped_urls = combine_previous_scrapes()
    already_scraped_heavy_cleaned = [heavy_clean(url) for url in already_scraped_urls]
    print(f"Found {len(already_scraped_urls)} already scraped URLs to check against")
        
    # Connect to the database for deletion operations
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database")
        return 0, 0
    
    cur = conn.cursor()
    total_removed = 0
    total_remaining = 0
    
    # Process both search_list and stealth_search_list tables
    tables_to_clean = ["search_list", "stealth_search_list"]
    for table_name in tables_to_clean:
        try:
            print(f"\nProcessing table: {table_name}")
            # Fetch all entries from the current table
            search_df = fetch_profiles_from_db(table_name)
            initial_count = len(search_df)
            print(f"  Found {initial_count} entries in {table_name}")
            
            removed_count = 0
            
            # Process each entry in the table
            checked_count = 0
            for index, row in search_df.iterrows():
                profile_url = row.get('profile_url')
                name = row.get('name', '')
                checked_count += 1
                
                # Skip if both profile_url and name are missing
                if not profile_url and not name:
                    continue
                    
                # Check if profile_url exists but is not a LinkedIn URL
                if profile_url and not is_linkedin_url(profile_url):
                    # Delete this entry from the table (non-LinkedIn URL)
                    cur.execute(
                        f"DELETE FROM {table_name} WHERE profile_url = %s",
                        (profile_url,)
                    )
                    removed_count += 1
                    continue
                    
                # Clean the URL if it exists
                if profile_url:
                    cleaned_url = clean_linkedin_url(profile_url)
                    
                    # Safely extract the heavy cleaned URL
                    heavy_cleaned_url = None
                    if cleaned_url and "https://www." in cleaned_url:
                        parts = cleaned_url.split("https://www.")
                        if len(parts) > 1:
                            heavy_cleaned_url = parts[1]
                    
                    # Check if URL already processed
                    if (cleaned_url and cleaned_url in already_scraped_urls) or (profile_url and profile_url in already_scraped_urls) or (heavy_cleaned_url and heavy_cleaned_url in already_scraped_urls) or (heavy_cleaned_url and heavy_cleaned_url in already_scraped_heavy_cleaned):
                        # Delete this entry from the table
                        cur.execute(
                            f"DELETE FROM {table_name} WHERE profile_url = %s",
                            (profile_url,)
                        )
                        removed_count += 1
                        continue
            
            # Commit the changes for this table
            conn.commit()
            
            # Get the remaining count for this table
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            remaining_count = cur.fetchone()[0]
            
            total_removed += removed_count
            total_remaining += remaining_count
            
        except Exception as e:
            conn.rollback()
            print(f"Error during {table_name} cleanup: {e}")
    
    print(f"\nTotal cleanup results:")
    print(f"  - Total removed: {total_removed} entries")
    print(f"  - Total remaining: {total_remaining} entries")
    
    # Close database connection
    cur.close()
    conn.close()
    
    return total_remaining, total_removed