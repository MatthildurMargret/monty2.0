import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.google_client import get_emails_with_label, parse_raw_email
from services.deal_processing import process_vcnewsdaily, process_fortune_termsheet, find_link_if_missing, process_fresh_funding, process_daily_digest, normalize_company_name

load_dotenv()

# Try to import supabase client
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️  supabase package not installed. Deals will only be saved to CSV. Install with: pip install supabase")


def get_supabase_client():
    """Create and return a Supabase client."""
    if not SUPABASE_AVAILABLE:
        return None
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        return None
    
    try:
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"⚠️  Error creating Supabase client: {e}")
        return None


def upload_deals_to_supabase(deals_df, table_name="all_deals"):
    """
    Upload deals DataFrame to Supabase.
    
    Args:
        deals_df: pandas DataFrame with deals data
        table_name: Name of the Supabase table ('all_deals' or 'early_deals')
    
    Returns:
        bool: True if successful, False otherwise
    """
    if deals_df is None or deals_df.empty:
        return False
    
    supabase = get_supabase_client()
    if not supabase:
        print(f"⚠️  Supabase client not available. Skipping upload to {table_name} table.")
        return False
    
    try:
        # Create a copy to avoid modifying the original
        df_copy = deals_df.copy()
        
        # Map column names from DataFrame format to database format (lowercase with underscores)
        column_mapping = {
            'Company': 'company',
            'Amount': 'amount',
            'Funding Round': 'funding_round',
            'Vertical': 'vertical',
            'Link': 'link',
            'Investors': 'investors',
            'Category': 'category',
            'Source': 'source',
            'Date': 'date',
            'Founders': 'founders'
        }
        
        # Rename columns to match database schema
        df_copy = df_copy.rename(columns=column_mapping)
        
        # For early_deals table, check if 'amount' column exists in the database
        # If not, we'll exclude it from the upload to avoid schema errors
        expected_columns = ['company', 'amount', 'funding_round', 'vertical', 'link', 
                          'investors', 'category', 'source', 'date', 'founders']
        
        # Ensure all expected columns exist (add missing ones as empty)
        for col in expected_columns:
            if col not in df_copy.columns:
                df_copy[col] = ""
        
        # For early_deals table, we'll try without 'amount' first if it causes errors
        # Select only the columns we need - but be flexible about 'amount'
        columns_to_upload = expected_columns.copy()
        if table_name == "early_deals":
            # Try to include amount, but we'll handle the error if it doesn't exist
            pass  # Keep all columns for now, error handling will catch it
        
        # Select only the columns we need
        df_copy = df_copy[columns_to_upload]
        
        # Convert DataFrame to list of dictionaries
        # Replace NaN values with None for JSON serialization
        deals_list = df_copy.where(pd.notna(df_copy), None).to_dict('records')
        
        # Ensure all required columns exist and handle None values
        for deal in deals_list:
            # Convert None to empty string for text fields, keep date as None if missing
            for key, value in deal.items():
                # Handle pandas Series/arrays - convert to scalar first
                if isinstance(value, (pd.Series, list, tuple)):
                    if len(value) == 0:
                        value = None
                    else:
                        value = value[0] if isinstance(value, (list, tuple)) else value.iloc[0]
                
                if key == 'date':
                    # Handle date column specially - convert to string or None
                    if value is None:
                        deal[key] = None
                    elif pd.isna(value):
                        deal[key] = None
                    elif isinstance(value, str):
                        deal[key] = value  # Already a string
                    else:
                        # Try to convert datetime to string
                        try:
                            deal[key] = pd.to_datetime(value).strftime('%Y-%m-%d')
                        except:
                            deal[key] = None
                else:
                    # For text fields, convert None/NaN to empty string
                    if value is None:
                        deal[key] = ""
                    elif pd.isna(value):
                        deal[key] = ""
                    else:
                        # Convert to string to ensure proper serialization
                        deal[key] = str(value) if value is not None else ""
        
        print(f"📤 Uploading {len(deals_list)} deals to Supabase table '{table_name}'...")
        
        # Insert deals in batches (Supabase has limits on batch size)
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(deals_list), batch_size):
            batch = deals_list[i:i + batch_size]
            try:
                response = supabase.table(table_name).insert(batch).execute()
                total_inserted += len(batch)
                print(f"   ✅ Inserted batch {i//batch_size + 1} ({len(batch)} deals)")
            except Exception as e:
                error_msg = str(e)
                # Check if it's a duplicate key error (which is okay)
                if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
                    print(f"   ⚠️  Some deals in batch {i//batch_size + 1} already exist (skipping duplicates)")
                    # Try inserting one by one to handle duplicates gracefully
                    for deal in batch:
                        try:
                            supabase.table(table_name).insert(deal).execute()
                            total_inserted += 1
                        except:
                            pass  # Skip duplicates
                # Check if it's a schema error (column doesn't exist)
                elif "column" in error_msg.lower() and "schema cache" in error_msg.lower():
                    print(f"   ⚠️  Schema mismatch detected. Trying without problematic columns...")
                    # Try again without 'amount' column if it's early_deals
                    if table_name == "early_deals" and "amount" in error_msg.lower():
                        # Remove 'amount' from all deals in batch
                        batch_without_amount = []
                        for deal in batch:
                            deal_copy = deal.copy()
                            deal_copy.pop('amount', None)
                            batch_without_amount.append(deal_copy)
                        try:
                            response = supabase.table(table_name).insert(batch_without_amount).execute()
                            total_inserted += len(batch_without_amount)
                            print(f"   ✅ Inserted batch {i//batch_size + 1} without 'amount' column ({len(batch_without_amount)} deals)")
                        except Exception as e2:
                            print(f"   ❌ Error inserting batch {i//batch_size + 1} even without 'amount': {e2}")
                    else:
                        print(f"   ❌ Error inserting batch {i//batch_size + 1}: {e}")
                else:
                    print(f"   ❌ Error inserting batch {i//batch_size + 1}: {e}")
        
        print(f"✅ Successfully uploaded {total_inserted} deals to {table_name} table")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower():
            print(f"❌ Table '{table_name}' does not exist in Supabase.")
            print(f"   Please create it first using the SQL setup script.")
        else:
            print(f"❌ Error uploading deals to Supabase: {e}")
        return False


def update_founders_in_supabase(deals_df, table_name="early_deals"):
    """
    Update existing deals in Supabase with founder information.
    
    Args:
        deals_df: pandas DataFrame with deals data including Founders column
        table_name: Name of the Supabase table ('all_deals' or 'early_deals')
    
    Returns:
        bool: True if successful, False otherwise
    """
    if deals_df is None or deals_df.empty:
        return False
    
    supabase = get_supabase_client()
    if not supabase:
        print(f"⚠️  Supabase client not available. Skipping update to {table_name} table.")
        return False
    
    try:
        # Create a copy to avoid modifying the original
        df_copy = deals_df.copy()
        
        # Convert Founders list to string if it's a list
        if 'Founders' in df_copy.columns:
            df_copy['Founders'] = df_copy['Founders'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) and x else (x if x else '')
            )
        
        # Map column names from DataFrame format to database format
        column_mapping = {
            'Company': 'company',
            'Date': 'date',
            'Founders': 'founders'
        }
        
        # Rename columns to match database schema
        df_copy = df_copy.rename(columns=column_mapping)
        
        # Ensure required columns exist
        if 'company' not in df_copy.columns or 'date' not in df_copy.columns:
            print("❌ Missing required columns (company, date) for updating founders")
            return False
        
        if 'founders' not in df_copy.columns:
            df_copy['founders'] = ''
        
        # Filter to only deals that have founders
        # Use .loc to avoid pandas array comparison issues
        mask = df_copy['founders'].notna() & (df_copy['founders'].astype(str) != '')
        deals_with_founders = df_copy.loc[mask].copy()
        
        if deals_with_founders.empty:
            print("   No deals with founders to update.")
            return True
        
        print(f"📤 Updating {len(deals_with_founders)} deals with founders in Supabase table '{table_name}'...")
        
        # Update deals one by one (using company and date as identifier)
        updated_count = 0
        column_missing = False
        
        for _, row in deals_with_founders.iterrows():
            try:
                # Update the deal using company and date as the identifier
                response = supabase.table(table_name)\
                    .update({'founders': row['founders']})\
                    .eq('company', str(row['company']))\
                    .eq('date', str(row['date']))\
                    .execute()
                
                if response.data:
                    updated_count += 1
            except Exception as e:
                error_msg = str(e)
                # Check if it's a column missing error
                if "column" in error_msg.lower() and ("founders" in error_msg.lower() or "schema cache" in error_msg.lower()):
                    if not column_missing:
                        # Only print this message once
                        print(f"\n⚠️  The 'founders' column does not exist in the '{table_name}' table.")
                        print(f"   To add it, run this SQL in your Supabase SQL editor:")
                        print(f"   ALTER TABLE {table_name} ADD COLUMN founders TEXT;")
                        print(f"   \n   Founders data has been saved to CSV but not uploaded to Supabase.")
                        print(f"   After adding the column, you can re-run this script to update Supabase.\n")
                        column_missing = True
                    # Skip remaining updates
                    continue
                # If deal doesn't exist, that's okay - just skip it
                elif "does not exist" not in error_msg.lower():
                    print(f"   ⚠️  Error updating {row.get('company', 'Unknown')}: {e}")
        
        if column_missing:
            print(f"⚠️  Could not update any deals - 'founders' column missing from database")
        elif updated_count > 0:
            print(f"✅ Successfully updated {updated_count} deals with founders in {table_name} table")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower():
            print(f"❌ Table '{table_name}' does not exist in Supabase.")
            print(f"   Please create it first using the SQL setup script.")
        else:
            print(f"❌ Error updating founders in Supabase: {e}")
        return False


def updated_newsletter_deals(days_back=6):
    try:
        processed_emails = pd.read_csv('data/deal_data/processed_emails.csv')
    except FileNotFoundError:
        processed_emails = pd.DataFrame(columns=["Email_ID"])

    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    emails = get_emails_with_label(start_date, label_name="deals", raw_format=True)
    
    all_deals = []
    processed_email_ids = processed_emails["Email_ID"].tolist()
    
    for email in emails:
        if email["id"] in processed_email_ids:
            continue

        if ("Venture Daily Digest" in email['from']):
            print("Processing: Venture Daily Digest")
            email_text = parse_raw_email(email)
            deals = process_daily_digest(email_text)
            print("Got deals: ", deals)
            all_deals += deals
            processed_email_ids.append(email["id"])
            
        if ("VC News Daily" in email['from']):
            print("Processing: VC News Daily")
            email_text = parse_raw_email(email)
            deals = process_vcnewsdaily(email_text)
            print("Got deals: ", deals)
            all_deals += deals
            processed_email_ids.append(email["id"])

        elif ("Term Sheet | Fortune" in email['from']):
            print("Processing: Term Sheet | Fortune")
            email_text = parse_raw_email(email)
            deals = process_fortune_termsheet(email_text)
            print("Got deals: ", deals)
            all_deals += deals
            processed_email_ids.append(email["id"])

        elif ("Fresh Funding & Products" in email['from']):
            print("Processing: Fresh Funding & Products")
            email_text = parse_raw_email(email)
            deals = process_fresh_funding(email_text)
            all_deals += deals
            processed_email_ids.append(email["id"])
    
    # Print deals for debugging
    for deal in all_deals:
        print("Deal: ", deal)
    
    # Save deals to CSV
    if all_deals:
        # Create directory if it doesn't exist
        os.makedirs('data/deal_data', exist_ok=True)
        
        # Create a DataFrame with the deals
        deals_df = pd.DataFrame(all_deals)
        
        # Select and reorder the columns
        columns_order = ['Company', 'Amount', 'Funding Round', 'Vertical', 'Link', 'Investors', 'Category', 'Source', 'Date']
        deals_df = deals_df[columns_order]
        
        # Save to CSV
        today = datetime.now().strftime('%Y-%m-%d')
        #deals_df.to_csv(f'deal_data/deals_{today}.csv', index=False)
        #print(f"Saved {len(deals_df)} deals to deal_data/deals_{today}.csv")
        
        # Append to master deals file if it exists, otherwise create it
        master_file = 'data/deal_data/all_deals.csv'
        if os.path.exists(master_file):
            master_df = pd.read_csv(master_file)
            # Combine and remove duplicates based on normalized Company name and Amount
            combined_df = (
                pd.concat([master_df, deals_df])
                .assign(Company_normalized=lambda d: d['Company'].apply(normalize_company_name))
                .drop_duplicates(subset=['Company_normalized', 'Amount'])
                .drop(columns=['Company_normalized'])
            )
            combined_df.to_csv(master_file, index=False)
            print(f"Updated master deals file with {len(deals_df)} new deals")
        else:
            deals_df.to_csv(master_file, index=False)
            print(f"Created master deals file with {len(deals_df)} deals")
        
        # Upload to Supabase
        upload_deals_to_supabase(deals_df, table_name="all_deals")
    else:
        print("No deals found")
        return None
    
    # Update processed emails list
    if processed_email_ids:
        os.makedirs('data/deal_data', exist_ok=True)
        new_processed_df = pd.DataFrame({"Email_ID": processed_email_ids})
        
        if os.path.exists('data/deal_data/processed_emails.csv'):
            existing_df = pd.read_csv('data/deal_data/processed_emails.csv')
            combined_df = pd.concat([existing_df, new_processed_df]).drop_duplicates()
            combined_df.to_csv('data/deal_data/processed_emails.csv', index=False)
        else:
            new_processed_df.to_csv('data/deal_data/processed_emails.csv', index=False)

    return deals_df

def find_early_stage_deals(all_deals):
    if all_deals is None:
        return None
       
    # Use normalized company name for deduplication
    all_deals['Company_normalized'] = all_deals['Company'].apply(normalize_company_name)
    all_deals.drop_duplicates(subset='Company_normalized', inplace=True)
    all_deals.drop(columns=['Company_normalized'], inplace=True)
    print(all_deals["Company"].unique())
    early_stage_deals = pd.DataFrame(columns=all_deals.columns)
    
    for index, row in all_deals.iterrows():
        # Exclude Series A deals from early stage deals
        funding_round = str(row["Funding Round"]).strip()
        if funding_round.lower() in ["series a", "seriesa"]:
            continue
        
        # Convert Amount to string to handle all cases
        amount_str = str(row["Amount"]).strip()
        
        # Skip if amount is not specified or empty
        if amount_str.lower() == 'not specified' or amount_str == '' or amount_str.lower() == 'nan':
            if row["Funding Round"] in ["Seed", "Pre-seed"]:
                early_stage_deals = pd.concat([early_stage_deals, pd.DataFrame([row])])
            continue
        
        try:
            # Normalize the amount string
            normalized_amount = amount_str.lower().strip()
            
            # Handle special cases and non-standard formats
            if any(x in normalized_amount for x in ['not specified', '...', 'undisclosed']):
                # If it's a seed or pre-seed round, include it anyway
                if row["Funding Round"] in ["Seed", "Pre-seed", "Pre-Seed"]:
                    early_stage_deals = pd.concat([early_stage_deals, pd.DataFrame([row])])
                continue
                
            if 'significant' in normalized_amount or 'investment' in normalized_amount:
                # Skip deals with vague amounts
                continue
                
            # Extract numeric part using regex
            import re
            numeric_match = re.search(r'(\d+\.?\d*)', normalized_amount)
            if not numeric_match:
                continue
                
            numeric_value = float(numeric_match.group(1))
            
            # Determine the scale (millions, thousands)
            if any(x in normalized_amount for x in ['m', 'million']):
                amount_value = numeric_value
            elif any(x in normalized_amount for x in ['k', 'thousand']):
                amount_value = numeric_value / 1000
            elif 'over' in normalized_amount or 'more than' in normalized_amount:
                # For 'over X million' type formats, use X as the value
                amount_value = numeric_value
            elif 'x' in normalized_amount or 'X' in normalized_amount:
                # For formats with X placeholder, skip
                continue
            else:
                # Assume it's in millions if no unit specified
                amount_value = numeric_value
                
            # Standardize the amount format in the row
            row["Amount"] = f"${amount_value}M"
            
            # Check if it's an early stage deal (less than $10M)
            # Exclude Series A deals even if they're under $10M
            if amount_value < 10.0 and funding_round.lower() not in ["series a", "seriesa"]:
                early_stage_deals = pd.concat([early_stage_deals, pd.DataFrame([row])])
                
        except ValueError as e:
            print(f"Could not parse amount '{amount_str}' for company '{row['Company']}': {e}")
            continue
    
    # Filter out deals that are already in the early_deals.csv file
    old_deals = pd.DataFrame()
    if os.path.exists('data/deal_data/early_deals.csv'):
        old_deals = pd.read_csv('data/deal_data/early_deals.csv')
        
    # Only keep deals that are not in old_deals
    if not old_deals.empty:
        merged = early_stage_deals.merge(
            old_deals[['Company', 'Date']], 
            on=['Company', 'Date'], 
            how='left', 
            indicator=True
        )
        new_deals = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    else:
        new_deals = early_stage_deals.copy()

    # Add links where missing to early stage deals
    for index, row in new_deals.iterrows():
        row = find_link_if_missing(row)
        new_deals.loc[index] = row
    
    # Append new deals to the same master file
    if not new_deals.empty:
        print(new_deals)
        combined = pd.concat([old_deals, new_deals], ignore_index=True)
        # Use normalized company name for deduplication
        combined['Company_normalized'] = combined['Company'].apply(normalize_company_name)
        combined.drop_duplicates(subset=['Company_normalized', 'Date'], inplace=True)
        combined.drop(columns=['Company_normalized'], inplace=True)
        os.makedirs('data/deal_data', exist_ok=True)
        combined.to_csv('data/deal_data/early_deals.csv', index=False)
        print(f"Appended {len(new_deals)} new early stage deals to early_deals.csv")
        
        # Upload new early stage deals to Supabase (without founders - they'll be added later)
        # Note: Founders will be extracted and added by add_deals_to_database()
        upload_deals_to_supabase(new_deals, table_name="early_deals")
    else:
        print("No new early stage deals found.")
    
    return new_deals

def extract_founders_from_link(company_name, link, parallel_client):
    """
    Extract founder names from a funding announcement link using Parallel API.
    
    Args:
        company_name: Name of the company
        link: URL to the funding announcement article
        parallel_client: Parallel API client instance
        
    Returns:
        list: List of founder names found (empty if none or error)
    """
    if not link or pd.isna(link) or str(link).strip() == '':
        return []
    
    if not str(link).startswith('http'):
        return []
    
    try:
        # Use Parallel API extract to find founders
        objective = f"Who is the CEO or founder of {company_name}? Return only the full name (first and last name) of one person. If multiple founders are mentioned, return the CEO or primary founder."
        
        extract = parallel_client.beta.extract(
            urls=[link],
            objective=objective,
            excerpts=True,
            full_content=False,
        )
        
        # Extract excerpts from response
        excerpts = []
        if extract.results and len(extract.results) > 0:
            result = extract.results[0]
            if result.get('excerpts'):
                excerpts = result['excerpts']
        
        # Use excerpts for AI extraction
        text_to_search = ' '.join(excerpts) if excerpts else ''
        
        # Use AI to extract founder name from text
        founder_name = None
        if text_to_search:
            try:
                from services.openai_api import ask_monty
                
                # Create a prompt for AI extraction
                extraction_prompt = f"""Extract the founder or CEO name from this funding announcement article about {company_name}.

IMPORTANT INSTRUCTIONS:
1. Return ONLY the full name (first and last name) of ONE person - the CEO or primary founder
2. If multiple founders are mentioned, return the CEO or the person explicitly identified as the primary founder
3. DO NOT return investor names, customer names, or other company names
4. DO NOT return names of people who are just mentioned in passing
5. Look for explicit mentions like "Founder and CEO of {company_name}", "CEO {company_name}", "{company_name} was founded by", etc.
6. Return the name in the format "First Last" or "First Middle Last"
7. If you cannot find a clear founder/CEO name, return exactly: "NOT_FOUND"

Return ONLY the name, nothing else. No explanations, no additional text."""

                # Truncate text if too long (to avoid token limits)
                # Keep the text but limit to ~8000 characters to stay within token limits
                text_for_ai = text_to_search[:8000] if len(text_to_search) > 8000 else text_to_search
                
                # Call AI to extract founder name
                ai_response = ask_monty(extraction_prompt, text_for_ai, max_tokens=50)
                
                # Clean and validate the response
                founder_name = ai_response.strip()
                
                # Remove any quotes or extra formatting
                founder_name = founder_name.strip('"\'')
                
                # Check if AI said it couldn't find a name
                if founder_name.upper() in ['NOT_FOUND', 'NONE', 'N/A', 'NONE FOUND', '']:
                    founder_name = None
                else:
                    # Basic validation: should be 2-4 words (first, middle, last, title)
                    name_parts = founder_name.split()
                    if len(name_parts) < 2 or len(name_parts) > 4:
                        founder_name = None
                    # Check if it looks like a name (starts with capital letters)
                    elif not all(part[0].isupper() for part in name_parts if part):
                        founder_name = None
                    # Exclude common false positives
                    name_lower = founder_name.lower()
                    false_positives = [
                        'define ventures', 'general catalyst', 'y combinator', 'sequoia',
                        'blue bottle coffee', 'the company', 'the startup', company_name.lower()
                    ]
                    if any(fp in name_lower for fp in false_positives):
                        founder_name = None
                        
            except Exception as e:
                print(f"    Error using AI extraction: {str(e)}")
                founder_name = None
        
        founders = [founder_name] if founder_name else []
        return founders
        
    except Exception as e:
        print(f"    Error extracting founders from link: {str(e)}")
        return []


def add_deals_to_database(new_deals=None, days_back=7):
    """
    Process early-stage deals by finding founders using Parallel API extract.
    Does NOT save founders to database - only extracts and reports them.
    
    Args:
        new_deals: Optional DataFrame of deals to process. If None, loads from early_deals.csv
        days_back: Only process deals from the last N days (default: 7)
    """
    import pandas as pd
    import os
    import time
    
    # Check if Parallel API is available
    try:
        from services.parallel_client import Parallel
        PARALLEL_AVAILABLE = True
    except ImportError:
        print("❌ parallel_client not available. Check services/parallel_client.py")
        return False
    
    # Check for API key
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        print("❌ PARALLEL_API_KEY not found in environment variables")
        return False
    
    # Initialize Parallel client
    try:
        parallel_client = Parallel(api_key=api_key)
        print("✅ Parallel API client initialized")
    except Exception as e:
        print(f"❌ Error initializing Parallel client: {e}")
        return False
    
    print("Starting founder discovery for early-stage deals using Parallel API...")
    
    try:
        # Load the deals
        if new_deals is None:
            input_file = 'data/deal_data/early_deals.csv'
            if not os.path.exists(input_file):
                print(f"Error: Input file {input_file} not found.")
                return False
        
            deals = pd.read_csv(input_file)
            if deals.empty:
                print("No deals to process.")
                return False
        else:
            deals = new_deals
            if deals.empty:
                print("No deals to process.")
                return False
        
        print(f"Processing {len(deals)} deals...")
        
        # Process each deal and collect enriched data
        enriched_deals = []
        deals_with_founders = 0
        total_founders = 0
        
        for index, row in deals.iterrows():
            try:
                deal_dict = row.to_dict()
                
                # Check date filter
                try:
                    date_value = deal_dict.get("Date")
                    if date_value is None or pd.isna(date_value):
                        # Skip if no date
                        continue
                    
                    # Handle both string and Timestamp/date objects
                    if isinstance(date_value, str):
                        date = datetime.strptime(date_value, "%Y-%m-%d")
                    elif isinstance(date_value, (pd.Timestamp, datetime)):
                        date = date_value if isinstance(date_value, datetime) else date_value.to_pydatetime()
                    else:
                        # Try to convert to datetime
                        date = pd.to_datetime(date_value).to_pydatetime()
                    
                    if date < datetime.now() - timedelta(days=days_back):
                        continue
                except (ValueError, KeyError, TypeError) as e:
                    # If date parsing fails, skip date filter but continue processing
                    pass
                
                company_name = deal_dict.get('Company', 'Unknown')
                link = deal_dict.get('Link', '')
                
                print(f"[{index+1}/{len(deals)}] Processing {company_name}")
                
                # Extract founders using Parallel API
                if link and pd.notna(link) and str(link).strip() != '':
                    print(f"  Extracting founders from: {link}")
                    founders = extract_founders_from_link(company_name, link, parallel_client)
                    
                    if founders:
                        deal_dict["Founders"] = founders
                        deals_with_founders += 1
                        total_founders += len(founders)
                        print(f"  ✓ Found {len(founders)} founder(s): {', '.join(founders)}")
                    else:
                        deal_dict["Founders"] = []
                        print(f"  ✗ No founders found")
                else:
                    deal_dict["Founders"] = []
                    print(f"  ⚠️  No valid link provided")
                
                enriched_deals.append(deal_dict)
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Error processing deal: {str(e)}")
                # Still add the deal even if extraction failed
                deal_dict = row.to_dict()
                deal_dict["Founders"] = []
                enriched_deals.append(deal_dict)
        
        # Summary
        print("\n" + "=" * 80)
        print("Founder Extraction Summary")
        print("=" * 80)
        print(f"Total deals processed: {len(enriched_deals)}")
        print(f"Deals with founders found: {deals_with_founders}")
        print(f"Total founders extracted: {total_founders}")
        print(f"Average founders per deal: {total_founders / len(enriched_deals) if enriched_deals else 0:.2f}")
        
        # Update early_deals.csv with enriched data
        if enriched_deals:
            # Convert to DataFrame
            enriched_df = pd.DataFrame(enriched_deals)
            
            # Convert Founders list to string for CSV and database
            if 'Founders' in enriched_df.columns:
                enriched_df['Founders'] = enriched_df['Founders'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) and x else ''
                )
            
            # Load existing early_deals.csv
            early_deals_file = 'data/deal_data/early_deals.csv'
            if os.path.exists(early_deals_file):
                existing_df = pd.read_csv(early_deals_file)
                
                # Update existing deals with new founder information
                # Create a mapping of (Company, Date) -> Founders
                founders_map = {}
                for _, row in enriched_df.iterrows():
                    company = row.get('Company', '')
                    date = row.get('Date', '')
                    founders = row.get('Founders', '')
                    if company and date:
                        # Normalize date format for matching
                        try:
                            if isinstance(date, str):
                                date_normalized = pd.to_datetime(date).strftime('%Y-%m-%d')
                            else:
                                date_normalized = pd.to_datetime(date).strftime('%Y-%m-%d')
                            founders_map[(company, date_normalized)] = founders
                        except:
                            founders_map[(company, str(date))] = founders
                
                # Update the existing dataframe
                if 'Founders' not in existing_df.columns:
                    existing_df['Founders'] = ''
                
                # Update founders for matching deals
                updated_count = 0
                for idx, row in existing_df.iterrows():
                    company = row.get('Company', '')
                    date = row.get('Date', '')
                    if company and date:
                        try:
                            # Normalize date format for matching
                            if isinstance(date, str):
                                date_normalized = pd.to_datetime(date).strftime('%Y-%m-%d')
                            else:
                                date_normalized = pd.to_datetime(date).strftime('%Y-%m-%d')
                            
                            key = (company, date_normalized)
                            if key in founders_map:
                                existing_df.at[idx, 'Founders'] = founders_map[key]
                                updated_count += 1
                        except:
                            key = (company, str(date))
                            if key in founders_map:
                                existing_df.at[idx, 'Founders'] = founders_map[key]
                                updated_count += 1
                
                # Save updated early_deals.csv
                os.makedirs('data/deal_data', exist_ok=True)
                existing_df.to_csv(early_deals_file, index=False)
                print(f"\n✅ Updated {updated_count} deals in {early_deals_file} with founder information")
                
                # Also update Supabase with founders for the deals we just processed
                print(f"\n📤 Updating founders in Supabase...")
                # Filter to only deals that have founders found
                mask = enriched_df['Founders'].notna() & (enriched_df['Founders'].astype(str) != '')
                deals_with_founders_df = enriched_df.loc[mask].copy()
                if not deals_with_founders_df.empty:
                    # Update early_deals table with founders (using company and date as identifier)
                    update_founders_in_supabase(deals_with_founders_df, table_name="early_deals")
                else:
                    print("   No deals with founders to update.")
            else:
                # If early_deals.csv doesn't exist, create it with enriched data
                os.makedirs('data/deal_data', exist_ok=True)
                enriched_df.to_csv(early_deals_file, index=False)
                print(f"\n✅ Created {early_deals_file} with enriched deals")
                
                # Upload to Supabase
                print(f"\n📤 Uploading deals to Supabase...")
                upload_deals_to_supabase(enriched_df, table_name="early_deals")
        
        return True
    
    except Exception as e:
        print(f"Error in add_deals_to_database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def reprocess_emails(sources=None, days_back=7):
    """
    Reprocess emails from specific sources by ignoring the processed_emails check.
    
    Args:
        sources (list): List of email sources to reprocess (e.g., ['VC News Daily', 'Term Sheet | Fortune'])
                       If None, reprocess all sources
        days_back (int): Number of days to look back for emails
    """
    if sources is None:
        sources = ["VC News Daily", "Term Sheet | Fortune", "Fresh Funding & Products"]
    
    print(f"Reprocessing emails from sources: {sources}")
    
    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    emails = get_emails_with_label(start_date, label_name="deals", raw_format=True)
    
    all_deals = []
    
    for email in emails:
        email_source = email['from']
        
        # Check if this email is from a source we want to reprocess
        should_process = False
        for source in sources:
            if source in email_source:
                should_process = True
                break
        
        if not should_process:
            continue
            
        print(f"Reprocessing: {email_source}")
        email_text = parse_raw_email(email)
        
        if "VC News Daily" in email_source:
            deals = process_vcnewsdaily(email_text)
        elif "Term Sheet | Fortune" in email_source:
            deals = process_fortune_termsheet(email_text)
            print("Got Term Sheet deals: ", deals)
        elif "Fresh Funding & Products" in email_source:
            deals = process_fresh_funding(email_text)
        else:
            deals = []
            
        all_deals += deals
    
    # Save deals to CSV
    if all_deals:
        # Create directory if it doesn't exist
        os.makedirs('deal_data', exist_ok=True)
        
        # Create a DataFrame with the deals
        deals_df = pd.DataFrame(all_deals)
        
        # Select and reorder the columns
        columns_order = ['Company', 'Amount', 'Funding Round', 'Vertical', 'Link', 'Investors', 'Category', 'Source', 'Date']
        deals_df = deals_df[columns_order]
        
        # Save to CSV
        today = datetime.now().strftime('%Y-%m-%d')
        deals_df.to_csv(f'data/deal_data/reprocessed_deals_{today}.csv', index=False)
        print(f"Saved {len(deals_df)} reprocessed deals to data/deal_data/reprocessed_deals_{today}.csv")
        
        # Append to master deals file
        master_file = 'data/deal_data/all_deals.csv'
        if os.path.exists(master_file):
            master_df = pd.read_csv(master_file)
            # Combine and remove duplicates based on normalized Company name and Date
            combined_df = (
                pd.concat([master_df, deals_df])
                .assign(Company_normalized=lambda d: d['Company'].apply(normalize_company_name))
                .drop_duplicates(subset=['Company_normalized', 'Date'])
                .drop(columns=['Company_normalized'])
            )
            combined_df.to_csv(master_file, index=False)
            print(f"Updated master deals file with reprocessed deals")
        
        # Upload reprocessed deals to Supabase
        upload_deals_to_supabase(deals_df, table_name="all_deals")
    else:
        print("No deals found during reprocessing.")
    
    return deals_df

def process_recent_early_deals(days_back=6):
    """
    Process early-stage deals from the past N days:
    1. Find links for deals that don't have them
    2. Extract founders from links
    3. Update CSV and Supabase
    
    Args:
        days_back: Number of days to look back (default: 6)
    """
    import pandas as pd
    import os
    
    print(f"🔄 Processing early-stage deals from the past {days_back} days...")
    
    # Load early_deals.csv
    input_file = 'data/deal_data/early_deals.csv'
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return False
    
    deals_df = pd.read_csv(input_file)
    if deals_df.empty:
        print("No deals found in early_deals.csv")
        return False
    
    print(f"📊 Loaded {len(deals_df)} total deals from early_deals.csv")
    
    # Filter deals from the past N days
    cutoff_date = datetime.now() - timedelta(days=days_back)
    deals_df['Date'] = pd.to_datetime(deals_df['Date'], errors='coerce')
    recent_deals = deals_df[deals_df['Date'] >= cutoff_date].copy()
    
    if recent_deals.empty:
        print(f"ℹ️  No deals found from the past {days_back} days")
        return False
    
    print(f"📅 Found {len(recent_deals)} deals from the past {days_back} days")
    
    # Step 1: Find links for deals that don't have them
    print("\n" + "=" * 80)
    print("Step 1: Finding missing links...")
    print("=" * 80)
    
    deals_needing_links = recent_deals[
        (recent_deals['Link'].isna()) | 
        (recent_deals['Link'] == '') | 
        (recent_deals['Link'].astype(str).str.strip() == 'nan') |
        (recent_deals['Link'].astype(str).str.startswith('https://www.google.com/search'))
    ].copy()
    
    if not deals_needing_links.empty:
        print(f"🔍 Found {len(deals_needing_links)} deals without valid links")
        
        # Update links in the dataframe
        for index, row in deals_needing_links.iterrows():
            company = row['Company']
            print(f"  Searching for link: {company}")
            updated_row = find_link_if_missing(row.to_dict())
            # Update the row in recent_deals
            for key, value in updated_row.items():
                if key in recent_deals.columns:
                    recent_deals.at[index, key] = value
        
        # Update the main deals_df with new links
        for index in deals_needing_links.index:
            for col in recent_deals.columns:
                if col in deals_df.columns:
                    deals_df.at[index, col] = recent_deals.at[index, col]
        
        # Save updated CSV
        deals_df.to_csv(input_file, index=False)
        print(f"✅ Updated {input_file} with new links")
        
        # Update links in Supabase for deals that got new links
        print("\n📤 Updating links in Supabase...")
        deals_with_new_links = recent_deals.loc[deals_needing_links.index].copy()
        # Convert Date back to string for Supabase upload
        if 'Date' in deals_with_new_links.columns:
            deals_with_new_links['Date'] = deals_with_new_links['Date'].dt.strftime('%Y-%m-%d')
        
        # Update Supabase - try to update existing records first, then insert if needed
        # We'll use upload_deals_to_supabase which handles duplicates
        upload_deals_to_supabase(deals_with_new_links, table_name="early_deals")
    else:
        print("✅ All deals already have links")
    
    # Step 2: Extract founders
    print("\n" + "=" * 80)
    print("Step 2: Extracting founders...")
    print("=" * 80)
    
    # Convert Date back to string format for add_deals_to_database
    recent_deals_for_processing = recent_deals.copy()
    if 'Date' in recent_deals_for_processing.columns:
        recent_deals_for_processing['Date'] = recent_deals_for_processing['Date'].dt.strftime('%Y-%m-%d')
    
    # Use add_deals_to_database but pass the filtered deals
    add_deals_to_database(new_deals=recent_deals_for_processing, days_back=days_back)
    
    print("\n" + "=" * 80)
    print("✅ Processing complete!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Check if we should only process recent early deals
    if len(sys.argv) > 1 and sys.argv[1] == "--recent-only":
        days_back = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        process_recent_early_deals(days_back=days_back)
    else:
        # Run the regular workflow
        days_back = 6
        deals = updated_newsletter_deals(days_back=days_back)
        
        early_stage_deals = find_early_stage_deals(deals)
        add_deals_to_database(early_stage_deals, days_back=7)