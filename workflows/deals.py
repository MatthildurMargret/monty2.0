import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import from services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.google_client import get_emails_with_label, parse_raw_email
from services.deal_processing import process_vcnewsdaily, process_fortune_termsheet, find_link_if_missing, process_fresh_funding, process_daily_digest

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
            # Combine and remove duplicates based on Company and Date
            combined_df = (
                pd.concat([master_df, deals_df])
                .assign(Company_clean=lambda d: d['Company'].str.replace(r"\s+", "", regex=True).str.lower())
                .drop_duplicates(subset=['Company_clean', 'Amount'])
                .drop(columns=['Company_clean'])
            )
            combined_df.to_csv(master_file, index=False)
            print(f"Updated master deals file with {len(deals_df)} new deals")
        else:
            deals_df.to_csv(master_file, index=False)
            print(f"Created master deals file with {len(deals_df)} deals")
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
       
    all_deals.drop_duplicates(subset='Company', inplace=True)
    print(all_deals["Company"].unique())
    early_stage_deals = pd.DataFrame(columns=all_deals.columns)
    
    for index, row in all_deals.iterrows():
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
            
            # Check if it's an early stage deal (less than $7M)
            if amount_value < 10.0:
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
        combined.drop_duplicates(subset=['Company', 'Date'], inplace=True)
        os.makedirs('data/deal_data', exist_ok=True)
        combined.to_csv('data/deal_data/early_deals.csv', index=False)
        print(f"Appended {len(new_deals)} new early stage deals to early_deals.csv")
    else:
        print("No new early stage deals found.")
    
    return new_deals

def add_deals_to_database(new_deals=None, days_back=7):
    """
    Process early-stage deals by finding founders and adding them to the database.
    Also saves the enriched deal data back to a CSV file.
    """
    from services.deal_processing import analyze_early_stage_deal
    import pandas as pd
    import os
    
    print("Starting founder discovery for early-stage deals...")
    
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
        for index, row in deals.iterrows():
            try:
                deal_dict = row.to_dict()
                date = datetime.strptime(deal_dict["Date"], "%Y-%m-%d")
                if date < datetime.now() - timedelta(days=days_back):
                    continue
                
                print(f"[{index+1}/{len(deals)}] Processing {deal_dict.get('Company', 'Unknown')}")
                
                # Analyze deal and find founders
                enriched_deal = analyze_early_stage_deal(deal_dict)
                enriched_deals.append(enriched_deal)
                
                # Report on found founders
                founders = enriched_deal.get("Founders", [])
                if founders:
                    print(f"  ✓ Found {len(founders)} founders")
                else:
                    print("  ✗ No founders found")
            except Exception as e:
                print(f"  ✗ Error processing deal: {str(e)}")
        
        # Save enriched data with founder information
        #pd.DataFrame(enriched_deals).to_csv(input_file, index=False)
        #print(f"Saved enriched deals with founder information to {input_file}")
        
        return True
    
    except Exception as e:
        print(f"Error in add_deals_to_database: {str(e)}")
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
            # Combine and remove duplicates based on Company and Date
            combined_df = pd.concat([master_df, deals_df]).drop_duplicates(subset=['Company', 'Date'])
            combined_df.to_csv(master_file, index=False)
            print(f"Updated master deals file with reprocessed deals")
    else:
        print("No deals found during reprocessing.")
    
    return deals_df

if __name__ == "__main__":
    # Uncomment the line below to run the regular workflow
    days_back = 7
    deals = updated_newsletter_deals(days_back=days_back)
    
    deals = pd.read_csv("data/deal_data/all_deals.csv")
    deals = deals[-50:]
    early_stage_deals = find_early_stage_deals(deals)
    #add_deals_to_database(early_stage_deals, 5)