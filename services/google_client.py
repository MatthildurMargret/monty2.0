import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import base64
from bs4 import BeautifulSoup
from services.openai_api import ask_monty
from services.groq_api import get_groq_response
import re
import pandas as pd
from collections import defaultdict
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def authenticate_gmail():
    # Define the scope for read-only access to Gmail
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/spreadsheets.readonly'
    ]
    CREDENTIALS_PATH = '/Users/matthildur/monty/config/google_credentials.json'
    TOKEN_PATH = '/Users/matthildur/monty/config/token.json'

    creds = None

    # Load and decode GOOGLE_CREDENTIALS_BASE64
    google_creds_b64 = os.getenv('GOOGLE_CREDENTIALS_BASE64')
    if not google_creds_b64:
        raise ValueError("GOOGLE_CREDENTIALS_BASE64 is not set in environment variables")

    google_creds_json = base64.b64decode(google_creds_b64).decode('utf-8')
    google_creds = json.loads(google_creds_json)

    # Try to load tokens from local file first (more reliable)
    if os.path.exists(TOKEN_PATH):
        try:
            with open(TOKEN_PATH, 'r') as token_file:
                token_data = json.load(token_file)
                creds = Credentials.from_authorized_user_info(token_data, SCOPES)
                print("✅ Loaded credentials from local token file")
        except Exception as e:
            print(f"Warning: Could not load token file: {e}")

    # If no local token, try loading from environment variable
    if not creds:
        google_tokens_b64 = os.getenv('GOOGLE_TOKENS_BASE64')
        if google_tokens_b64:
            try:
                google_tokens_json = base64.b64decode(google_tokens_b64).decode('utf-8')
                creds = Credentials.from_authorized_user_info(json.loads(google_tokens_json), SCOPES)
                print("✅ Loaded credentials from environment variable")
            except Exception as e:
                print(f"Warning: Could not load saved credentials from env: {e}")
                creds = None

    # If we have credentials, check if they're valid or can be refreshed
    if creds:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Save the refreshed token to local file
                token_data = json.loads(creds.to_json())
                os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
                with open(TOKEN_PATH, 'w') as token_file:
                    json.dump(token_data, token_file)
                print("✅ Credentials refreshed and saved to local file")
                
                # Also print the base64 version for env var update
                new_token_b64 = base64.b64encode(creds.to_json().encode('utf-8')).decode('utf-8')
                print("⚠️  Update your GOOGLE_TOKENS_BASE64 with this new value (optional, local file is used):")
                print(new_token_b64)
            except Exception as e:
                print(f"Warning: Could not refresh credentials: {e}")
                creds = None
        elif not creds.valid:
            print("Warning: Credentials are invalid and cannot be refreshed")
            creds = None

    # If no valid creds, do OAuth flow
    if not creds or not creds.valid:
        print("Starting OAuth flow...")
        flow = InstalledAppFlow.from_client_config(google_creds, SCOPES)
        creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
        
        # Save token to local file
        token_data = json.loads(creds.to_json())
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        with open(TOKEN_PATH, 'w') as token_file:
            json.dump(token_data, token_file)
        print("✅ Credentials saved to local token file")
        
        # Print new token to rebase64 and set in env (optional)
        new_token_json = creds.to_json()
        new_token_b64 = base64.b64encode(new_token_json.encode('utf-8')).decode('utf-8')
        print("\n✅ New GOOGLE_TOKENS_BASE64 value (optional, local file is used):\n")
        print(new_token_b64)
        print("\n")

    return creds


def get_emails_with_label(start_date, label_name="building tool", raw_format=True):
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)

    # Query for mails with the specified label
    end_date = (datetime.today() + timedelta(days=1)).strftime('%Y/%m/%d')
    query = f"label:{label_name} after:{start_date} before:{end_date}"

    results = service.users().messages().list(userId='me', q=query).execute()

    if 'messages' not in results:
        print("No emails found with this label.")
        return []

    emails = []

    for msg in results['messages']:
        msg_id = msg['id']
        # Fetch email metadata in 'metadata' format to extract headers
        msg_metadata = service.users().messages().get(userId='me', id=msg_id, format='metadata').execute()
        headers = msg_metadata.get('payload', {}).get('headers', [])

        # Extract sender ("From") even if raw_format=True
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")
        to = next((h['value'] for h in headers if h['name'] == 'To'), "Unknown To")

        if raw_format:
            # Fetch raw email content
            msg_details = service.users().messages().get(userId='me', id=msg_id, format='raw').execute()
            raw_email = base64.urlsafe_b64decode(msg_details['raw'].encode('ASCII'))

            # Include sender in raw mode
            emails.append({"id": msg_id, "from": sender, "to": to, "raw_email": raw_email})
            continue  # Skip further processing

        # Process structured email data (if raw_format=False)
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
        payload = msg_metadata.get("payload", {})

        body = extract_email_body(payload)

        email_data = {
            'id': msg_id,
            'subject': subject,
            'from': sender,
            'body': body,
            'to': to
        }
        emails.append(email_data)

    return emails


def parse_raw_email(email):
    from email.parser import BytesParser
    from email import policy

    msg = BytesParser(policy=policy.default).parsebytes(email['raw_email'])
    text_content = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                text_content = part.get_content()
                break
    else:
        text_content = msg.get_content()

    soup = BeautifulSoup(text_content, 'html.parser')
    text = soup.get_text(separator="\n")
    return text

def send_html_email(to, subject, message_text):
    """
    Send an email using the Gmail API.
    
    Args:
        to (str or list): Email address(es) of the recipient(s). Can be a single email string or a list of email addresses.
        subject (str): Subject of the email
        message_text (str): Body of the email (can include HTML)
        
    Returns:
        dict: The sent message if successful
    """
    import base64
    from email.mime.text import MIMEText
    
    # Authenticate and build the Gmail service
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    
    message = MIMEText(message_text, 'html')
    
    # Handle both single email and list of emails
    if isinstance(to, list):
        message['to'] = ', '.join(to)
    else:
        message['to'] = to
        
    message['subject'] = subject
    
    # Encode the message
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    
    try:
        sent_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()
        print(f"Email sent to {to} with message ID: {sent_message['id']}")
        return sent_message
    except Exception as e:
        print(f"An error occurred while sending the email: {e}")
        return None


def send_email(html_content):
    """
    Send the weekly summary email with top founders and pre-seed deals.
    """
    try:
        # Set up email parameters
        to_addresses = ['matthildur@montageventures.com', 'nia@montageventures.com', 'connie@montageventures.com',
                'daphne@montageventures.com', 'matt@montageventures.com', 'todd@montageventures.com']
        #to_addresses = ['matthildur@montageventures.com']
        subject = f"Monty's Weekly Update - {datetime.now().strftime('%B %d, %Y')}"
        
        # Send the email
        print(f"Sending email to {to_addresses}...")
        result = send_html_email(to_addresses, subject, html_content)
        
        if result:
            print("✅ Email sent successfully!")
            return True
        else:
            print("❌ Failed to send email")
            return False
            
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False