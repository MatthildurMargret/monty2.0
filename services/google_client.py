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

    # Load and decode GOOGLE_TOKENS_BASE64 if present
    google_tokens_b64 = os.getenv('GOOGLE_TOKENS_BASE64')
    if google_tokens_b64:
        google_tokens_json = base64.b64decode(google_tokens_b64).decode('utf-8')
        creds = Credentials.from_authorized_user_info(json.loads(google_tokens_json), SCOPES)

    # If no valid creds, do OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(google_creds, SCOPES)
            creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
            # Print new token to rebase64 and set in env
            new_token_json = creds.to_json()
            new_token_b64 = base64.b64encode(new_token_json.encode('utf-8')).decode('utf-8')
            print("New GOOGLE_TOKENS_BASE64 value:\n")
            print(new_token_b64)

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
