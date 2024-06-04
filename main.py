import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('./.env')

import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

def authenticate_youtube():
    # Get the client secrets file path from the environment variable
    SERVICE_ACCOUNT_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    # Define the scopes required
    SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

    # Create credentials
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    # Refresh the token if necessary
    credentials.refresh(Request())

    # Build the service
    youtube = build('youtube', 'v3', credentials=credentials)
    return youtube

def main():
    # Authenticate and build the YouTube service
    youtube = authenticate_youtube()

    # Your existing code to interact with YouTube API
    video_id = 'FzAxtfdDHnY'
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()
    print(json.dumps(response, indent=4))

if __name__ == "__main__":
    main()
