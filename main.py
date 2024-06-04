import os
from dotenv import load_dotenv
import json
import googleapiclient.discovery

# Load environment variables from .env file
load_dotenv('./.env')

def authenticate_youtube():
    # Get the API key from the environment variable
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    # Print the API key to ensure it's being set
    print(f'API Key: {api_key}')
    
    if api_key is None:
        raise ValueError("The environment variable YOUTUBE_API_KEY is not set.")
    
    # Build the service using the API key
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=api_key)
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
