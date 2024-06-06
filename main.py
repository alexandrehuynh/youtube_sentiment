import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils import get_video_comments, preprocess_comments, extract_video_id

# Load environment variables from .env file
load_dotenv('./.env')

def authenticate_youtube():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key is None:
        raise ValueError("The environment variable YOUTUBE_API_KEY is not set.")
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

def main():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key is None:
        raise ValueError("The environment variable YOUTUBE_API_KEY is not set.")

    video_url = input("Enter the YouTube video URL: ")
    video_id = extract_video_id(video_url)

    if video_id is None:
        print("Invalid YouTube video URL.")
        return

    comments = get_video_comments(video_id, api_key)
    preprocessed_comments = preprocess_comments(comments)

    if not preprocessed_comments:
        print("No valid comments found after preprocessing.")
        return

    # Print the retrieved comments
    print("Retrieved comments:")
    for comment in preprocessed_comments:
        print(comment)

if __name__ == "__main__":
    main()