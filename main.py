import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from utils import get_video_comments, preprocess_comments, extract_video_id, get_video_title
import predict_sentiments

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

    youtube = authenticate_youtube()
    video_title = get_video_title(youtube, video_id)
    comments = get_video_comments(video_id, api_key)
    preprocessed_comments = preprocess_comments(comments)

    if not preprocessed_comments:
        print("No valid comments found after preprocessing.")
        return

    # Predict sentiments and save to CSV
    sentiment_csv_filename = predict_sentiments.main_predict(preprocessed_comments, video_title)
    print(f"Comments with predicted sentiments saved to '{sentiment_csv_filename}'.")

if __name__ == "__main__":
    main()
