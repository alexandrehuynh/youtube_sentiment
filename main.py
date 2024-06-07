import os
import csv
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from utils import get_video_comments, preprocess_comments, extract_video_id, get_video_title
import pandas as pd
import joblib
from datetime import datetime

# Load environment variables from .env file
load_dotenv('./.env')

def authenticate_youtube():
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key is None:
        raise ValueError("The environment variable YOUTUBE_API_KEY is not set.")
    youtube = build('youtube', 'v3', developerKey=api_key)
    return youtube

def save_comments_with_sentiments_to_csv(comments, sentiments, video_title, directory='comments'):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = ''.join(e for e in video_title if e.isalnum() or e.isspace()).replace(' ', '_')
    filename = f"{directory}/{safe_title}_{timestamp}_with_sentiments.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['comment', 'sentiment'])
        for comment, sentiment in zip(comments, sentiments):
            writer.writerow([comment, sentiment])
    
    return filename

def load_model(model_path='models/sentiment_model.pkl', vectorizer_path='models/vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(model, vectorizer, comments):
    comments_vectors = vectorizer.transform(comments)
    predicted_sentiments = model.predict(comments_vectors)
    return predicted_sentiments

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

    # Load the trained model and vectorizer
    model, vectorizer = load_model()

    # Predict sentiments for the preprocessed comments
    sentiments = predict_sentiment(model, vectorizer, preprocessed_comments)

    # Save comments with predicted sentiments to a CSV file
    sentiment_csv_filename = save_comments_with_sentiments_to_csv(preprocessed_comments, sentiments, video_title)
    print(f"Sentiments predicted and saved to '{sentiment_csv_filename}'.")

if __name__ == "__main__":
    main()
