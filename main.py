import os
from dotenv import load_dotenv
import googleapiclient.discovery
from utils import get_video_comments, preprocess_comments, extract_video_id
from models import train_sentiment_model, predict_sentiment

# Load environment variables from .env file
load_dotenv('./.env')

def authenticate_youtube():
    # Get the API key from the environment variable
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    if api_key is None:
        raise ValueError("The environment variable YOUTUBE_API_KEY is not set.")
    
    # Build the service using the API key
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=api_key)
    return youtube

def main():
    # Get the API key from the environment variable
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    if api_key is None:
        raise ValueError("The environment variable YOUTUBE_API_KEY is not set.")

    # Get the video URL from user input
    video_url = input("Enter the YouTube video URL: ")
    
    # Extract the video ID from the URL
    video_id = extract_video_id(video_url)

    if video_id is None:
        print("Invalid YouTube video URL.")
        return

    # Verify the video ID by making a request to the YouTube API
    youtube = authenticate_youtube()
    try:
        request = youtube.videos().list(
            part="id",
            id=video_id
        )
        response = request.execute()

        if not response["items"]:
            print("Video not found.")
            return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    
    # Get the video comments using the video ID
    comments = get_video_comments(video_id, api_key)
    
    # Preprocess the comments
    preprocessed_comments = preprocess_comments(comments)
    
    # Check if preprocessed comments are empty
    if not preprocessed_comments:
        print("No valid comments found after preprocessing.")
        return
    
    # Train the sentiment model
    labels = [...]  # Provide the sentiment labels for the comments
    model, vectorizer = train_sentiment_model(preprocessed_comments, labels)
    
    if model is None or vectorizer is None:
        print("Failed to train the sentiment model.")
        return
    
    # Predict the sentiment of new comments
    new_comments = [...]  # Provide new comments to predict sentiment
    new_comments_vectors = vectorizer.transform(new_comments)
    predicted_sentiments = predict_sentiment(model, new_comments_vectors)
    
    # Print the predicted sentiments
    for comment, sentiment in zip(new_comments, predicted_sentiments):
        print(f"Comment: {comment}")
        print(f"Predicted Sentiment: {sentiment}")

if __name__ == "__main__":
    main()