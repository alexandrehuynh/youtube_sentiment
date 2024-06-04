import os
from utils import get_video_comments, preprocess_comments
from models import train_sentiment_model, predict_sentiment

# Get the API key from the environment variable
api_key = os.environ.get('YOUTUBE_API_KEY')

def main():
    video_url = input("Enter the YouTube video URL: ")
    video_id = video_url.split('=')[-1]

    comments = get_video_comments(video_id, api_key)
    preprocessed_comments = preprocess_comments(comments)

    # You'll need to provide labeled data for training
    # For example:
    labels = [0, 2, 1, 0, 2, 1, 0, 2, ...]  # 0 for negative, 1 for neutral, 2 for positive

    model, vectorizer = train_sentiment_model(preprocessed_comments, labels)

    # Predict sentiment for new comments
    new_comments_vectors = vectorizer.transform(preprocessed_comments)
    predicted_sentiments = predict_sentiment(model, new_comments_vectors)

    # Display the results
    for comment, sentiment in zip(comments, predicted_sentiments):
        if sentiment == 0:
            print(f"Negative: {comment}")
        else:
            print(f"Positive: {comment}")

if __name__ == "__main__":
    main()