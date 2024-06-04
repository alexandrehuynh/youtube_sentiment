import os
from googleapiclient.discovery import build
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# get the API key from the environment variable
api_key = os.environ.get('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=api_key)

# function to retrieve comments from a YouTube video
def get_video_comments(video_id, max_comments=100):
    comments = []
    try:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_comments,
            textFormat='plainText'
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')
        while next_page_token:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=max_comments,
                pageToken=next_page_token,
                textFormat='plainText'
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')

    except Exception as e:
        print(f'An error occurred: {e}')

    return comments

# function to preprocess the comments
def preprocess_comments(comments):
    preprocessed_comments = []
    stop_words = set(stopwords.words('english'))

    for comment in comments:
        # Remove URLs, HTML tags, and special characters
        comment = re.sub(r'http\S+|<[^>]+>|\W', ' ', comment)
        
        # Convert to lowercase
        comment = comment.lower()
        
        # Tokenize
        tokens = nltk.word_tokenize(comment)
        
        # Remove stop words and join tokens back into a string
        filtered_tokens = [token for token in tokens if token not in stop_words]
        preprocessed_comment = ' '.join(filtered_tokens)
        
        preprocessed_comments.append(preprocessed_comment)

    return preprocessed_comments

# main function
if __name__ == "__main__":
    video_id = "VIDEO_ID_HERE"
    comments = get_video_comments(video_id)
    print("Retrieved Comments:")
    for comment in comments:
        print(comment)
    
    preprocessed_comments = preprocess_comments(comments)
    print("\nPreprocessed Comments:")
    for comment in preprocessed_comments:
        print(comment)