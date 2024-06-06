import re
from googleapiclient.discovery import build
import nltk
from nltk.corpus import stopwords
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)
    if 'youtube' in parsed_url.hostname:
        if parsed_url.path == '/watch':
            video_id = parse_qs(parsed_url.query).get('v', None)
            if video_id:
                return video_id[0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    elif 'youtu.be' in parsed_url.hostname:
        return parsed_url.path[1:]
    return None

def get_video_comments(video_id, api_key, max_comments=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_comments,
            textFormat='plainText'
        )
        response = request.execute()

        while response:
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            if 'nextPageToken' in response:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    pageToken=response['nextPageToken'],
                    maxResults=max_comments,
                    textFormat='plainText'
                )
                response = request.execute()
            else:
                break

    except Exception as e:
        print(f'An error occurred: {e}')

    return comments

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
        
        if preprocessed_comment.strip():  # Check if the preprocessed comment is not empty
            preprocessed_comments.append(preprocessed_comment)

    return preprocessed_comments