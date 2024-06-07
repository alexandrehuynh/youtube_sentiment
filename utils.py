import re
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        if parsed_url.path[:7] == '/embed/':
            return parsed_url.path.split('/')[2]
        if parsed_url.path[:3] == '/v/':
            return parsed_url.path.split('/')[2]
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

            if 'nextPageToken' in response and len(comments) < max_comments:
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

def get_video_title(youtube, video_id):
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()
    return response['items'][0]['snippet']['title']
