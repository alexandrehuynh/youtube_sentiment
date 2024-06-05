import re
from googleapiclient.discovery import build
import nltk
from nltk.corpus import stopwords

def extract_video_id(url):
    # Regular expression pattern to match the video ID
    pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=)?(?:embed\/)?(?:v\/)?(?:shorts\/)?(?:\S+)'
    
    # Find the video ID in the URL
    match = re.match(pattern, url)
    
    if match:
        # Extract the video ID from the matched URL
        video_id = match.group().split('/')[-1]
        video_id = video_id.split('?')[0]
        video_id = video_id.split('&')[0]
        return video_id
    else:
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