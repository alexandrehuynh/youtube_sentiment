import pandas as pd
import joblib
import os
from datetime import datetime
import csv
from utils import preprocess_comments

def list_model_files(directory='models'):
    files = [f for f in os.listdir(directory) if f.startswith('sentiment_model') and f.endswith('.pkl')]
    return files

def list_vectorizer_files(directory='models'):
    files = [f for f in os.listdir(directory) if f.startswith('vectorizer') and f.endswith('.pkl')]
    return files

def select_model_file(directory='models'):
    files = list_model_files(directory)
    if not files:
        raise FileNotFoundError("No model files found in the specified directory.")

    print("Select a model file to use for prediction:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")

    choice = int(input("Enter the number of the model file you want to use: "))
    if choice < 1 or choice > len(files):
        raise ValueError("Invalid choice")
    
    return os.path.join(directory, files[choice - 1])

def select_vectorizer_file(directory='models'):
    files = list_vectorizer_files(directory)
    if not files:
        raise FileNotFoundError("No vectorizer files found in the specified directory.")

    print("Select a vectorizer file to use for prediction:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")

    choice = int(input("Enter the number of the vectorizer file you want to use: "))
    if choice < 1 or choice > len(files):
        raise ValueError("Invalid choice")
    
    return os.path.join(directory, files[choice - 1])

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(model, vectorizer, comments):
    comments_vectors = vectorizer.transform(preprocess_comments(comments))
    predicted_sentiments = model.predict(comments_vectors)
    return predicted_sentiments

def save_comments_with_sentiments_to_csv(comments, sentiments, video_title, directory='comments_sentiments'):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = ''.join(e for e in video_title if e.isalnum() or e.isspace()).replace(' ', '_')
    filename = f"{directory}/{safe_title}_{timestamp}.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['comment', 'sentiment'])
        for comment, sentiment in zip(comments, sentiments):
            writer.writerow([comment, sentiment])
    
    return filename

def main_predict(comments, video_title):
    # Select model and vectorizer files
    model_directory = input("Enter the directory where model files are located (default is 'models'): ") or 'models'
    model_path = select_model_file(model_directory)
    vectorizer_path = select_vectorizer_file(model_directory)

    # Load the trained model and vectorizer
    model, vectorizer = load_model(model_path, vectorizer_path)

    # Predict sentiments for the preprocessed comments
    sentiments = predict_sentiment(model, vectorizer, comments)

    # Save comments with predicted sentiments to a CSV file
    sentiment_csv_filename = save_comments_with_sentiments_to_csv(comments, sentiments, video_title)
    print(f"Sentiments predicted and saved to '{sentiment_csv_filename}'.")
    return sentiment_csv_filename
