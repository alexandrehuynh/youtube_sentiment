import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model(model_path='sentiment_model.pkl', vectorizer_path='vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(model, vectorizer, comments):
    comments_vectors = vectorizer.transform(comments)
    predicted_sentiments = model.predict(comments_vectors)
    return predicted_sentiments

if __name__ == "__main__":
    # Load the trained model and vectorizer
    model, vectorizer = load_model()

    # Load new comments to predict
    new_comments = pd.read_csv('path_to_new_comments.csv')['comment']
    predictions = predict_sentiment(model, vectorizer, new_comments)

    # Save predictions to CSV
    result = pd.DataFrame({'comment': new_comments, 'sentiment': predictions})
    result.to_csv('predicted_sentiments.csv', index=False)
    print("Sentiments predicted and saved to 'predicted_sentiments.csv'")
