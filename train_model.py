import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_data(filename):
    data = pd.read_csv(filename)
    return data['comment'], data['sentiment']

def train_sentiment_model(comments, labels):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(comments)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return model, vectorizer

def save_model(model, vectorizer, directory='models'):
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    model_path = f"{directory}/sentiment_model.pkl"
    vectorizer_path = f"{directory}/vectorizer.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model and vectorizer saved to {model_path} and {vectorizer_path}")

if __name__ == "__main__":
    comments, sentiments = load_data('./comments/Craig Jones vs Gabi Garcia press conference highlights! (FUNNY!)_with_sentiments.csv')
    model, vectorizer = train_sentiment_model(comments, sentiments)
    save_model(model, vectorizer)
