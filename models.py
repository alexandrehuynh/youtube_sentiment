from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def train_sentiment_model(comments, labels):
    try:
        # Convert comments to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(comments)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        # Train the Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        return model, vectorizer
    except ValueError as e:
        print(f"Error: {str(e)}")
        return None, None

def predict_sentiment(model, vectorizer, new_comments):
    new_comments_vectors = vectorizer.transform(new_comments)
    predicted_sentiments = model.predict(new_comments_vectors)
    return predicted_sentiments
