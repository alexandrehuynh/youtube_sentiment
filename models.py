from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_sentiment_model(comments, labels):
    try:
        # Convert comments to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(comments)

        # Train the Naive Bayes classifier
        model = MultinomialNB()
        model.fit(X, labels)

        return model, vectorizer
    except ValueError as e:
        print(f"Error: {str(e)}")
        return None, None

def predict_sentiment(model, new_comments_vectors):
    predicted_sentiments = model.predict(new_comments_vectors)
    return predicted_sentiments