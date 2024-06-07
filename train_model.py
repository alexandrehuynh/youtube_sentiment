import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
from datetime import datetime

def list_csv_files(directory='datasets'):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    return files

def select_csv_file(directory='datasets'):
    files = list_csv_files(directory)
    if not files:
        raise FileNotFoundError("No CSV files found in the specified directory.")
    print("Select a CSV file to train the model:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}. {file}")
    choice = int(input("Enter the number of the CSV file you want to use: "))
    if choice < 1 or choice > len(files):
        raise ValueError("Invalid choice")
    return os.path.join(directory, files[choice - 1])

def select_model():
    models = ['Naive Bayes', 'Logistic Regression']
    print("Select a model to train:")
    for idx, model in enumerate(models, start=1):
        print(f"{idx}. {model}")
    choice = int(input("Enter the number of the model you want to train: "))
    if choice < 1 or choice > len(models):
        raise ValueError("Invalid choice")
    return models[choice - 1]

def load_data(filename):
    data = pd.read_csv(filename)
    return data['comment'], data['sentiment']

def preprocess_text(comments):
    # Implement any necessary preprocessing steps, ensuring consistency
    return comments

def train_sentiment_model(comments, labels, model_type):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocess_text(comments))
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Address class imbalance using SMOTE
    smote = SMOTE()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    if model_type == 'Naive Bayes':
        model = MultinomialNB()
    elif model_type == 'Logistic Regression':
        model = LogisticRegression()
    else:
        raise ValueError("Invalid model type selected")

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return model, vectorizer

def save_model(model, vectorizer, directory='models'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{directory}/sentiment_model_{timestamp}.pkl"
    vectorizer_path = f"{directory}/vectorizer_{timestamp}.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model and vectorizer saved to {model_path} and {vectorizer_path}")

if __name__ == "__main__":
    directory = input("Enter the directory where CSV files are located (default is 'datasets'): ") or 'datasets'
    csv_file = select_csv_file(directory)
    model_type = select_model()
    comments, labels = load_data(csv_file)
    model, vectorizer = train_sentiment_model(comments, labels, model_type)
    save_model(model, vectorizer)
