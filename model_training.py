
from preprocessing import preprocess_vector
from training_and_calculating_metrics import train_model
from save_to_json import save_model, save_results_to_json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def training_model(data):
    text = data["review"].copy()
    y = data["sentiment"]

    vectorizer = TfidfVectorizer(ngram_range = (1,2))

    X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42, stratify=y) 
    
    X_train = X_train.apply(preprocess_vector)
    X_test = X_test.apply(preprocess_vector)   

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = train_model(X_train_tfidf, X_test_tfidf, y_train, y_test)

    model_info = {
        "name": "Logistic Regression", 
        "f1": results["test"]["f1"]
    }

    save_results_to_json(results, model_info)
    save_model(results["estimator"], vectorizer)