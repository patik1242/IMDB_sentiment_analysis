
from preprocessing import preprocess_vector
from training_and_calculating_metrics import train_model
from save_to_json import save_model, save_results_to_json
from analysis import mcnemar_results

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def training_model(clean_training):
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)

    text = clean_training["review"].copy()
    y = clean_training["sentiment"]

    vectorizer = TfidfVectorizer(ngram_range = (1,2), max_df=0.9, max_features=120000, min_df=2)

    X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42, stratify=y) 
    
    X_train = X_train.apply(preprocess_vector)
    X_test = X_test.apply(preprocess_vector)   

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    results = train_model(X_train_tfidf, X_test_tfidf, y_train, y_test)

    best_f1_score = 0
    best_estimator = None
    best_model_name = None

    for name, model_dict in results.items():
        f1 = model_dict["test"]["f1"]
        if f1 > best_f1_score:
            best_f1_score = f1
            best_estimator = model_dict["estimator"]
            best_model_name = name
                
    model_info = {
        "name": best_model_name, 
        "f1": best_f1_score
    }

    results_sorted = sorted(results.items(), key=lambda x: x[1]["test"]["f1"], reverse=True)
    best_name, best_dict = results_sorted[0]
    second_name, second_dict = results_sorted[1]
    
    y_pred_best = best_dict["estimator"].predict(X_test_tfidf)
    y_pred_second = second_dict["estimator"].predict(X_test_tfidf)
    
    mc_results = mcnemar_results(y_pred_best, y_pred_second, y_test)
    save_results_to_json(results, model_info, second_name, mc_results)
    save_model(best_estimator, vectorizer)


    plt.figure(figsize=(12,6))

    df_plot = pd.DataFrame({
        "Model":[m for m, d in results.items()],
        "Accuracy":[d["test"]["accuracy"] for m, d in results.items()],
        "Precision":[d["test"]["precision"] for m, d in results.items()],
        "Recall":[d["test"]["recall"] for m, d in results.items()],
        "F1":[d["test"]["f1"] for m, d in results.items()],
    })

    df_plot.set_index('Model').plot(kind="bar", figsize=(12,6))
    plt.title("Test metrics")
    plt.ylabel("Metrics")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(charts_dir/"Comparing_metrics.png")
    plt.close()

    