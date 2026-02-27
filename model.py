
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
    """
    Główna funkcja treningowa. 
    Przygotowuje dane (wyodrębnia tekst i etykiety), dzieli dane na zbiór treningowy i testowy, 
    preprocessinguje tekst, wektoryzuje TF-IDF, trenuje kilka modeli klasyfikacji, ewaluuję je i
    wybiera najlepszy model na podstawie F1 
    (łączy precision i recall, czy model nie robi jakiś fałszywych alarmów czy nie przegapia pozytywnych) i kary za przeuczenie. 
    Porównuje 2najlepsze modele testem McNemara, zapisuje wyniki do JSON, 
    zapisuje najlepszy model i wektoryzator i generuje wykres porównujący metryki testowe. 
    """
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)

    print("Przygotowuje dane do treningu... ")
    text = clean_training["review"].copy()
    y = clean_training["sentiment"]

    vectorizer = TfidfVectorizer(ngram_range = (1,2), max_df=0.9, max_features=120000, min_df=2)

    print("Dzielę dane na train i test, wykonuję preprocessing + TF-IDF...")
    X_train, X_test, y_train, y_test = train_test_split(text, y, test_size=0.2, random_state=42, stratify=y) 
    
    #Preprocessing
    X_train_preprocess = X_train.apply(preprocess_vector)
    X_test_preprocess = X_test.apply(preprocess_vector)   

    #TF-IDF
    X_train_tfidf = vectorizer.fit_transform(X_train_preprocess)
    X_test_tfidf = vectorizer.transform(X_test_preprocess)

    print("Rozpoczynam trening modeli i obliczanie metryk...")
    results = train_model(X_train_tfidf, X_test_tfidf, y_train, y_test, X_test)
    
    best_estimator = None

    ALPHA = 0.5
    MAX_GAP = 0.07

    scored = []
    for name, model_dict in results.items():
        train_f1 = model_dict["train"]["f1"]
        test_f1 = model_dict["test"]["f1"]
        gap = train_f1 - test_f1
        
        #Kara za przeuczenie
        score = test_f1 - ALPHA*gap
        if gap <= MAX_GAP:
            scored.append((name, score, gap, test_f1))

    if not scored:
        for name, model_dict in results.items():
            train_f1 = model_dict["train"]["f1"]
            test_f1 = model_dict["test"]["f1"]
            gap = train_f1 - test_f1
            score = test_f1 - ALPHA*gap
            scored.append((name, score, gap, test_f1))

    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    best_name, best_scored, best_gap, best_f1 = scored_sorted[0]
    second_name = scored_sorted[1][0] if len(scored_sorted) > 1 else None

    best_estimator = results[best_name]["estimator"]

    model_info = {
        "name": best_name, 
        "f1": best_f1, 
        "score": best_scored,
        "gap": best_gap, 
        "ALPHA": ALPHA
    }
    
    if second_name is not None:
        y_pred_best = results[best_name]["estimator"].predict(X_test_tfidf)
        y_pred_second = results[second_name]["estimator"].predict(X_test_tfidf)
        print("Porównuję dwa najlepsze modele testem McNemara...")
        mc_results = mcnemar_results(y_pred_best, y_pred_second, y_test)
    else:
        mc_results = None
    
    print("Zapisuje wyniki w formacie json...")
    save_results_to_json(results, model_info, second_name, mc_results)
    print("Zapisuje najlepszy model na podstawie F1 i gap...")
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

    