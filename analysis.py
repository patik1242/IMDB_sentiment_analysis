from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom, chi2
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc

def false_sentences(text, y_pred, y_test, name = "model"):
    """
    Zapisuje błędnie sklasyfikowane przykłady.
    Wykrywa przypadki fałszywie pozytywne i negatywne, oblicza ich liczbę.
    """
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    fp_mask = (y_pred==1) & (y_test==0)
    fn_mask = (y_pred==0) & (y_test==1)
    tp = np.sum((y_pred==1)& (y_test==1))
    tn = np.sum((y_pred==0)& (y_test==0))

    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)

    df = pd.DataFrame({
        "text": text, 
        "true": y_test, 
        "pred": y_pred
    })

    df[fp_mask].to_csv(analysis_dir/f"{name}_false_positive.csv", index = False)
    df[fn_mask].to_csv(analysis_dir/f"{name}_false_negative.csv", index = False)

    labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
    values = [tp, tn, fp, fn]

    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title("Type of errors in sentiment analysis")
    plt.savefig(analysis_dir/f"{name}-error_pie.png")
    plt.close()

def mcnemar_results(y_pred_a, y_pred_b, y_test):
    """
    Za pomocą testu McNemara porównuje dwa modele klasyfikacji.
    Analizuje przypadki, w których modele się różnią. 
    n01 - Model A błędny, Model B poprawny
    n10 - Model A poprawny, Model B błędny
    n11 - Model A poprawny, Model B poprawny
    n00 - Model A błędny, Model B błędny 

    Dla małych wartości n<25 wykonywany jest test dokładny, a 
    dla większych stosowany jest przybliżenie chi-kwadrat. 
    """
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    correct_a = (y_pred_a==y_test)
    correct_b = (y_pred_b==y_test)

    n01 = int(np.sum((~correct_a) & correct_b))
    n10 = int(np.sum(correct_a & (~correct_b)))
    n11 = int(np.sum((correct_a) & correct_b))
    n00 = int(np.sum((~correct_a) & (~correct_b)))

    n = n01 + n10

    if n==0:
        return {"n01": n01, "n10": n10, "n": n, "n11": n11, "n00": n00, "method": "none", "p_value": 1.0}
    
    n_min = min(n01,n10)
    if n<25:
        pvalue = 2*binom.cdf(n_min, n, 0.5) 
        pvalue = min(1.0, pvalue)
        return {"n01": n01, "n10": n10, "n": n, "n11": n11, "n00": n00, "method": "exact","p_value": float(min(1.0, pvalue))}

    chi2_statistic = (abs(n01-n10)-1)**2/n
    pvalue = chi2.sf(chi2_statistic,1)

    matrix = np.array([
        [n11, n10],
        [n01, n00]], dtype=int)
    
    plt.figure(figsize=(6,6))
    sns.heatmap(matrix, 
                cmap = "Blues", 
                annot = True, 
                fmt = "d", 
                xticklabels=["B correct", "B wrong"], 
                yticklabels=["A correct", "A wrong"])
    
    plt.title(f"McNemar (p = {pvalue:.3g})")
    plt.xlabel("Model B")
    plt.ylabel("Model A")
    plt.savefig(analysis_dir/f"McNemarResults.png")
    plt.close()

    return {"n01": n01, "n10": n10, "n": n, "n11": n11, "n00": n00, "method": "chi2", "chi2": float(chi2_statistic), "p_value": float(pvalue)}

def plot_learning_curve(model,X,y, model_name):
    """
    Zapisuje krzywą uczenia dla danego modelu. 
    Learning curve pokazuje przebieg nauki dla każdego modelu na 
    zbiorze testowym i treningowym. 
    """
    charts_dir = Path("charts") / "learning_curves"
    charts_dir.mkdir(parents=True, exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="f1_macro", n_jobs=-1, train_sizes= np.linspace(0.1,1.0,5), random_state=42, shuffle=True
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(6,6))
    plt.plot(train_sizes, train_mean, label="Train F1")
    plt.plot(train_sizes, val_mean, label="Validation F1")
    plt.xlabel("Training size")
    plt.ylabel("F1 Macro")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(charts_dir/f"Learning_curve_{model_name}.png")
    plt.close()

def plot_roc_curve(y_true, y_score, model_name):
    """
    Rysuje i zapisuje krzywą ROC i wylicza ROC AUC dla podanych wyników. 
    Krzywa ROC pokazuje zależność pomiędzy True Positive Rate (jaki % pozytywnych przykładów model poprawnie wykrywa), 
    a False Positive Rate (jaki % negatywnych przykładów model oznacza jako pozytywne).
    
    Dla różnych progów sprawdzam jak się zmienia liczba wykrytych pozytywnych i liczba fałszywych alarmów. 
    ROC AUC mówi jak model dobrze rozdziela klasy niezależnie od progu
    """
    charts_dir = Path("charts")/"roc_curve"
    charts_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label = f"{model_name} (AUC = {roc_auc:.3f})")
    plt.plot([0,1],[0,1], linestyle = "--")
    plt.xlabel("False positive Rate") #ile ze wszystkich negatywnych oznaczył jako pozytywne
    plt.ylabel("True Positive Rate") #ile ze wszystkich pozytywnych oznaczał jako pozytywne
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(charts_dir/f"ROC_{model_name}.png")
    plt.close()

