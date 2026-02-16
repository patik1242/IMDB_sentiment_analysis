from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, chi2

def false_sentences(text, y_pred, y_test, name = "model"):
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
    plt.pie(values, labels, autopct='%1.1f%%')
    plt.title("Type of errors in sentiment analysis")
    plt.savefig(analysis_dir/f"{name}-error_pie.png")
    plt.close()

def mcnemar_results(y_pred_a, y_pred_b, y_test):
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)

    correct_a = (y_pred_a==y_test)
    correct_b = (y_pred_b==y_test)

    b = int(np.sum((~correct_a) & correct_b))
    c = int(np.sum(correct_a & (~correct_b)))
    n = b+c

    if n==0:
        return {"b": b, "c": c, "n": n, "method": "none", "p_value": 1.0}
    
    n_min = min(b,c)
    if n<25:
        pvalue = 2*binom.cdf(n_min, n, 0.5) 
        pvalue = min(1.0, pvalue)
        return {"b": b, "c": c, "n": n,"method": "exact","p_value": float(min(1.0, pvalue))}

    chi2_statistic = (abs(b-c)-1)**2/n
    pvalue = chi2.sf(chi2_statistic,1)
    return {"b": b, "c": c, "n": n, "method": "chi2", "chi2": float(chi2_statistic), "p_value": float(pvalue)}

