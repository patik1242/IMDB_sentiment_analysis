from analysis import false_sentences, plot_learning_curve, plot_roc_curve

from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC

from pathlib import Path
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred, model_name, split):
    """
    Liczę podstawowe metryki klasyfikacji (accuracy, precision, recall, f1) i zapisuję macierze pomyłek. 
    Macro - liczy metryke osobno dla każdej klasy i robi średnią, a każda klasa ma taką samą wagę.
    Average - klasa liczniejsza ma większy wpływ na wynik 
    """
    charts_dir = Path("charts") / "confusion_matrixes"
    charts_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true,y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nMetryki dla {model_name} ({split}): \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    title = f"confusion_matrix - {model_name} ({split})"
    if split == "test":
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        plt.title(title)
        filename = charts_dir/f"{title}.png"
        i=1
        while filename.exists():
            filename = charts_dir/f"{title}_{i}.png"
            i+=1

        plt.savefig(filename)
        plt.close()
            
    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1
    }

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, text, model_name):
    """
    Funkcja genereduje predykcje dla train i test, liczy ROC AUC jeśli model ma predict_proba albo decision_function
    Tworzy wykres ROC, liczy metryki i zapisuje przykłady FP, FN i pie chart.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = None

    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:,1]
    elif hasattr(model, "decision_function"):
        y_test_proba = model.decision_function(X_test)

    if y_test_proba is not None: 
        roc_auc = roc_auc_score(y_test, y_test_proba)
        plot_roc_curve(y_test, y_test_proba, model_name)
    else: 
        roc_auc = None
    
    train_metrics = calculate_metrics(y_train, y_train_pred, model_name, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, model_name, "test")

    false_sentences(text, y_test_pred, y_test, name = f"{model_name}")
    return train_metrics, test_metrics, roc_auc

def train_model(X_train, X_test, y_train, y_test, text):
    """
    Trenuje klasyfikatory, wykonuje ewaluację. 
    """
    results = {}
    clf = {
        "Logistic Regression": LogisticRegression(random_state=42, solver="saga", class_weight="balanced", C=1.0, max_iter=3000), 
        "LinearSVC": LinearSVC(C=0.25, dual=True, loss="squared_hinge", random_state=42, class_weight="balanced", max_iter=30000), 
        "Ridge Classifier": RidgeClassifier(alpha=2.0, solver="lsqr", class_weight= "balanced", random_state=42)
        }
    
    for model_name, classifier in clf.items():
        print(f"[{model_name}] Trenuję model...")
        classifier.fit(X_train, y_train)
        print(f"[{model_name}] Ewaluuję model...")

        train_metrics, test_metrics, roc_auc = train_and_evaluate_model(
            classifier, X_train, X_test, y_train, y_test, text, model_name)
        print(f"[{model_name}] Tworzę learning curve...")

        plot_learning_curve(classifier, X_train, y_train, model_name)
        results[model_name] = {"params": classifier.get_params(), 
                               "estimator": classifier, 
                               "train": train_metrics, 
                               "test": test_metrics, 
                               "roc_auc": roc_auc
                               }

    return results