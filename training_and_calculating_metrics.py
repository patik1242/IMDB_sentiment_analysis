from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

def calculate_metrics(y_true, y_pred, model_name, split):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true,y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nMetryki dla {model_name} ({split}): \n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1
    }

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:,1]
    roc_auc = roc_auc_score(y_test, y_test_proba)

    train_metrics = calculate_metrics(y_train, y_train_pred, model_name, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, model_name, "test")

    return train_metrics, test_metrics, roc_auc

def train_model(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=42, solver="saga", class_weight="balanced", C=10)
    clf.fit(X_train, y_train)

    train_metrics, test_metrics, roc_auc = train_and_evaluate_model(clf, X_train, X_test, y_train, y_test, "Logistic Regression")

    return {
        "Logistic Regression": {
            "params": clf.get_params(), 
            "estimator": clf,
            "train": train_metrics, 
            "test": test_metrics, 
            "roc_auc": roc_auc
        }
    }