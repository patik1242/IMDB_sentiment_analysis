import json, pickle
from pathlib import Path
from datetime import datetime

def save_results_to_json(results, model_info):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_summary = {
        "timestamp": timestamp,
        "model": {
            "name": "Logistic Regression", 
            "f1_score": model_info["f1"]
        }, 
        "train_metrics": results["train"], 
        "test_metrics": results["test"], 
        "params": results["params"]
    }

    output_path = results_dir/f"results_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    return output_path

def save_model(estimator, vectorizer = None):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    with open(models_dir/"model.pkl", "wb") as f:
        pickle.dump(estimator,f)

    if vectorizer is not None:
        with open(models_dir/"vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer,f)
