import json, pickle
from pathlib import Path
from datetime import datetime

def save_results_to_json(results, model_info, second_best_model, mcnemar_results):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_summary = {
        "timestamp": timestamp,
        "model": {
            "name": model_info["name"], 
            "f1_score": model_info["f1"]
        }, 
        "all_models": {}
    }

    for name, model_dict in results.items():
        results_summary["all_models"][name] = {
            "params" : model_dict["params"], 
            "train_metrics": model_dict["train"], 
            "test_metrics": model_dict["test"], 
            "roc_auc": model_dict["roc_auc"]
        }

    if mcnemar_results is not None:
        results_summary["mcnemar"] = {
            "model_a": model_info["name"],
            "model_b": second_best_model,
            **mcnemar_results
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
