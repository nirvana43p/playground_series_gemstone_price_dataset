from pathlib import Path
import json

def save_simple_metrics_report(validation_score, best_params, model_name, SEED):
    metrics = {
               "Model Name" : model_name,
               "RMSE - CV" : validation_score,
               "Best Hyperparameters": best_params,
               "SEED" : SEED
              }
    metric_path = Path("notes/report.json")
    metric_path.write_text(json.dumps(metrics))