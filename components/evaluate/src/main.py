import os
import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from kfp.v2.dsl import Metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--X_test_path', type=str, required=True)
    parser.add_argument('--y_test_path', type=str, required=True)
    parser.add_argument('--report_path', type=str, required=True)
    args = parser.parse_args()
    print(f"[DEBUG] args.report_path = {args.report_path}")

    pipe = joblib.load(args.model_path)
    X_test = pd.read_csv(args.X_test_path)
    y_test = pd.read_csv(args.y_test_path).values.ravel()
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']]

    # Make sure parent dir exists
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    # Write metrics file for the artifact
    report_df.to_csv(args.report_path, index=True)

    # Log summary metrics to Vertex AI pipeline UI
    metrics = Metrics()
    metrics.log_metric("precision_avg", report_df['precision'].mean())
    metrics.log_metric("recall_avg", report_df['recall'].mean())
    metrics.log_metric("f1_avg", report_df['f1-score'].mean())

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    # Precision-Recall Curve
    if hasattr(pipe, "predict_proba"):
        y_scores = pipe.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        avg_precision = average_precision_score(y_test, y_scores)
        plt.figure(figsize=(8, 5))
        plt.plot(recall, precision, marker='.', label=f'AP = {avg_precision:.2f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("pr_curve.png")
