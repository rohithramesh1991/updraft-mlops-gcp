import os
import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from xgboost import XGBClassifier

# --- BEGIN Custom classes required to load pipeline ---
class TopKFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=20):
        self.k = k
        self.top_features_ = []
    def fit(self, X, y):
        model = XGBClassifier(
            eval_metric='aucpr', n_estimators=100, max_depth=3, learning_rate=0.1, tree_method='hist'
        )
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns)
        self.top_features_ = importances.nlargest(self.k).index.tolist()
        return self
    def transform(self, X):
        return X[self.top_features_]

class XGBWithAutoWeight(XGBClassifier, ClassifierMixin):
    def fit(self, X, y, **kwargs):
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        self.scale_pos_weight = neg / pos
        return super().fit(X, y, **kwargs)
# --- END Custom classes ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--x_test_path', type=str, required=True)
    parser.add_argument('--y_test_path', type=str, required=True)
    parser.add_argument('--report_path', type=str, required=True)
    parser.add_argument('--pr_curve_path', type=str, required=True)
    parser.add_argument('--cm_path', type=str, required=True)
    args = parser.parse_args()
    print(f"[DEBUG] args.report_path = {args.report_path}")

    pipe = joblib.load(args.model_path)
    X_test = pd.read_csv(args.x_test_path)
    y_test = pd.read_csv(args.y_test_path).values.ravel()
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']]

    # Make sure parent dir exists
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    # Write metrics file for the artifact
    report_df.to_csv(args.report_path, index=True)

    # Log summary metrics to Vertex AI pipeline UI
    # metrics = Metrics()
    # metrics.log_metric("precision_avg", report_df['precision'].mean())
    # metrics.log_metric("recall_avg", report_df['recall'].mean())
    # metrics.log_metric("f1_avg", report_df['f1-score'].mean())

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(args.cm_path)

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
        plt.savefig(args.pr_curve_path)
