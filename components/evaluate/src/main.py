import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate(model_path, X_test_path, y_test_path, report_path):
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv(report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--X_test_path', type=str, required=True)
    parser.add_argument('--y_test_path', type=str, required=True)
    parser.add_argument('--report_path', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model_path, args.X_test_path, args.y_test_path, args.report_path)
