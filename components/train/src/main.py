import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def train(input_path, model_path, X_test_path, y_test_path):
    df = pd.read_csv(input_path)
    target_col = 'target_column'  # TODO: change this to your real target column
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, model_path)
    pd.DataFrame(X_test_scaled).to_csv(X_test_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--X_test_path', type=str, required=True)
    parser.add_argument('--y_test_path', type=str, required=True)
    args = parser.parse_args()
    train(args.input_path, args.model_path, args.X_test_path, args.y_test_path)
