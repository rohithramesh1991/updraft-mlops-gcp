import os
import argparse
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train_path', type=str, required=True)
    parser.add_argument('--y_train_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()

    X_train = pd.read_csv(args.x_train_path)
    y_train = pd.read_csv(args.y_train_path).values.ravel()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Optional, use only if needed
        ('model', XGBClassifier(
            eval_metric='aucpr',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            tree_method='hist',
            reg_lambda=5,
            scale_pos_weight=scale_pos_weight  # Set weight here!
        ))
    ])
    pipe.fit(X_train, y_train)
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(pipe, os.path.join(args.model_dir, "model.joblib"))
