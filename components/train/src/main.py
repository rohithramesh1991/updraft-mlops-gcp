import os
import argparse
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train_path', type=str, required=True)
    parser.add_argument('--y_train_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    X_train = pd.read_csv(args.x_train_path)
    y_train = pd.read_csv(args.y_train_path).values.ravel()

    pipe = Pipeline([
        ('top_k_selector', TopKFeatureSelector(k=20)),
        ('scaler', StandardScaler()),
        ('model', XGBWithAutoWeight(
            eval_metric='aucpr',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            tree_method='hist',
            reg_lambda=5
        ))
    ])
    pipe.fit(X_train, y_train)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(pipe, args.model_path)