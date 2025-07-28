import argparse
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, df):
        df = df.copy()
        cols = [
            'months_since_3plus', 'time_since_default', 'historic_default_balance',
            'active_payday', 'gambling_value'
        ]
        df[cols] = df[cols].replace(99999, np.nan)
        for col in cols:
            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(df[col].median())
        df = df[df['unsecured'] > 0].copy()
        df['age_at_application'] = df['application_finished_at'].dt.year - df['birth_year']
        df['age_bin'] = pd.cut(df['age_at_application'], bins=[16, 29, 39, 49, 64, 81],
                               labels=['16-29','30-39','40-49','50-64','65+'], include_lowest=True, right=False)
        df['is_gambler'] = (df['gambling_value'].fillna(0) != 0).astype(int)
        df['income_per_loan'] = df['gross_income'] / (df['previous_loan_count'] + 1)
        df['revolving_utilization'] = df['revolving_credit_balance'] / (df['revolving_credit_limit'] + 1)
        df['mortgage_utilization'] = df['mortgage_count'] * df['revolving_utilization']
        df['dti_times_gambling'] = df['debt_to_income'] * np.abs(df['gambling_value'].fillna(0))
        df['high_credit_limit'] = (df['revolving_credit_limit'] > df['revolving_credit_limit'].quantile(0.95)).astype(int)
        df['recent_search_ratio'] = df['searches_last_month'] / (df['searches_last_3months'] + 1)
        df['mortgage_and_revolving_ratio'] = (df['mortgage_balance'] + df['revolving_credit_balance']) / (df['gross_income'] + 1)
        df['has_active_payday_and_high_dti'] = ((df['active_payday'].fillna(0) > 0) & (df['debt_to_income'] > 40)).astype(int)
        df['high_unsecured_and_returned_dd'] = ((df['unsecured'] > df['unsecured'].median()) & (df['returned_dd_count'] > 2)).astype(int)
        df['recent_account_opening'] = (df['months_since_newest_account'] < 6).astype(int)
        df['many_recent_searches'] = (df['searches_last_month'] > 3).astype(int)
        df['heavy_returned_dd'] = (df['returned_dd_count'] > 2).astype(int)
        df['heavy_gambler'] = (df['gambling_value'] > df['gambling_value'].quantile(0.90)).astype(int)
        df = pd.get_dummies(df, columns=['application_reason', 'source'], prefix=['app_reason', 'source'])
        df.drop(columns=['mob_6_arrears', 'mob_9_arrears', 'mob_18_arrears', 'application_finished_at'], errors='ignore', inplace=True)
        for col in df.select_dtypes(include=['object', 'category']):
            df[col] = df[col].astype(str)
            df[col] = LabelEncoder().fit_transform(df[col])
        return df

class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.drop_cols_ = []
    def fit(self, X, y=None):
        corr = pd.DataFrame(X).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.drop_cols_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
    def transform(self, X):
        return pd.DataFrame(X).drop(columns=self.drop_cols_, errors='ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--x_train_path', type=str, required=True)
    parser.add_argument('--x_test_path', type=str, required=True)
    parser.add_argument('--y_train_path', type=str, required=True)
    parser.add_argument('--y_test_path', type=str, required=True)
    args = parser.parse_args()

    # --- Directory creation fix ---
    for path in [args.x_train_path, args.x_test_path, args.y_train_path, args.y_test_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    df = pd.read_csv(args.input_path)

    if 'application_finished_at' in df.columns:
        df['application_finished_at'] = pd.to_datetime(df['application_finished_at'], errors='coerce')
        print("application_finished_at dtype:", df['application_finished_at'].dtype)
        print("Nulls in application_finished_at:", df['application_finished_at'].isnull().sum())
    else:
        print("application_finished_at column NOT FOUND!")

    df_features = FeatureEngineering().fit_transform(df)
    excluded_cols = ['vault_id', 'mob_12_arrears']
    df_numeric = df_features.drop(columns=excluded_cols, errors='ignore').select_dtypes(include='number')
    y = df_features['mob_12_arrears'].astype(int)
    X = df_numeric.drop(columns=['mob_12_arrears'], errors='ignore')
    X_filtered = CorrelationFilter(threshold=0.8).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, stratify=y, test_size=0.2, random_state=42)
    X_train.to_csv(args.x_train_path, index=False)
    X_test.to_csv(args.x_test_path, index=False)
    pd.DataFrame(y_train).to_csv(args.y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(args.y_test_path, index=False)
