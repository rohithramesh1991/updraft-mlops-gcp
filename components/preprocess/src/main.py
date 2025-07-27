import argparse
import pandas as pd

def preprocess(input_path, output_path):
    df = pd.read_csv(input_path)
    # Example cleaning logic -- customize as needed:
    df = df[df['unsecured'] > 0].copy()
    df = df.dropna(subset=[
        'mob_12_arrears', 'application_reason', 'source',
        'searches_last_month', 'searches_last_3months',
        'months_since_newest_account'
    ]).copy()
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)
