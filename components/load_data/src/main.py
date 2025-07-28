import os
import argparse
from google.cloud import bigquery
import pandas as pd

def load_data(project, dataset, table, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{dataset}.{table}_prepped`"
    df = client.query(query).to_dataframe()
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--table', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    load_data(args.project, args.dataset, args.table, args.output_path)
