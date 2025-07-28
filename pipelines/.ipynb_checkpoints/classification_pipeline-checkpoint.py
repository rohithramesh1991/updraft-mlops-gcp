from kfp.v2.dsl import pipeline, component, Input, Output, Dataset, Model, Metrics

@component(base_image="gcr.io/YOUR_PROJECT_ID/load_data:TAG")
def load_data_op(project: str, dataset: str, table: str, output_path: Output[Dataset]):
    from google.cloud import bigquery
    import pandas as pd
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{dataset}.{table}_prepped`"
    df = client.query(query).to_dataframe()
    df.to_csv(output_path.path, index=False)

@component(base_image="gcr.io/YOUR_PROJECT_ID/preprocess:TAG")
def preprocess_op(input_path: Input[Dataset], output_path: Output[Dataset]):
    import pandas as pd
    df = pd.read_csv(input_path.path)
    # Example cleaning logic -- match your real code!
    df = df[df['unsecured'] > 0].copy()
    df = df.dropna(subset=[
        'mob_12_arrears', 'application_reason', 'source',
        'searches_last_month', 'searches_last_3months',
        'months_since_newest_account'
    ]).copy()
    df.to_csv(output_path.path, index=False)

@component(base_image="gcr.io/YOUR_PROJECT_ID/train:TAG")
def train_op(input_path: Input[Dataset], model_path: Output[Model], x_test_path: Output[Dataset], y_test_path: Output[Dataset]):
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    df = pd.read_csv(input_path.path)
    target_col = 'target_column'  # TODO: change this!
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = XGBClassifier()
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, model_path.path)
    pd.DataFrame(X_test_scaled).to_csv(x_test_path.path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path.path, index=False)

@component(base_image="gcr.io/YOUR_PROJECT_ID/evaluate:TAG")
def evaluate_op(model_path: Input[Model], x_test_path: Input[Dataset], y_test_path: Input[Dataset], report_path: Output[Metrics]):
    import pandas as pd
    import joblib
    from sklearn.metrics import classification_report

    model = joblib.load(model_path.path)
    X_test = pd.read_csv(x_test_path.path)
    y_test = pd.read_csv(y_test_path.path).values.ravel()
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).to_csv(report_path.path)

@pipeline(name="classification-pipeline")
def classification_pipeline(project: str, dataset: str, table: str):
    d = load_data_op(project=project, dataset=dataset, table=table)
    p = preprocess_op(input_path=d.outputs["output_path"])
    t = train_op(input_path=p.outputs["output_path"])
    evaluate_op(
        model_path=t.outputs["model_path"],
        x_test_path=t.outputs["x_test_path"],
        y_test_path=t.outputs["y_test_path"]
    )
