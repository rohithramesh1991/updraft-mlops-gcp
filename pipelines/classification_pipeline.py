from kfp.v2.dsl import pipeline, component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="{{LOAD_IMAGE_URI}}"
)
def load_data_op(
    project: str,
    dataset: str,
    table: str,
    output_path: Output[Dataset]
):
    # This function is just a stub.
    # The real logic is in components/load_data/src/main.py, inside the container.
    pass

@component(
    base_image="{{PREPROCESS_IMAGE_URI}}"
)
def preprocess_op(
    input_path: Input[Dataset],
    X_train_path: Output[Dataset],
    X_test_path: Output[Dataset],
    y_train_path: Output[Dataset],
    y_test_path: Output[Dataset]
):
    # Stub: all logic is in the preprocess container's main.py
    pass

@component(
    base_image="{{TRAIN_IMAGE_URI}}"
)
def train_op(
    X_train_path: Input[Dataset],
    y_train_path: Input[Dataset],
    model_path: Output[Model]
):
    # Stub: all logic is in the train container's main.py
    pass

@component(
    base_image="{{EVALUATE_IMAGE_URI}}"
)
def evaluate_op(
    model_path: Input[Model],
    X_test_path: Input[Dataset],
    y_test_path: Input[Dataset],
    report_path: Output[Metrics]
):
    # Stub: all logic is in the evaluate container's main.py
    pass

@pipeline(name="classification-pipeline")
def classification_pipeline(
    project: str,
    dataset: str,
    table: str
):
    # Step 1: Load Data
    d = load_data_op(
        project=project,
        dataset=dataset,
        table=table
    )
    
    # Step 2: Preprocess
    p = preprocess_op(
        input_path=d.outputs["output_path"]
    )
    
    # Step 3: Train
    t = train_op(
        X_train_path=p.outputs["X_train_path"],
        y_train_path=p.outputs["y_train_path"]
    )
    
    # Step 4: Evaluate
    evaluate_op(
        model_path=t.outputs["model_path"],
        X_test_path=p.outputs["X_test_path"],
        y_test_path=p.outputs["y_test_path"]
    )
