from kfp.v2 import dsl
from kfp.v2.dsl import pipeline, Dataset, Model, Metrics
import kfp

# Load component YAMLs
load_data_op = kfp.components.load_component_from_file('components_yaml/load_data_op.yaml')
preprocess_op = kfp.components.load_component_from_file('components_yaml/preprocess_op.yaml')
train_op = kfp.components.load_component_from_file('components_yaml/train_op.yaml')
evaluate_op = kfp.components.load_component_from_file('components_yaml/evaluate_op.yaml')
deploy_model_op = kfp.components.load_component_from_file('components_yaml/deploy_model_op.yaml')

@pipeline(name='classification-pipeline')
def classification_pipeline(
    project: str,
    dataset: str,
    table: str,
    region: str,
    model_display_name: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
    traffic_split: str
):
    d = load_data_op(
        project=project,
        dataset=dataset,
        table=table
    )
    p = preprocess_op(
        input_path=d.outputs["output_path"]
    )
    t = train_op(
        x_train_path=p.outputs["x_train_path"],
        y_train_path=p.outputs["y_train_path"]
    )
    evaluate_op(
        model_dir=t.outputs["model_dir"],
        x_test_path=p.outputs["x_test_path"],
        y_test_path=p.outputs["y_test_path"]
    )
    deploy = deploy_model_op(
        model_dir=t.outputs["model_dir"],
        project=project,
        region=region,
        display_name=model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_split=traffic_split
    )