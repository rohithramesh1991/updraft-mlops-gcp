from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp, EndpointCreateOp
from kfp.dsl import pipeline
import kfp

# Load component YAMLs
load_data_op = kfp.components.load_component_from_file('components_yaml/load_data_op.yaml')
preprocess_op = kfp.components.load_component_from_file('components_yaml/preprocess_op.yaml')
train_op = kfp.components.load_component_from_file('components_yaml/train_op.yaml')
evaluate_op = kfp.components.load_component_from_file('components_yaml/evaluate_op.yaml')

@pipeline(name='classification-pipeline')
def classification_pipeline(
    project: str,
    dataset: str,
    table: str
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
        model_path=t.outputs["model_path"],
        x_test_path=p.outputs["x_test_path"],
        y_test_path=p.outputs["y_test_path"]
    )
    
    uploaded = ModelUploadOp(
        project=project,
        location=project and None,  # use default region from pipeline runtime if omitted
        display_name="classification-xgb-model",
        unmanaged_container_model=t.outputs["model_path"]
    )

    # (Optional) Create a dedicated endpoint first
    endpoint = EndpointCreateOp(
        project=project,
        location=None,  # use pipeline's default region
        display_name="classification-endpoint"
    )

    # Deploy the model to endpoint with autoscaling
    deploy_task = ModelDeployOp(
        model=uploaded.outputs['model'],
        endpoint=endpoint.outputs['endpoint'],
        deployed_model_display_name="classification-xgb",
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=2,
        traffic_split={"0": "100"}
    )
