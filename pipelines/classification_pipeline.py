from kfp.v2 import dsl
import kfp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp, ModelDeployOp

# Load existing component YAMLs
load_data_op = kfp.components.load_component_from_file('components_yaml/load_data_op.yaml')
preprocess_op = kfp.components.load_component_from_file('components_yaml/preprocess_op.yaml')
train_op = kfp.components.load_component_from_file('components_yaml/train_op.yaml')
evaluate_op = kfp.components.load_component_from_file('components_yaml/evaluate_op.yaml')

@dsl.pipeline(name='classification-pipeline')
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

    # Use CustomTrainingJobOp with n1-standard-2
    train_job = CustomTrainingJobOp(
        component_spec=train_op.component_spec,
        project=project,
        location="us-central1",  # adjust per your region
        display_name="train-xgb-customjob",
        machine_type="n1-standard-2",
        replica_count=1,
        boot_disk_size_gb=100,
        base_output_directory=f"gs://{project}/pipeline-root/train",
    )
    t = train_job(
        x_train_path=p.outputs["x_train_path"],
        y_train_path=p.outputs["y_train_path"]
    )

    e = evaluate_op(
        model_path=t.outputs["model_path"],
        x_test_path=p.outputs["x_test_path"],
        y_test_path=p.outputs["y_test_path"]
    )

    # Upload and deploy to Vertex AI endpoint
    uploaded = ModelUploadOp(
        project=project,
        location="us-central1",
        display_name="classification-xgb-model",
        artifact_uri=t.outputs["model_path"],
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )
    deploy_op = ModelDeployOp(
        model=uploaded.outputs["model"],
        deployed_model_display_name="classification-xgb",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2
    )