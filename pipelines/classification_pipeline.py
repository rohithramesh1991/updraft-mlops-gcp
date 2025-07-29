from kfp.v2 import dsl
import kfp
from kfp.v2.dsl import importer
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp

# Load your YAML component specs
load_data_op = kfp.components.load_component_from_file('components_yaml/load_data_op.yaml')
preprocess_op = kfp.components.load_component_from_file('components_yaml/preprocess_op.yaml')
train_op = kfp.components.load_component_from_file('components_yaml/train_op.yaml')
evaluate_op = kfp.components.load_component_from_file('components_yaml/evaluate_op.yaml')

@dsl.pipeline(name='classification-pipeline')
def classification_pipeline(project: str, dataset: str, table: str):
    d = load_data_op(project=project, dataset=dataset, table=table)
    p = preprocess_op(input_path=d.outputs["output_path"])

    TrainJobOp = create_custom_training_job_from_component(
        component_spec=train_op,
        display_name="train-xgb-customjob",
        replica_count=1,
        machine_type="n1-standard-2",
        boot_disk_size_gb=100,
        base_output_directory=f"gs://{project}/pipeline-root/train"
    )
    t = TrainJobOp(x_train_path=p.outputs["x_train_path"], y_train_path=p.outputs["y_train_path"])

    e = evaluate_op(model_path=t.outputs["model_path"],
                    x_test_path=p.outputs["x_test_path"],
                    y_test_path=p.outputs["y_test_path"])

    model_importer = importer(
    artifact_uri=t.outputs["model_path"].resource_name,
    artifact_class=artifact_types.UnmanagedContainerModel,
    metadata={
        "containerSpec": {
            "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
        }
    }
)

    uploaded = ModelUploadOp(
        project=project,
        location="us-central1",
        display_name="classification-xgb-model",
        unmanaged_container_model=model_importer.outputs["artifact"]
    )


    deploy_op = ModelDeployOp(
        model=uploaded.outputs["model"],
        deployed_model_display_name="classification-xgb",
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=2
    )
