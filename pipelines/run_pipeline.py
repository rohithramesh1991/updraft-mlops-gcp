import os
from google.cloud import aiplatform

def main():
    project = os.environ["GCP_PROJECT"]
    region = os.environ["GCP_REGION"]
    pipeline_spec_path = "pipelines/classification_pipeline.json"
    pipeline_root = os.environ["PIPELINE_ROOT"]
    dataset = os.environ["BQ_DATASET"]
    table = os.environ["BQ_TABLE"]

    aiplatform.init(project=project, location=region)
    job = aiplatform.PipelineJob(
        display_name="classification-pipeline",
        template_path=pipeline_spec_path,
        pipeline_root=pipeline_root,
        parameter_values={
            "project": project,
            "dataset": dataset,
            "table": table,
            "region":region,
            "model_display_name": "xgb-pipeline-model",
            "machine_type": "n1-standard-2",
            "min_replica_count": 1,
            "max_replica_count": 2,
            "traffic_split": '{"0": "100"}'
        }
    )
    job.run(sync=True)

if __name__ == "__main__":
    main()
