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
            "pipeline_root": pipeline_root
        }
    )
    job.run(sync=True)

if __name__ == "__main__":
    main()
