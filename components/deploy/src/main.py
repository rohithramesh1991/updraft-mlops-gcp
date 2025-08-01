import os
import argparse
from google.cloud import aiplatform
import json

def deploy_model(project, region, model_dir, display_name,
                 machine_type, min_replica_count, max_replica_count, traffic_split, endpoint_uri):
    aiplatform.init(project=project, location=region)
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        sync=True,
    )
    endpoint = model.deploy(
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_split=traffic_split,
        sync=True,
    )
    
    # ------- ------------------------------------------------#
    os.makedirs(os.path.dirname(endpoint_uri), exist_ok=True)
    # ------------- ------------------------------------------#
    
    with open(endpoint_uri, "w") as f:
        f.write(endpoint.resource_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--display_name', type=str, required=True)
    parser.add_argument('--machine_type', type=str, required=True)
    parser.add_argument('--min_replica_count', type=int, required=True)
    parser.add_argument('--max_replica_count', type=int, required=True)
    parser.add_argument('--traffic_split', type=str, required=True)
    parser.add_argument('--endpoint_uri', type=str, required=True)
    args = parser.parse_args()
    traffic_split = json.loads(args.traffic_split)
    traffic_split = {k: int(v) for k, v in traffic_split.items()}
    deploy_model(
        project=args.project,
        region=args.region,
        model_dir=args.model_dir,
        display_name=args.display_name,
        machine_type=args.machine_type,
        min_replica_count=args.min_replica_count,
        max_replica_count=args.max_replica_count,
        traffic_split=traffic_split,
        endpoint_uri=args.endpoint_uri,
    )
