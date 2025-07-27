# Updraft ML Pipeline for GCP

## Structure

- Components: Each ML pipeline step is a component (load, preprocess, train, evaluate) with its own Dockerfile.
- Pipeline: Defined in `pipelines/classification_pipeline.py`, compiled and submitted to Vertex AI Pipelines.
- CI/CD: Automated with GitHub Actions.

## Usage

1. Clone repo on Vertex AI Workbench.
2. Build and push component Docker images.
3. Compile and run pipeline.
