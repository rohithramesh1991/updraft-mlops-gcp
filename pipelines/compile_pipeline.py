from classification_pipeline import classification_pipeline
from kfp.v2 import compiler

compiler.Compiler().compile(
    pipeline_func=classification_pipeline,
    package_path='updraft-mlops-gcp/pipelines/classification_pipeline.json'
)
