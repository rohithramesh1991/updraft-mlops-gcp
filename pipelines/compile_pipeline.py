from classification_pipeline import classification_pipeline
from kfp.v2 import compiler
import os

os.makedirs('pipelines', exist_ok=True)

compiler.Compiler().compile(
    pipeline_func=classification_pipeline,
    package_path='pipelines/classification_pipeline.json'
)
