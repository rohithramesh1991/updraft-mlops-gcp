# from kfp.v2 import dsl
# from kfp.v2.dsl import pipeline, Dataset, Model, Metrics
# import kfp

# # Load component YAMLs
# load_data_op = kfp.components.load_component_from_file('components_yaml/load_data_op.yaml')
# preprocess_op = kfp.components.load_component_from_file('components_yaml/preprocess_op.yaml')
# train_op = kfp.components.load_component_from_file('components_yaml/train_op.yaml')
# evaluate_op = kfp.components.load_component_from_file('components_yaml/evaluate_op.yaml')

# @pipeline(name='classification-pipeline')
# def classification_pipeline(
#     project: str,
#     dataset: str,
#     table: str
# ):
#     d = load_data_op(
#         project=project,
#         dataset=dataset,
#         table=table
#     )
#     p = preprocess_op(
#         input_path=d.outputs["output_path"]
#     )
#     t = train_op(
#         X_train_path=p.outputs["X_train_path"],
#         y_train_path=p.outputs["y_train_path"]
#     )
#     evaluate_op(
#         model_path=t.outputs["model_path"],
#         X_test_path=p.outputs["X_test_path"],
#         y_test_path=p.outputs["y_test_path"]
#     )
import kfp
preprocess_op = kfp.components.load_component_from_file('components_yaml/preprocess_op.yaml')
print(preprocess_op.__dict__) 