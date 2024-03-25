from kedro.pipeline import node, pipeline
from .nodes import *


def create_pipeline(**kwargs):
    return pipeline([
        # node(
        #     func=read_csv_file,
        #     inputs="raw_data",
        #     outputs="raw_dataframe",
        #     name="nodel_read_raw_data"
        #     ),
        node(
            func=drop_features,
            inputs="raw_data",
            outputs="intermediate_data",
            name="node_intermediate_data"
            ),
        node(
            func=process_outlier_fts,
            inputs="intermediate_data",
            outputs="intermediate_data_p2",
            name="node_intermediate_data_p2"
            ),
        node(
            func=get_dummies,
            inputs="intermediate_data_p2",
            outputs="processed_data",
            name="node_process_data"
            ),
        node(
            func=get_train_test_dataset,
            inputs=["processed_data", "params:test_size"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="node_train_test_split"
            ),
        node(
            func=train_logistic_regression,
            inputs=["X_train", "y_train"],
            outputs="lr_model",
            name="node_train_model_lr"
            ),
        node(
            func=train_knn,
            inputs=["X_train", "y_train"],
            outputs="knn_model",
            name="node_train_model_knn"
            ),
        node(
            func=evalues_model,
            inputs=["lr_model", "X_test", "y_test"],
            outputs="performance_lr_model",
            name="node_evalues_model_lr"
            ),
        node(
            func=evalues_model,
            inputs=["knn_model", "X_test", "y_test"],
            outputs="performance_knn_model",
            name="node_evalues_model_knn"
            )
    ])