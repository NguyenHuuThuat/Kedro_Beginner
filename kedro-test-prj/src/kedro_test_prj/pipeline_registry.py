"""Project pipelines."""
from __future__ import annotations
from typing import Dict

# from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro_test_prj.pipelines import (
    pipeline
)

def register_pipelines()-> Dict[str, Pipeline]:
    all_pipeline = pipeline.create_pipeline()

    return {
        "__default__": all_pipeline
    }
