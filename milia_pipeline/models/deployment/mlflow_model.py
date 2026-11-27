"""The Models-from-Code script MLflow logs for a MILIA deployment package (:mod:`.registry`).

MLflow copies this file into the logged model and runs it when the model is loaded; it names
the model with ``mlflow.models.set_model``, so no model object is pickled. MILIA never imports
it: the model is :class:`~.mlflow_pyfunc.MiliaPackageModel`.
"""

import mlflow

from milia_pipeline.models.deployment.mlflow_pyfunc import MiliaPackageModel

mlflow.models.set_model(MiliaPackageModel())
