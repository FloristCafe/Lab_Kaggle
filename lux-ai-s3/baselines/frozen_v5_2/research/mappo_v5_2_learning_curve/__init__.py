"""Isolated B0 orchestration; tracking must not consume training RNG."""

import os

# MLflow telemetry asynchronously calls global random.randint in this runtime.
# Disable infrastructure-only random consumers BEFORE importing MLflow.
os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["MLFLOW_ENABLE_ASYNC_LOGGING"] = "false"
