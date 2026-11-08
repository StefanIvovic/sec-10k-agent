"""MLflow setup: traces every LangGraph run and DSPy call into the tracking server."""
import logging

import mlflow

from .config import get_settings

log = logging.getLogger(__name__)
_done = False


def init_mlflow() -> bool:
    global _done
    if _done:
        return True
    s = get_settings()
    try:
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        mlflow.set_experiment(s.mlflow_experiment)
        mlflow.langchain.autolog()   # LangGraph node spans
        mlflow.dspy.autolog()        # DSPy module / LM spans (prompts, outputs, token usage)
        _done = True
    except Exception as e:  # tracing must never break the agent
        log.warning("MLflow tracing disabled: %s", e)
    return _done
