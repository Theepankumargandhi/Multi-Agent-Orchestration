"""Evaluation datasets, deterministic graders, and experiment infrastructure."""

from importlib import import_module

# SDK-only controls must not import LangChain or initialize the agent platform.
# Keep the historical public exports available, loading each only when requested.
_EXPORTS = {
    "adaptive_router": ("CostAwareRouter", "RoutingObservation"),
    "calibration": ("JudgeCalibrationReport", "PairwiseCase"),
    "platform": ("AgentRun", "EvalCase", "ExpectedBehavior", "ExperimentConfig", "ExperimentReport",
                 "ExperimentStore", "VariantConfig", "grade_case", "run_experiment"),
    "retrieval_metrics": ("RetrievalCase", "RetrievalReport"),
}


def __getattr__(name):
    for module, names in _EXPORTS.items():
        if name in names:
            value = getattr(import_module(f"evals.{module}"), name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'evals' has no attribute {name!r}")

__all__ = [
    "AgentRun",
    "CostAwareRouter",
    "EvalCase",
    "ExpectedBehavior",
    "ExperimentConfig",
    "ExperimentReport",
    "ExperimentStore",
    "JudgeCalibrationReport",
    "PairwiseCase",
    "RetrievalCase",
    "RetrievalReport",
    "RoutingObservation",
    "VariantConfig",
    "grade_case",
    "run_experiment",
]
