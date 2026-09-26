"""Read-only Streamlit UI for post-training data, runs, and deployment lineage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from post_training.models import DatasetManifest, ModelRegistryState, TrainingRunManifest

RESULTS_DIR = Path(os.getenv("POST_TRAINING_RESULTS_DIR", "data/evaluations/post-training"))


def _load(path: Path, model):
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main() -> None:
    st.set_page_config(page_title="AgentForge Post-Training Lab", page_icon="🧬", layout="wide")
    st.title("🧬 AgentForge Post-Training & Model Lineage Lab")
    st.caption("Human-reviewed data, leakage evidence, adapter training, evaluation, and release lineage.")

    dataset_paths = sorted(RESULTS_DIR.glob("**/manifest.json"))
    datasets = [(path, _load(path, DatasetManifest)) for path in dataset_paths]
    datasets = [(path, item) for path, item in datasets if item is not None]
    st.header("Training datasets")
    if not datasets:
        st.info("Run `python -m post_training.cli prepare ...` to build the first dataset.")
    else:
        selected_path, dataset = st.selectbox(
            "Dataset",
            datasets,
            format_func=lambda item: f"{item[1].dataset_id} | {item[1].unique_prompts} reviewed prompts",
        )
        columns = st.columns(4)
        columns[0].metric("Unique prompts", dataset.unique_prompts)
        columns[1].metric("Train", dataset.split_counts.get("train", 0))
        columns[2].metric("Validation", dataset.split_counts.get("validation", 0))
        columns[3].metric("Frozen test", dataset.split_counts.get("test", 0))
        st.write(
            {
                "manifest": str(selected_path),
                "fingerprint": dataset.manifest_fingerprint,
                "human_reviewed": dataset.all_records_human_reviewed,
                "duplicates_removed": dataset.duplicate_records_removed,
                "leakage_findings": len(dataset.leakage_findings),
                "protected_datasets": dataset.protected_dataset_sha256,
            }
        )

    run_paths = sorted(RESULTS_DIR.glob("**/run_manifest.json"))
    runs = [(path, _load(path, TrainingRunManifest)) for path in run_paths]
    runs = [(path, item) for path, item in runs if item is not None]
    st.header("Training runs")
    if not runs:
        st.info("No verified training run manifests found.")
    else:
        st.dataframe(
            [
                {
                    "run": run.run_name,
                    "stage": run.stage,
                    "base_model": run.base_model,
                    "dataset": run.dataset_id,
                    "trainable_%": round(run.trainable_percentage, 4),
                    "loss": run.metrics.get("train_loss", run.metrics.get("final_loss", "")),
                    "fingerprint": run.run_fingerprint,
                }
                for _, run in runs
            ],
            width="stretch",
        )
        _, selected_run = st.selectbox(
            "Inspect run", runs, format_func=lambda item: f"{item[1].run_name} | {item[1].run_id}"
        )
        left, right = st.columns(2)
        left.write("Metrics")
        left.json(selected_run.metrics)
        right.write("Framework versions")
        right.json(selected_run.framework_versions)
        st.write("Integrity-bound artifacts")
        st.dataframe([item.model_dump() for item in selected_run.artifacts], width="stretch")

    registry_path = RESULTS_DIR / "registry.json"
    registry = _load(registry_path, ModelRegistryState)
    st.header("Model registry and release state")
    if registry is None:
        st.info("Register a verified run to create model lineage.")
    else:
        columns = st.columns(3)
        columns[0].metric("Registered models", len(registry.models))
        columns[1].metric("Production", registry.production_version or "none")
        columns[2].metric("Canary", registry.canary_version or "none")
        st.dataframe([item.model_dump(mode="json") for item in registry.models], width="stretch")
        st.write("Audit timeline")
        st.dataframe(registry.audit_log, width="stretch")

    evaluation_paths = sorted(RESULTS_DIR.glob("**/evaluation.json"))
    if evaluation_paths:
        st.header("Base versus adapter evaluation")
        selected = st.selectbox("Evaluation artifact", evaluation_paths, format_func=lambda path: path.name)
        try:
            st.json(json.loads(selected.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            st.error("Evaluation artifact is invalid.")


if __name__ == "__main__":
    main()
