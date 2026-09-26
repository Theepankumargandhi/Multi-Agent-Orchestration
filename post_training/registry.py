"""Tamper-evident local registry for trained adapters and deployment lineage."""

from __future__ import annotations

from pathlib import Path

from evals.flywheel import DeploymentState, PromotionDecision, verify_promotion_decision
from post_training.models import ModelRecord, ModelRegistryState, utc_now
from post_training.trainer import verify_run_manifest


class ModelRegistry:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ModelRegistryState:
        if not self.path.exists():
            return ModelRegistryState()
        return ModelRegistryState.model_validate_json(self.path.read_text(encoding="utf-8"))

    def _save(self, state: ModelRegistryState) -> ModelRegistryState:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        return state

    def register(
        self, version: str, run_manifest_path: Path, metadata: dict | None = None
    ) -> ModelRecord:
        run_manifest_path = run_manifest_path.resolve()
        manifest = verify_run_manifest(run_manifest_path)
        if manifest.status != "completed":
            raise ValueError("only completed training runs may be registered")
        state = self.load()
        if any(item.version == version for item in state.models):
            raise ValueError(f"model version already exists: {version}")
        record = ModelRecord(
            version=version,
            run_id=manifest.run_id,
            stage=manifest.stage,
            base_model=manifest.base_model,
            dataset_id=manifest.dataset_id,
            run_fingerprint=manifest.run_fingerprint,
            manifest_path=str(run_manifest_path),
            artifact_sha256=manifest.run_fingerprint,
            metadata=metadata or {},
        )
        state.models.append(record)
        state.audit_log.append(
            {
                "at": utc_now(),
                "event": "model_registered",
                "version": version,
                "run_fingerprint": manifest.run_fingerprint,
            }
        )
        self._save(state)
        return record

    def promote_to_canary(
        self, version: str, decision: PromotionDecision
    ) -> ModelRegistryState:
        if not decision.approved_for_canary or not verify_promotion_decision(decision):
            raise ValueError("a valid passing promotion decision is required")
        if decision.candidate != version:
            raise ValueError("promotion candidate must exactly match the registered model version")
        state = self.load()
        target = next((item for item in state.models if item.version == version), None)
        if target is None:
            raise ValueError(f"unknown model version: {version}")
        run_manifest = verify_run_manifest(Path(target.manifest_path))
        if run_manifest.run_fingerprint != target.run_fingerprint:
            raise ValueError("registered model lineage no longer matches its training run")
        if state.canary_version and state.canary_version != version:
            raise ValueError("another model version is already in canary")
        target.status = "canary"
        target.promotion_evidence_fingerprint = decision.decision_fingerprint
        state.canary_version = version
        state.audit_log.append(
            {
                "at": utc_now(),
                "event": "model_entered_canary",
                "version": version,
                "evidence": decision.decision_fingerprint,
            }
        )
        return self._save(state)

    def sync_deployment(self, deployment: DeploymentState) -> ModelRegistryState:
        """Promote or retire registry records from the audited canary controller state."""
        state = self.load()
        by_version = {item.version: item for item in state.models}
        if deployment.status == "stable" and deployment.active_version in by_version:
            active = by_version[deployment.active_version]
            if active.promotion_evidence_fingerprint != deployment.decision_fingerprint:
                raise ValueError("deployment evidence does not match registry promotion evidence")
            if state.production_version and state.production_version in by_version:
                by_version[state.production_version].status = "retired"
            active.status = "production"
            state.production_version = active.version
            state.canary_version = None
            state.audit_log.append(
                {
                    "at": utc_now(),
                    "event": "model_promoted_to_production",
                    "version": active.version,
                    "evidence": deployment.decision_fingerprint,
                }
            )
        elif deployment.status == "rolled_back" and state.canary_version in by_version:
            failed = by_version[state.canary_version]
            failed.status = "retired"
            state.audit_log.append(
                {"at": utc_now(), "event": "model_rolled_back", "version": failed.version}
            )
            state.canary_version = None
        return self._save(state)
