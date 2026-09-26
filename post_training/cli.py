"""Command-line entry point for AgentForge post-training workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.flywheel import DeploymentState, PromotionDecision
from post_training.dataset import build_training_datasets
from post_training.registry import ModelRegistry
from post_training.smoke import run_lora_smoke
from post_training.trainer import load_training_config, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, train, and register AgentForge adapters.")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build reviewed, leakage-checked SFT/DPO datasets.")
    prepare.add_argument("--preferences", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--protected-dataset", type=Path, action="append", default=[])
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--test-fraction", type=float, default=0.2)

    smoke = commands.add_parser("smoke", help="Run the dependency-free low-rank optimizer smoke test.")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=42)

    training = commands.add_parser("train", help="Run TRL SFT or DPO from a JSON config.")
    training.add_argument("--config", type=Path, required=True)

    register = commands.add_parser("register", help="Register a verified training run.")
    register.add_argument("--registry", type=Path, required=True)
    register.add_argument("--version", required=True)
    register.add_argument("--run-manifest", type=Path, required=True)

    listing = commands.add_parser("registry-list", help="Print the local model registry.")
    listing.add_argument("--registry", type=Path, required=True)

    canary = commands.add_parser("registry-canary", help="Attach a passing offline gate to a model.")
    canary.add_argument("--registry", type=Path, required=True)
    canary.add_argument("--version", required=True)
    canary.add_argument("--decision", type=Path, required=True)

    sync = commands.add_parser("registry-sync", help="Synchronize with the audited canary controller.")
    sync.add_argument("--registry", type=Path, required=True)
    sync.add_argument("--deployment-state", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = build_training_datasets(
            args.preferences,
            args.output_dir,
            protected_paths=args.protected_dataset,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        ).model_dump(mode="json")
    elif args.command == "smoke":
        result = run_lora_smoke(args.output_dir, seed=args.seed).model_dump(mode="json")
    elif args.command == "train":
        result = train(load_training_config(args.config)).model_dump(mode="json")
    elif args.command == "register":
        result = ModelRegistry(args.registry).register(
            args.version, args.run_manifest
        ).model_dump(mode="json")
    elif args.command == "registry-list":
        result = ModelRegistry(args.registry).load().model_dump(mode="json")
    elif args.command == "registry-canary":
        decision = PromotionDecision.model_validate_json(args.decision.read_text(encoding="utf-8"))
        result = ModelRegistry(args.registry).promote_to_canary(
            args.version, decision
        ).model_dump(mode="json")
    else:
        deployment = DeploymentState.model_validate_json(
            args.deployment_state.read_text(encoding="utf-8")
        )
        result = ModelRegistry(args.registry).sync_deployment(deployment).model_dump(mode="json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
