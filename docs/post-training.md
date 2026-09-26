# AgentForge post-training lab

The post-training lab connects human-reviewed flywheel evidence to reproducible adapter
training, held-out evaluation, and controlled deployment. Heavy ML dependencies are isolated
from the API and dashboard images.

## Capabilities

- converts reviewed chosen/rejected traces into conversational SFT and DPO JSONL;
- removes identical duplicates and rejects conflicting labels;
- deterministically assigns train, validation, and frozen test splits;
- blocks exact and near-duplicate contamination against protected evaluation datasets;
- fingerprints sources, generated datasets, configurations, checkpoints, and run manifests;
- runs LoRA or 4-bit NF4 QLoRA with TRL, Transformers, PEFT, and bitsandbytes;
- supports completion-only SFT and DPO preference optimization;
- records framework versions, seeds, hyperparameters, metrics, trainable parameter counts,
  model cards, and per-artifact SHA-256 digests;
- evaluates locally served adapters through an OpenAI-compatible EvalOps adapter;
- registers only integrity-verified completed runs;
- requires a passing, untampered offline decision before canary status;
- synchronizes model lineage with the existing canary promotion and rollback controller.

## Trust boundaries

- Training accepts only preference records with an explicit reviewer and trace fingerprints.
- Protected evaluation prompts are never placed in training output.
- Remote model endpoints are blocked by default. Set `allow_remote=true` explicitly in a
  local experiment definition when required; API keys are read from an environment variable,
  never stored in the report configuration.
- `trust_remote_code` defaults to false.
- Training writes only to a dedicated empty run directory and never pushes artifacts to a hub.
- A successful loss curve is not release evidence. Promotion still requires held-out quality,
  regression, safety, latency, cost, and human-provenance gates.

## Installation

Create a separate accelerator environment so the production API image does not contain the
training toolchain:

```bash
python -m venv .venv-training
# Windows: .venv-training\Scripts\activate
# Linux/macOS: source .venv-training/bin/activate
python -m pip install -r requirements-post-training.txt
```

QLoRA uses bitsandbytes NF4 quantization. Accelerator and bitsandbytes support varies by
platform, so verify the current hardware compatibility before choosing 4-bit mode. Set
`qlora_4bit=false` for non-quantized LoRA when enough memory is available.

## 1. Create training data

First export only human-backed preferences from the flywheel. An empty export means there is
not yet enough valid preference evidence; do not replace it with invented labels.

```bash
python -m evals.flywheel export-preferences \
  --traces data/evaluations/flywheel/traces.jsonl \
  --output data/evaluations/flywheel/preferences.jsonl

python -m post_training.cli prepare \
  --preferences data/evaluations/flywheel/preferences.jsonl \
  --protected-dataset evals/datasets/agent_quality_seed.jsonl \
  --protected-dataset evals/datasets/code_agent_smoke.jsonl \
  --output-dir data/evaluations/post-training/dataset
```

The builder writes `sft.jsonl`, `dpo.jsonl`, and `manifest.json`. The manifest stores split
counts and integrity hashes. A contamination finding aborts the build.

## 2. Run the CI-safe low-rank smoke test

```bash
python -m post_training.cli smoke \
  --output-dir data/evaluations/post-training/smoke
```

This dependency-free fixture genuinely optimizes rank-two adapter matrices while its base
matrix remains frozen. It verifies gradient flow, convergence, manifests, hashing, model cards,
and registry plumbing. It is deliberately not an LLM and must not be presented as one.

## 3. Run SFT and DPO

Copy and edit the checked-in configurations:

- `post_training/configs/sft_qlora.example.json`
- `post_training/configs/dpo_qlora.example.json`

Then run:

```bash
python -m post_training.cli train --config post_training/configs/sft_qlora.example.json
python -m post_training.cli train --config post_training/configs/dpo_qlora.example.json
```

Use the SFT model or adapter as the DPO base. Target module names are architecture-specific;
inspect the selected model before changing the example. Training aborts when dependencies are
missing, QLoRA has no supported accelerator, the dataset integrity check fails, or the output
directory is non-empty.

## 4. Serve and evaluate

Serve the base and adapter using an OpenAI-compatible runtime such as vLLM or llama.cpp. Keep
each endpoint local unless remote evaluation is an intentional, reviewed decision. Then copy
`post_training/configs/local_adapter_eval.example.json` and run:

```bash
python -m evals.run_experiments \
  --config post_training/configs/local_adapter_eval.example.json
```

The normal EvalOps report compares held-out quality, confidence intervals, failures, latency,
tokens, cost, and slices on identical case IDs.

## 5. Register and release

```bash
python -m post_training.cli register \
  --registry data/evaluations/post-training/registry.json \
  --version adapter-v1 \
  --run-manifest data/evaluations/post-training/runs/sft-v1/run_manifest.json

python -m evals.flywheel gate \
  --report data/evaluations/<adapter-evaluation>.json \
  --baseline base-model --candidate adapter-v1 \
  --output data/evaluations/post-training/promotion.json

python -m post_training.cli registry-canary \
  --registry data/evaluations/post-training/registry.json \
  --version adapter-v1 \
  --decision data/evaluations/post-training/promotion.json
```

After the canary controller promotes or rolls back the version, synchronize its evidence:

```bash
python -m post_training.cli registry-sync \
  --registry data/evaluations/post-training/registry.json \
  --deployment-state data/evaluations/flywheel/deployment.json
```

The registry rejects duplicate versions, modified artifacts, failed decisions, decision
fingerprint changes, version mismatches, and deployment evidence mismatches.

## Dashboard

```bash
streamlit run post_training/dashboard.py
# or
docker compose --profile evaluation up --build post_training_dashboard
```

The read-only container runs as UID 10001 with a read-only root filesystem, dropped Linux
capabilities, no-new-privileges, and only the local result directory mounted read-only.

## What can be claimed

The checked-in implementation proves the data, training, integrity, evaluation, and release
plumbing plus low-rank optimization in CI. It does not include a fabricated GPU result. Publish
an LLM improvement only after collecting reviewed preferences, completing a named-model SFT/DPO
run, and evaluating it on a separately frozen test split.
