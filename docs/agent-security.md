# Agent Security Red-Team Lab

## Purpose

The coding agent treats model output as a proposal, not authorization. `SecurityPolicyEngine` evaluates every proposed read, write, delete, search, test, and finish action before the corresponding workspace method can run. This creates complete mediation at the application layer in addition to the existing Docker, filesystem, typed-action, reviewer, quality-gate, and owner-approval controls.

The lab is deterministic and credential-free. It is intended to demonstrate measurable security engineering, not to claim that regex policy alone solves agent security.

## Threat coverage

The versioned corpus in `evals/datasets/agent_security_redteam.jsonl` maps scenarios to OWASP LLM and Agentic Security Initiative identifiers and selected MITRE ATLAS techniques. It exercises:

- direct and indirect prompt injection and agent goal hijacking;
- unsafe tool use, unexpected code execution, and dependency on shell primitives;
- path traversal, protected control-plane paths, and identity/privilege abuse;
- credential disclosure and environment-variable exfiltration;
- malicious test weakening and reviewer/evaluation tampering;
- reserved-memory poisoning and unbounded resource consumption;
- attacks arriving through users, repositories, retrieval, tool output, memory, and peer agents.

Benign controls include ordinary inspection, source changes, regression tests, safe YAML parsing, a credential-free HTTP client, test execution, and completion. They keep the suite from optimizing only for refusal.

## Enforcement model

The policy produces a structured event containing the policy version, stable decision ID, action capability, target path, stage, allow/block result, severity, matched rule IDs, risk categories, source taints, approval requirement, and latency.

High-confidence rules block before mutation. Indirect injection found only in prior untrusted context marks a subsequent privileged action for the existing human-approval boundary. Raw repository or tool content is not copied into security events; provenance artifacts contain SHA-256 and non-secret taint labels.

Current rules are deliberately narrow:

| Rule | Purpose |
|---|---|
| `AGENT-PATH-001` | Reject traversal, absolute, secret, and protected control-plane paths |
| `AGENT-SECRET-001` | Reject credential-like values in generated writes |
| `AGENT-EXEC-001` | Reject common dynamic execution, unsafe deserialization, shell, and TLS bypass primitives |
| `AGENT-EXFIL-001` | Reject code combining a network sink with a sensitive source |
| `AGENT-TEST-001` | Reject common whole-suite bypass patterns |
| `AGENT-DOS-001` | Reject recognizable unbounded shell resource attacks |
| `AGENT-INJECTION-001` | Reject privileged actions that carry instruction-override language |

Post-patch scans and independent review remain active. The layers are intentionally redundant because action-time and artifact-time controls catch different failure modes.

## Run and gate

```bash
python -m code_agent.security_evaluation \
  --max-asr 0.05 \
  --min-benign-pass-rate 0.95 \
  --output data/evaluations/security/latest.json
```

The command compares an intentionally permissive baseline with the defended policy. CI fails if defended attack success exceeds the configured ceiling or benign pass rate falls below the floor. The report includes:

- attack success and containment rates;
- benign pass and false-positive rates;
- secret leakage and unsafe-tool-call rates;
- approval rate and p50/p95 policy latency;
- attack-success slices by risk category and ingress surface;
- replay identifiers, framework mappings, decision evidence, and dataset/policy/action/context fingerprints.

Replay a scenario by selecting its case ID from the committed JSONL corpus and running the same fingerprinted policy version. Generated reports omit raw context and action content, preventing the evaluation artifact itself from becoming a credential or injection store.

Open the local dashboard:

```bash
streamlit run code_agent/security_dashboard.py
```

Or run the hardened, read-only dashboard container on port 8505:

```bash
docker compose --profile evaluation up --build security_dashboard
```

## Honest limitations and next production steps

The checked-in suite is a small deterministic regression corpus, not evidence of universal robustness. Pattern matching can be evaded and can create false positives outside the measured benign set. A production deployment should add AST/data-flow scanning, per-tenant capability grants, a separately deployed policy decision point, signed policy bundles, broader multilingual mutation attacks, human-authored holdouts, continuous attack generation, and isolated microVM workers for hostile public repositories.
