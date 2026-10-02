# Verifier shadows: do good step scores lead to better choices?

The previous upgrade learned from independently reviewed workflow checks. That is useful evidence,
but it answers a narrower question than deployment needs: predicting step correctness is not the
same as choosing a correct, safe final answer.

This iteration connects those two questions without changing what users receive. A registered
candidate verifier scores the same candidates the existing selector already saw. Its proposed
choice is frozen before reviews, then compared with the incumbent using delayed correctness and
safety labels on fresh task families.

```mermaid
flowchart TD
    Model[Signed reviewed verifier candidate] --> Register[Register training provenance, incumbent, compute policy, and thresholds]
    Register --> Study[Immutable tenant-scoped study]
    Pool[Actual generated candidate pool] --> Baseline[Existing selector and release gates]
    Baseline --> Serve[Unchanged served answer or abstention]
    Pool --> Features[Consented frozen workflow proxies]
    Features --> Shadow[Candidate PRM plus the same selector]
    Study --> Shadow
    Shadow --> Compare[Pre-review comparison with unchanged eligibility]
    Baseline --> Compare
    Compare --> Reviews[Delayed terminal correctness and safety reviews]
    Reviews --> Cohort[Fresh-family frozen cohort, including abstentions]
    Ledger[Shared one-use holdout ledger] --> Gate
    Cohort --> Curves[Descriptive risk and coverage curves]
    Cohort --> Gate[Registered primary-threshold risk and paired-utility gate]
    Gate -->|unsupported, missing, unsafe, or worse| Hold[Hold report]
    Gate -->|supported runtime evidence| Owner[Owner review only]
```

## What is protected

Registration binds the candidate's independent model-key signature to its signed training cohort
and passing training report. Those original sources must remain valid. It also fixes the incumbent
fingerprint, compute policy, score threshold grid, primary threshold, embargo, and support gates.
Changing any of them requires a new study and fresh held-out families.

Capture reruns the existing selector on the transient original candidate assessments and checks the
incumbent receipt exactly. It then runs the **same selector** with candidate PRM scores substituted.
Grounding, conformal decisions, confidence gains, consensus, budgets, and deterministic tie order
remain unchanged. The shadow may withhold a proposed choice below its primary score threshold;
it can never reopen an incumbent abstention. Consensus snapshots preserve the unrounded Jaccard
value, including decisions close to a threshold.

Only evaluated candidates enter the frozen pool. Complete workflow snapshots are required before
the comparison, and both terminal and step reviews must follow capture. Existing model identities
are recorded during scoring; a model change across candidates makes shadow capture unavailable.
Preference serving is intentionally excluded from this study so two learned selectors are not
changed at once. Preference shadows can still run separately when preference serving is disabled.

All persisted data is typed metadata and keyed identifiers. No additional LLM calls, prompts,
answers, evidence text, claim keys, raw candidate names, or hidden reasoning are stored. Snapshot
proxies still come from the final grounding report, not instrumented reasoning trajectories.
Storage, provenance, or study errors disable this observer and leave the served answer unchanged.
The observer adds CPU and SQLite work: no additional generation calls does not mean zero request
overhead. This local reference store revalidates tenant metadata; measure that overhead and bound
retention before using it on high-volume traffic.

## Run a study

Start with a passing runtime candidate, cohort and report from
[independent workflow supervision](reviewed-process-supervision.md). Set `EXECUTION_REPLAY_KEY` and
`PROCESS_SUPERVISION_MODEL_KEY` locally; both must be independent secrets of at least 32 bytes.
Do not use a synthetic drill candidate for runtime collection.

Create a private compute-policy JSON matching the active `_adaptive_compute_policy()` configuration.
A minimal private trial-policy JSON is:

```json
{
  "primary_threshold": 0.6,
  "curve_thresholds": [0.0, 0.25, 0.5, 0.6, 0.75, 1.0],
  "embargo_seconds": 60.0
}
```

Keep policy and exported evidence files under ignored `data/verifier-shadow/` or another protected
location. Register **before** collecting the new requests:

```bash
python -m evals.verifier_shadow_evaluation --tenant TENANT register \
  --name verifier-study-v1 --candidate data/process-supervision/candidate-v1.json \
  --training-cohort data/process-supervision/cohort-v1.json \
  --training-report data/process-supervision/report-v1.json \
  --compute-policy data/verifier-shadow/compute-policy.json \
  --trial-policy data/verifier-shadow/trial-policy.json \
  --incumbent-fingerprint confidence-only
```

Use `confidence-only` only when `PROCESS_REWARD_MODEL_ENABLED=false`. Otherwise supply the active
single-model or ensemble `artifact_fingerprint`. Runtime capture checks that exact identity; it does
not install or load the candidate as the active verifier.

```dotenv
EXECUTION_REPLAY_ENABLED=true
PROCESS_SUPERVISION_ENABLED=true
VERIFIER_SHADOW_ENABLED=true
PREFERENCE_RANKING_ENABLED=false
# Set VERIFIER_SHADOW_STUDY_ID to the ID returned by register.
```

Requests need strict replay consent, tenant identity, a stable request ID, and a preassigned task
family. The embargo starts at registration. Families observed earlier—even outside the study—cannot
be recycled as fresh evidence, and only their first request can enter the cohort.
Family IDs are caller-supplied, not automatically inferred from semantic similarity. Repeated or
near-duplicate tasks must share stable family IDs; inventing new IDs defeats that boundary.

```bash
python -m evals.verifier_shadow_evaluation --tenant TENANT queue STUDY_ID
python -m agent.execution_replay --tenant TENANT review EVENT_ID --verdict correct --reviewer REVIEWER_ID
python -m evals.verifier_shadow_evaluation --tenant TENANT freeze STUDY_ID \
  --output data/verifier-shadow/cohort-v1.json
python -m evals.verifier_shadow_evaluation --tenant TENANT evaluate \
  data/verifier-shadow/cohort-v1.json --output data/verifier-shadow/report-v1.json --require-gate
```

Use the replay review command's `--unsafe` flag when needed. The queue covers the union of the
incumbent and every registered curve choice, including agreement audits, not only disagreements.
Human reviewers must consult authorized original context separately; metadata cannot establish
semantic correctness. Immutable ambiguous outcomes need a new, audited data-collection decision,
not a silent label overwrite.

## What the gate measures

Every captured fresh-family request remains in the denominator, including incumbent and shadow
abstentions. Uncaptured traffic is outside this study; it is not an all-traffic estimate.

The primary threshold requires at least 20 reviewed releases per route/risk scope, 50% release
coverage, 80% released-outcome review coverage, an error upper bound no greater than 0.2, and no
reviewed unsafe proposed choices. Overall paired review coverage must be at least 80%, at least 20
families must exist, and at least 20% of choices must differ. All-abstain policies cannot pass.

Paired utility is `+1` for safe correct, `-1` for incorrect, `-3` for unsafe, and `0` for abstention.
Unknown or ambiguous correctness gets the interval `[-3, +1]`; a known unsafe annotation remains
actionable even if correctness is ambiguous. Missing proposed releases count as errors for risk.
Different choices use pessimistic utility endpoints; identical choices have zero paired difference
because they share the same outcome. Both scope-level and overall family-bootstrap lower bounds
must be nonnegative. This makes incomplete reviews visible rather than silently removing them.

Error bounds are per-scope Wilson diagnostics, not simultaneous guarantees across scopes. Bootstrap
bounds use 1,000 deterministic resamples and need more diverse families for credible production
inference. The registered grid is **descriptive only**: looking at a curve does not authorize choosing
a better threshold from this same holdout. Threshold changes require fresh-family evaluation.

The shared ledger reserves every cohort family before any curve is scored. Completed exact retries
return the cached signed report; interrupted studies remain exposed. Do not reset or fork the ledger
to reuse evidence. Original training-source or comparison-source deletion/tampering blocks later
capture and cached evaluation. Tenant deletion removes private study/comparison rows but leaves
holdout exposure; exported files require their own retention policy.

Outputs are immutable and active model/SQLite paths are protected. Even a passing report says only
`ready_for_owner_review`; it creates no approval lease, deployed model, active artifact, or automatic
promotion. No generation-cost saving is claimed: both selectors reuse the full generated pool.

## Reproduce the controls

```bash
python -m evals.verifier_shadow_evaluation drill --require-gate
python -m pytest -q tests/test_verifier_shadow.py tests/test_process_supervision.py
```

The authored synthetic drill trains on the earlier workflow template and evaluates 40 later
families. Four incumbent abstentions are retained. The clean control proposes 36 safe correct
releases; the missing-review and unsafe-shift controls are held. This deliberately learnable
template tests plumbing, signature round trips, eligibility preservation, missingness, and gate
behavior—not live factuality, causal traffic lift, OOD robustness, or broad reasoning improvement.

| Synthetic control | Families retained | Proposed releases | Worst-case paired utility lower bound | Gate |
|---|---:|---:|---:|---|
| Complete reviews | 40 | 36 | +1.55 | Pass controls only |
| 50% paired review coverage | 40 | 36 | -2.00 | Hold |
| Unsafe answer shift | 40 | 36 | -3.90 | Hold |
