# Claim-level grounding and hallucination control

AgentForge includes an opt-in release gate between answer generation and evaluation. When enabled,
the response node creates a private draft without streaming it to the client. A grounding verifier
maps factual claims to retrieved web, local-RAG, knowledge-graph, or calculator evidence. Only a
verified, repaired, or explicit abstention message is released.

## Decision pipeline

1. Retrieved evidence is divided into bounded, fingerprinted evidence items.
2. The draft is divided into factual claims; questions, source lists, and honest failure messages are
   excluded.
3. Each claim is scored with lexical containment and a dependency-free hashed semantic baseline.
4. Numerical, time-sensitive, guarantee, security, legal, and medical-style claims use a stricter
   support threshold.
5. A claim is not supported when matching evidence states the opposite (see below).
6. Markdown citation URLs must appear in the retrieved-source allowlist.
7. Weighted claim coverage and citation precision produce an explainable policy confidence score.
8. The policy releases the draft, retains only supported claims, or fails closed with an abstention.

Each report includes claim/evidence fingerprints, component scores, unsupported high-risk and
contradicted counts, the policy decision, and an optional HMAC-bound receipt. Ordinary `general`
conversation is marked `not_required` so the control does not force retrieval for every chat message.

## Contradiction checks (policy v2)

Word overlap alone rated "The model is available on Groq" at 0.90 support against "The model is not
available on Groq". For each clause of a claim, the verifier now finds the evidence sentence about the
same subject (at least half of the claim's content words) and checks three signals:

- **Numbers:** the claim has a number the sentence lacks while the sentence states its own numbers
  ("released in 2023" against "released in 2025").
- **Negation:** one side is negated and the other is not, and the negation governs a word the other
  side contains ("not available" against "available"). Negation of other words, as in "runs tests,
  not shell commands", does not count.
- **Opposite words:** the claim uses one side of a pair such as rose/fell, passed/failed or
  enabled/disabled while the sentence uses only the other side.

A contradicted claim is never released: it blocks a pass regardless of coverage, and repair removes
it. Citation markers and retrieval scores (`page=22`, `bm25=6.6`) are ignored as facts.

Measured on authored and real data (Groq `openai/gpt-oss-120b` answers over this repository's docs):

| Set | Before (v1) | After (v2) |
|---|---|---|
| 12 authored contradictions | 0 caught | 11 caught |
| 8 held-out contradictions, written after the rules | not measured | 8 caught |
| 20 faithful paraphrases | 20 kept | 19 kept (the miss is a v1 lexical-score limit) |
| 69 claims from real answers (used while tuning) | not measured | 0 false contradictions |
| 44 claims from fresh real answers (not used for tuning) | not measured | 1 false contradiction |

Limits: before/after is deliberately not checked, because "A before B" also reads as "B after A".
Paraphrased opposites outside the word lists, unit conversions and implicit negation are not
detected. This is a deterministic heuristic, not entailment.

## Safe streaming behavior

With verification disabled, token streaming behaves as before. When enabled, the model call does not
stream draft tokens. This is intentional: a hallucinated sentence cannot be recalled after an SSE
client has received it. The graph emits one final AI message after verification or repair.

## Configuration

```dotenv
GROUNDING_VERIFICATION_ENABLED=true
GROUNDING_MIN_CLAIM_SCORE=0.22
GROUNDING_MIN_COVERAGE=0.80
GROUNDING_FAIL_CLOSED=true
GROUNDING_INTEGRITY_KEY=replace-with-a-dedicated-16-character-minimum-secret
```

Start with the checked-in defaults, then tune thresholds on a human-labelled, domain-representative
claim/evidence holdout. A production implementation should replace or ensemble the hashing baseline
with a calibrated NLI/cross-encoder or constrained LLM judge, while retaining deterministic URL and
high-risk policy checks.

## Evaluation

```bash
python -m evals.grounding_evaluation \
  --output data/evaluations/grounding/latest.json \
  --min-pass-rate 1 \
  --max-unsafe-release-rate 0 \
  --min-receipt-integrity 1
```

The 14 checked-in cases exercise supported releases, unsupported facts, fabricated citations,
partial repair, unsupported numerical claims, web/RAG/graph/math evidence, missing evidence,
general-chat bypass, honest retrieval failure, artifact tampering, and contradicted claims. CI publishes the report, and
the AgentOps command center displays it under **Grounding gate**.

This suite validates policy mechanics against synthetic cases. It does not establish a universal
hallucination rate or factuality score. Publish externally meaningful results only after independent
claim annotation, inter-rater agreement, held-out evaluation, adversarial paraphrases, and live-model
repeated runs.
