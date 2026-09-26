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
5. Markdown citation URLs must appear in the retrieved-source allowlist.
6. Weighted claim coverage and citation precision produce an explainable policy confidence score.
7. The policy releases the draft, retains only supported claims, or fails closed with an abstention.

Each report includes claim/evidence fingerprints, component scores, unsupported high-risk counts,
the policy decision, and an optional HMAC-bound receipt. Ordinary `general` conversation is marked
`not_required` so the control does not force retrieval for every chat message.

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

The 12 checked-in cases exercise supported releases, unsupported facts, fabricated citations,
partial repair, unsupported numerical claims, web/RAG/graph/math evidence, missing evidence,
general-chat bypass, honest retrieval failure, and artifact tampering. CI publishes the report, and
the AgentOps command center displays it under **Grounding gate**.

This suite validates policy mechanics against synthetic cases. It does not establish a universal
hallucination rate or factuality score. Publish externally meaningful results only after independent
claim annotation, inter-rater agreement, held-out evaluation, adversarial paraphrases, and live-model
repeated runs.
