# Evidence intelligence and conflict-aware RAG

Retrieval success does not imply evidence quality. Search results can repeat the same article, contain stale claims, disagree on numbers, carry prompt injections, or create the appearance of corroboration through copied content. AgentForge therefore has an optional pre-synthesis evidence adjudication node.

## Control flow

For web, hybrid, RAG, knowledge-graph, and calculator routes, `evidence_adjudication_agent` runs after retrieval and before response generation. It:

1. quarantines retrieval prompt-injection patterns;
2. clusters near-duplicate content across domains;
3. selects one representative per duplicate cluster using source and freshness signals;
4. enforces independent-source requirements;
5. validates dated evidence against the requested recency window;
6. builds numeric and negation conflict edges between semantically overlapping statements;
7. passes only the adjudicated evidence objects to synthesis, citation completion, grounding, conformal control, and adaptive deliberation.

The original retrieved text remains available for operational debugging but cannot re-enter the model context after a quality report exists. If adjudication errors, every source is removed and the response fails closed.

## Decisions

- `pass`: evidence satisfies the configured controls.
- `degraded`: usable evidence remains, but duplicates were collapsed or an unsafe source was quarantined.
- `abstain`: evidence is absent, insufficiently independent, too stale, or contradictory.
- `not_required`: the route does not require retrieval.

Every report contains only bounded metadata, evidence fingerprints, domains, conflict values, and control outcomes. Reports can be SHA-256 fingerprinted locally or HMAC-bound with a dedicated key.

## Configuration

```dotenv
EVIDENCE_QUALITY_ENABLED=true
EVIDENCE_MIN_WEB_SOURCES=2
EVIDENCE_DUPLICATE_SIMILARITY=0.82
EVIDENCE_AUTHORITY_DOMAINS=openai.com,github.com,python.org
EVIDENCE_QUALITY_INTEGRITY_KEY=replace-with-an-independent-secret
```

Authority domains are operator policy, not a universal truth ranking. Local RAG, knowledge-graph, and calculator evidence are treated as first-party evidence but still undergo injection, duplication, and conflict checks.

## Evaluation

```bash
python -m evals.evidence_quality_evaluation \
  --output data/evaluations/evidence-quality/latest.json \
  --min-pass-rate 1 \
  --min-benign-utility 1 \
  --max-unsafe-release-rate 0
```

The checked-in corpus covers benign web/local/math evidence, retrieval prompt injection, cross-domain copied-content laundering, numeric disagreement, negation disagreement, stale and undated results, and insufficient source diversity. The **Evidence intelligence** dashboard tab exposes containment, benign utility, quarantine, conflict, freshness, duplication, and unsafe-release metrics.

The default contradiction detector is deterministic and deliberately auditable. It catches overlapping propositions with incompatible numbers or negation polarity; it is not a general natural-language inference model. Production deployments should calibrate a domain NLI or judge ensemble on independently annotated contradiction pairs, version the source policy, and retain human escalation for material conflicts.
