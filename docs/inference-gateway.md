# Production LLM inference gateway

AgentForge includes an opt-in control plane in front of LangChain model calls. It turns provider selection into an auditable policy decision and protects the agent from provider outages, unbounded spend, unsafe experimentation, and cross-tenant cache reuse.

## Runtime controls

- Provider-neutral async adapter protocol with OpenAI and Groq LangChain adapters in the research graph.
- Per-provider deadlines, consecutive-failure circuit breakers, cooldowns, and half-open recovery probes.
- Ordered provider fallback. Admission control can skip an expensive provider and select an affordable fallback without invoking the rejected provider.
- Per-tenant daily cost budgets based on configured input/output prices and worst-case completion admission estimates.
- Tenant-, system-prompt-, policy-version-, and rollout-cohort-scoped similarity caching. The dependency-free default uses a deterministic hashed unigram/bigram vector; raw prompts are not retained in cache keys or receipts. Caching is opt-in and disabled for high-risk and streamed final-response calls, and a canary response cannot enter the production cohort's cache.
- Stable canary assignment using request fingerprints rather than mutable process randomness.
- Shadow inference with output-similarity, latency, cost, and response fingerprints. Shadow responses never replace the production response.
- Automatic exclusion of high-risk traffic from caches, canaries, and shadows.
- Prompt-size admission limits before provider invocation.
- Integrity-bound receipts containing provider, model, attempts, usage, cost, rollout state, circuit/budget decisions, and keyed content fingerprints, but no prompt or response text. The live runtime refuses to enable without a dedicated HMAC key.

The existing agent behavior is unchanged while `MODEL_GATEWAY_ENABLED=false`. Enabling the flag routes internal and final LangGraph model calls through the gateway. Streamed final calls disable caching and shadow traffic so cached or shadow tokens cannot alter the SSE channel; an assigned canary remains eligible because its output is the response being served.

## Configure it

Start with one provider and real current prices:

```env
MODEL_GATEWAY_ENABLED=true
MODEL_GATEWAY_POLICY_VERSION=inference-policy-v1
MODEL_GATEWAY_FINGERPRINT_KEY=replace-with-at-least-16-random-characters
MODEL_GATEWAY_DAILY_BUDGET_USD=5
MODEL_GATEWAY_MAX_PROMPT_TOKENS=16000
MODEL_GATEWAY_TIMEOUT_MS=30000
MODEL_GATEWAY_FAILURE_THRESHOLD=3
MODEL_GATEWAY_COOLDOWN_SECONDS=30
MODEL_GATEWAY_OPENAI_INPUT_COST_PER_MILLION=0
MODEL_GATEWAY_OPENAI_OUTPUT_COST_PER_MILLION=0
```

When both OpenAI and Groq are configured, either can be used as the preferred, fallback, canary, or shadow provider. Rollout names are `openai` and `groq`:

```env
MODEL_GATEWAY_CANARY_PROVIDER=groq
MODEL_GATEWAY_CANARY_PERCENTAGE=5
MODEL_GATEWAY_SHADOW_ENABLED=true
MODEL_GATEWAY_SHADOW_PROVIDER=groq
```

Do not canary and shadow the same request in a paid environment without accounting for both calls. Shadow spend is charged to the tenant ledger. High-risk requests always use the requested production provider path.

Similarity caching is deliberately off by default:

```env
MODEL_GATEWAY_SEMANTIC_CACHE_ENABLED=true
MODEL_GATEWAY_SEMANTIC_CACHE_THRESHOLD=0.92
MODEL_GATEWAY_SEMANTIC_CACHE_TTL_SECONDS=900
```

The built-in cache is bounded by TTL and process lifetime. For production, replace it with an encrypted shared vector cache, add deletion controls, and namespace it by authorization policy, data residency, model/prompt version, and tenant. Never cache regulated or user-specific responses solely because prompts are similar.

## Evaluate the control plane

The checked-in 12-case suite covers cache hits, tenant isolation, high-risk bypass, provider fallback, open circuits, half-open recovery, budget-aware fallback, budget exhaustion, shadow non-interference, stable canary cohorts, prompt limits, and privacy-safe receipts:

```bash
python -m evals.model_gateway_evaluation \
  --output data/evaluations/model-gateway/latest.json \
  --min-pass-rate 1 \
  --max-privacy-violations 0 \
  --max-canary-deviation 2.5
```

The credential-free gate currently passes all 12 controls, verifies every receipt, records zero content-leakage violations, recovers every declared provider/circuit failure, and assigns 19.9% of 2,000 stable request IDs to a 20% target cohort. These are deterministic control-plane results, not provider uptime, model quality, real savings, or load-test evidence.

The **Inference gateway** tab in the AgentOps command center displays the report and per-control integrity evidence. CI uploads the same report as `inference-gateway-control-plane`.

## Production follow-up

For a deployed study, use an external atomic budget ledger, distributed circuit state, current provider prices, encrypted shared cache storage, per-route quality gates, and OpenTelemetry metrics for cache hits, circuit state, fallbacks, budget rejections, provider latency, shadow agreement, and real cost. Run canaries only after the behavioral arena approves the candidate, and rollback when live safety, quality, or latency SLOs regress.
