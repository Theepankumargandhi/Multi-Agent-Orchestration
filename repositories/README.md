# Repositories for sandboxed coding tasks

This is the default source root for coding tasks, not the agent's working directory. Place only repositories you are authorized to inspect here, for example `repositories/sample/`. Submit `"repository":"sample"`, not an absolute path. The worker resolves the name below `CODE_AGENT_REPOSITORY_ROOT` and rejects traversal outside that root.

Everything beneath this directory except this README is Git-ignored. Local source code must not be added to AgentForge's repository with `git add -f`. Ignoring files does not encrypt them or authorize sending their contents to a model provider.

## Local setup

Run from the AgentForge repository root, on a trusted host with Docker available:

```bash
docker build -f docker/Dockerfile.code-sandbox -t agentforge-code-sandbox:local .
```

Configure the private environment and restart the API:

```dotenv
CODE_AGENT_ENABLED=true
CODE_AGENT_REPOSITORY_ROOT=repositories
CODE_AGENT_IMAGE=agentforge-code-sandbox:local
CODE_AGENT_NETWORK_ENABLED=false
CODE_AGENT_EXECUTION_MODE=embedded
CODE_AGENT_WORKFLOW=verified_pr
```

Submit an authenticated task to `POST /code/tasks` with an `Idempotency-Key`, an issue, and a trusted fixed command such as `["python","-m","pytest","-q"]`. Follow the task events and evidence dossier. Only an `awaiting_approval` result can release a patch after an explicit owner decision; approval never applies the diff to the original repository. Exact payloads and endpoints are in [the coding-agent guide](../docs/code-agent.md).

The included image contains Python, pytest, and Ruff. Tree-sitter indexes Python, TypeScript, JavaScript, Java, Go, and Rust, but parser support does not provide those compilers or permission to edit every extension. Other stacks need a trusted prebuilt image, supported test-command policy, and appropriate edit allowlists. Dependencies are not installed over the network during a task.

## What reaches the workspace and model

Each job copies its selected repository into a unique, size-bounded temporary workspace. The original repository is not edited. Current copy exclusions in [workspace.py](../code_agent/workspace.py) are:

- directory components `.git`, `.venv`, `venv`, `env`, `__pycache__`, `.pytest_cache`, `.ruff_cache`, `node_modules`, `chroma_db`, `graph_chroma_db`, `data`, and `.runtime_logs`;
- symlinks, files named `.env` or beginning `.env.`, and `.pem`, `.key`, `.p12`, `.pfx`, or `.crt` suffixes.

These are filename/path rules, not a complete secret detector. A credential in an ordinary source file, an SSH key without one of those suffixes, or a database outside excluded directories is not automatically removed by copying. Sanitize repositories before submitting them. Binary/oversized content is restricted by agent file reads, but that does not mean all such files are excluded from the workspace or hidden from repository tests.

Indexing parses code as data; it does not import or execute it. The host-side model can receive selected source snippets, issue text, diffs, and bounded tool output. Provider credentials are not injected into the test container, but sending repository context to the configured provider is still a data-disclosure decision. Stored patches and dossiers can contain private source and need their own access and retention controls.

See [code intelligence](../docs/code-intelligence.md) for incremental parsing, hashing/feature defaults, optional learned retrieval, query decomposition, deduplication, and recall/token-budget ablations.

## Isolation and deployed workers

Tests execute in a non-root Docker container with networking disabled by default, a read-only root filesystem, dropped capabilities, and resource/time limits. The temporary workspace is writable. Repository tests are untrusted code; this is not equivalent to microVM isolation.

For API/worker separation, set `CODE_AGENT_EXECUTION_MODE=external` on the API and run `python -m code_agent.worker` on the dedicated worker. It needs the same configured local job/artifact storage, authorized source root, and sandbox image. A relative root is resolved from the worker's current directory; use a deliberate absolute path when processes start elsewhere. Shared SQLite is a single-host reference design, not a multi-node queue.

The stock Compose and Kubernetes stacks leave coding execution disabled and do not mount repositories or the Docker socket. Do not expose the host Docker daemon through a public API container. Worker provisioning, repository mounts, and stronger sandbox isolation are separate operator responsibilities.
