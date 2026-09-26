# Sandboxed coding repositories

Place or clone repositories to evaluate beneath this directory, then set `CODE_AGENT_ENABLED=true`. The API accepts only relative repository names rooted here.

This directory is copied into a filtered temporary workspace for every task. The agent never edits the original repository. `.env`, keys, certificates, Git metadata, virtual environments, caches, local databases, and runtime data are excluded from the copy.

Do not expose the Docker daemon socket through the public API container. For a deployed system, run this subsystem as a dedicated worker on an isolated host or Kubernetes runtime-class/node pool.
