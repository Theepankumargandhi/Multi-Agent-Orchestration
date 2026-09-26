# Dependency audit exceptions

Reviewed: 2026-09-06. Re-review by: 2026-10-06.

The CI audit temporarily ignores `PYSEC-2026-311`, `CVE-2026-45830`, `CVE-2026-45831`, and `CVE-2026-45833` for ChromaDB 1.5.9 because no fixed PyPI release exists at review time.

The reported attack paths are Chroma's network FastAPI collection-management and multi-tenant authorization endpoints. This project uses Chroma only as an embedded `PersistentClient` through `langchain-chroma`; it does not start, mount, or expose the Chroma HTTP server. Only authenticated application ingestion scripts create/update collections, and the Kubernetes network policy exposes only the application API.

This is a time-bounded mitigation, not a claim that the dependency is safe in every mode. Remove the exceptions as soon as a fixed version is available. Do not deploy Chroma's Python HTTP server from this repository. A production multi-tenant version should prefer a patched managed vector store or an isolated Qdrant/Postgres-pgvector service with its own authentication and network policy.
