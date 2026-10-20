# Dependency audit exceptions

The October 19, 2026 unwaived scan also reports advisories affecting
`cryptography` 46.0.7. Those findings are not covered by the ChromaDB exceptions
below. The current dependency bound blocks the reported patched releases;
the [validation record](../validation-2026-10-19.md) keeps this as a merge gate.
No exception was added or renewed during this check.

Reviewed: 2026-09-06. Re-review by: 2026-10-06.

Status observed on 2026-10-16: **re-review overdue**. The release-readiness scan
did not renew this risk acceptance or add exceptions. Do not treat a CI pass
with the existing ignores as an unqualified security clearance.

The CI audit temporarily ignores `PYSEC-2026-311`, `CVE-2026-45830`, `CVE-2026-45831`, and `CVE-2026-45833` for ChromaDB 1.5.9 because no fixed PyPI release exists at review time.

The reported attack paths are Chroma's network FastAPI collection-management and multi-tenant authorization endpoints. This project uses Chroma only as an embedded `PersistentClient` through `langchain-chroma`; it does not start, mount, or expose the Chroma HTTP server. Only authenticated application ingestion scripts create/update collections, and the Kubernetes network policy exposes only the application API.

This is a time-bounded mitigation, not a claim that the dependency is safe in every mode. Remove the exceptions as soon as a fixed version is available. Do not deploy Chroma's Python HTTP server from this repository. A production multi-tenant version should prefer a patched managed vector store or an isolated Qdrant/Postgres-pgvector service with its own authentication and network policy.

The [original research advisory](https://www.hiddenlayer.com/sai-security-advisory/2026-06-chromadb-5)
describes the collection-update model-loading path for CVE-2026-45833. The
[reviewed advisory](https://github.com/advisories/GHSA-36p7-vc44-83pf), checked on
2026-10-16, still lists 0.4.17 through 1.5.9 as affected and no patched version.
Embedded-only use is a deployment-scope mitigation, not proof that every use of
the dependency is safe. An owner must re-evaluate the complete waiver set and
deployment constraints before a production release; this scan checked two
advisories directly, not every dependency or transitive vulnerability.
