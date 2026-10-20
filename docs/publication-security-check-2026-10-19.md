# Publication security check

Checked on October 19, 2026, before preparing the GitHub update.

The scan covered the tracked and non-ignored working-tree files, built frontend
assets, and all 1,467 objects reachable through the local Git references. It
compared these with the four secret values configured locally and checked for
common provider tokens, AWS access-key shapes and private-key blocks. No
credential matches remained after reviewing the exact synthetic test markers.

The security evaluation intentionally includes credential-shaped test inputs in
`evals/datasets/agent_security_redteam.jsonl` and `tests/test_agent_security.py`.
The private-key fixture is not a loadable key. These exact authored markers were
reviewed individually; the surrounding files were not excluded from scanning.

## Files kept out of Git

The ignore checks confirmed that environment files, Streamlit secrets, private
document storage, local evaluation reports, submitted repositories, vector
indexes, runtime logs, frontend dependencies and built assets are excluded.
The index contains neither the local `.env` file nor private runtime data.
No environment files, private-key files or local database files appeared in the
reachable history path check.

The public `.env.example` and Kubernetes secret example contain empty values or
explicit replacement placeholders, not the configured credentials. The staging
area was unchanged by this check, and no commit or push was performed.

## Scope

This is a local publication check, not a security certification or a scan of
unfetched remote references. Secret detection combines exact local-value
matching with recognizable credential patterns; it cannot identify every
possible encoded, unknown or previously revoked secret. The final staged diff
must be checked again before committing, and ignored paths must not be
force-added.

Detailed scan results stay in the ignored local evaluation directory. They
contain counts, file paths and checks, not credential values or document text.
