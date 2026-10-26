# Contributing

1. Use a fresh Python 3.11–3.13 virtual environment: install `requirements.txt`, then run
   `python -m pip install --no-deps --editable .`.
2. Keep credentials in `.env`; never commit keys or user data.
3. Add a regression test or eval case for every behavior change.
4. Before opening a pull request, run `python -m pytest -q`, `python -m ruff check .` and the
   offline gates listed in the README.
5. Keep graph state JSON-serializable and scope evidence to the active route.
6. Record meaningful architecture or security trade-offs in `docs/architecture/decisions.md` or
   `docs/security/threat-model.md`.

Tests that call a model provider use the `integration` marker and do not run in CI.

For UI changes, use Node 24 LTS and run `npm ci --ignore-scripts`, `npm run format:check`,
`npm run build` and `npm test` from `frontend/`. UI tests use recorded responses; never put real
keys in fixtures.
