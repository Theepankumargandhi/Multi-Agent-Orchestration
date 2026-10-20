# Contributing

1. Create a fresh Python 3.11–3.13 virtual environment, install `requirements.txt`, then run `python -m pip install --no-deps --editable .`. Do not reuse a virtual environment copied from another machine.
2. Keep credentials in `.env`; never commit keys or user data.
3. Add a regression test or eval case for behavior changes.
4. Run pytest with coverage, the offline eval runner, Ruff, and `pip check` as shown in the README.
5. Keep graph state JSON-serializable and scope evidence to the active route.
6. Document meaningful architecture/security trade-offs in the ADR or threat model.

For native code parsing, run `python scripts/setup_code_parsers.py` during setup
(network access may be needed). Runtime indexing uses already-cached grammars
only, falling back to Python AST/bounded patterns when unavailable. The explicit
native-parser integration test requires this setup; the fallback tests do not.

Integration tests that call model providers should use the `integration` marker and must not run in credential-free CI by default.

For Workbench changes, use Node 24 LTS and run `npm ci --ignore-scripts`,
`npm run format:check`, `npm run build` and `npm test` from `frontend/`.
`npm run format` applies the shared formatting rules. Add UI/transport controls
with authored responses, and run `tests/test_workbench.py` for backend changes.
Never add real keys or private reports to browser fixtures. A passing DOM test is
not a substitute for checking desktop/mobile layout and keyboard navigation.
