# Contributing

1. Create a Python 3.11–3.13 virtual environment and install `requirements.txt`.
2. Keep credentials in `.env`; never commit keys or user data.
3. Add a regression test or eval case for behavior changes.
4. Run pytest with coverage, the offline eval runner, Ruff, and `pip check` as shown in the README.
5. Keep graph state JSON-serializable and scope evidence to the active route.
6. Document meaningful architecture/security trade-offs in the ADR or threat model.

Integration tests that call model providers should use the `integration` marker and must not run in credential-free CI by default.
