# Focused integration examples

The [document SDK guide](../docs/document-sdk.md) is the starting point for using
text evidence in another application. Install from the repository root with
`python -m pip install ./sdk`.

- `document_sdk_quickstart.py`: offline authored technical-report question,
  literal answer and checked page/version citations.
- `document_sdk_fastapi.py`: small localhost `POST /answer` example using an
  injected corpus. It is not an authenticated multi-tenant deployment.
- `multimodal_evidence.py`: explicit local PDF/PNG/JPEG evidence with optional
  caller-installed Tesseract OCR and spatial receipts. See the
  [multimodal guide](../docs/multimodal-evidence-sdk.md) for native table arithmetic.
- `patch-verification.preds.json`: explicitly authored mini-SWE-agent-format
  interoperability input for the [portable patch verifier](../docs/portable-patch-verification.md).
- `maintainer-pilot.request.json`: intentionally invalid placeholder template;
  genuine consent and revision details are required before registration.

None of these examples demonstrates a trained model or real maintainer adoption.
The optional spatial SDK supports local OCR and native ruled tables, not general
vision, chart understanding or generated summaries. Never replace authored example text with private documents and publish
the resulting JSON without separate permission.
