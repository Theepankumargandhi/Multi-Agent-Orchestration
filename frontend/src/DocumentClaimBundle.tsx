import { useEffect, useRef, useState } from 'react';
import type { ApiClient } from './api';
import { Notice, errorText, isAbort } from './components';

export function DocumentClaimBundle({
  api,
  setId,
  auditId,
  checkpoint,
  complete,
}: {
  api: ApiClient;
  setId: string;
  auditId: string;
  checkpoint: string;
  complete: boolean;
}) {
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [fingerprint, setFingerprint] = useState('');
  const lifetime = useRef(new AbortController());
  useEffect(() => {
    lifetime.current = new AbortController();
    return () => lifetime.current.abort();
  }, []);
  useEffect(() => {
    setConsent(false);
    setFingerprint('');
  }, [checkpoint]);
  async function download() {
    if (!consent || !complete || busy) return;
    const signal = lifetime.current.signal;
    setBusy(true);
    setError('');
    try {
      const raw = await api.requestJSONText(
        `/workbench/documents/reconciliations/${setId}/audits/${auditId}/replay-bundle`,
        {
          method: 'POST',
          signal,
          body: { checkpoint_fingerprint: checkpoint, include_extraction_text: true },
        },
      );
      if (signal.aborted) return;
      let data: { protocol?: string; fingerprint?: string } | null;
      try {
        data = JSON.parse(raw);
      } catch {
        throw Error('The service did not return a replay bundle.');
      }
      if (
        data?.protocol !== 'agentforge-claim-replay-bundle-v1' ||
        typeof data.fingerprint !== 'string' ||
        !/^[a-f0-9]{64}$/.test(data.fingerprint)
      )
        throw Error('The service did not return a replay bundle.');
      const blob = new Blob([raw], { type: 'application/json;charset=utf-8' });
      if (blob.size > 3_000_000) throw Error('The replay bundle exceeds its 3 MB budget.');
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      try {
        link.href = url;
        link.download = 'agentforge-claim-replay-bundle.json';
        document.body.append(link);
        link.click();
      } finally {
        link.remove();
        URL.revokeObjectURL(url);
      }
      setFingerprint(data.fingerprint);
    } catch (e) {
      if (!signal.aborted && !isAbort(e)) setError(errorText(e));
    } finally {
      if (!signal.aborted) setBusy(false);
    }
  }
  return (
    <details className="claim-bundle-panel">
      <summary>Portable offline replay bundle</summary>
      <p>
        Recompute this audit with the SDK without the service, Docker, OCR or a provider. The bundle
        includes every assertion and full extraction text from the frozen sources, including uncited
        regions. It contains no original files, images or owner reviews.
      </p>
      <Notice>
        Replay verifies numeric consistency with an untrusted snapshot, not source authenticity,
        extraction accuracy, current review authority or answer approval. Refuted and unresolved
        assertions remain in the bundle.
      </Notice>
      {!complete ? (
        <p className="muted">Complete the audit before exporting a replay bundle.</p>
      ) : (
        <>
          {error && <Notice error>{error}</Notice>}
          <label className="check-label">
            <input
              type="checkbox"
              checked={consent}
              disabled={busy}
              onChange={(e) => setConsent(e.target.checked)}
            />
            I authorize downloading the full private extraction snapshot, including uncited text.
            Downloaded copies cannot be revoked.
          </label>
          <button
            className="button secondary"
            disabled={busy || !consent}
            onClick={() => void download()}
          >
            {busy ? 'Preparing replay bundle…' : 'Download private replay bundle'}
          </button>
          {fingerprint && (
            <>
              <p className="small-text">
                Bundle fingerprint: <code>{fingerprint}</code>. Retain it separately if you trust
                this snapshot; this is not a publisher signature.
              </p>
              <pre className="code-block">{`python -m agentforge_sdk.bundle_cli agentforge-claim-replay-bundle.json --expected-fingerprint ${fingerprint}`}</pre>
              <p className="small-text">
                Exit 0 means replay verified, not that every assertion is supported. Add
                --require-supported to fail when any numeric assertion is refuted or unresolved.
              </p>
            </>
          )}
        </>
      )}
    </details>
  );
}
