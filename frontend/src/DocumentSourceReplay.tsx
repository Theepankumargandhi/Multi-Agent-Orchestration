import { useEffect, useRef, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';

type Report = {
  source_id: string;
  audit_checkpoint_fingerprint: string;
  status: string;
  original_bytes_match: boolean;
  profile_matches: boolean | null;
  snapshot_matches: boolean;
  extraction_reexecuted: boolean;
  numeric_checks_reproduced: number;
  numeric_counts: { supported: number; refuted: number; unresolved: number } | null;
  runtime_image_id: string | null;
  elapsed_seconds: number;
};

export function DocumentSourceReplay({
  api,
  setId,
  auditId,
  checkpoint,
  complete,
  sources,
}: {
  api: ApiClient;
  setId: string;
  auditId: string;
  checkpoint: string;
  complete: boolean;
  sources: { id: string; name: string }[];
}) {
  const [confirm, setConfirm] = useState(false);
  const [ocr, setOCR] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [reports, setReports] = useState<Record<string, Report>>({});
  const lifetime = useRef(new AbortController());
  useEffect(() => {
    const controller = new AbortController();
    lifetime.current = controller;
    setReports({});
    setConfirm(false);
    setOCR(false);
    setBusy('');
    setError('');
    return () => controller.abort();
  }, [checkpoint, setId, auditId, complete]);
  async function replay(sourceId: string) {
    if (!confirm || !complete || busy) return;
    const signal = lifetime.current.signal;
    setBusy(sourceId);
    setError('');
    setReports((previous) => {
      const next = { ...previous };
      delete next[sourceId];
      return next;
    });
    try {
      const value = await api.replayOriginal<Report>(
        setId,
        auditId,
        sourceId,
        checkpoint,
        ocr,
        signal,
      );
      if (signal.aborted) return;
      if (value.source_id !== sourceId || value.audit_checkpoint_fingerprint !== checkpoint)
        throw Error('The replay response belongs to a different source or checkpoint.');
      setReports((previous) => ({ ...previous, [sourceId]: value }));
    } catch (e) {
      if (!signal.aborted && !isAbort(e)) setError(`${errorText(e)} No automatic retry was made.`);
    } finally {
      if (!signal.aborted) setBusy('');
    }
  }
  return (
    <details className="source-replay-panel">
      <summary>Replay against original source bytes</summary>
      <p>
        Check the stored file hash and re-extract one original in the network-disabled parser.
        Compare its fresh extraction with the audit's frozen source. This does not rewrite
        documents, previews, saved results or owner reviews.
      </p>
      <Notice>
        Same parser, not an independent extractor. A matching replay does not prove publisher
        identity, extraction accuracy or source truth. Refuted claims remain refuted. Each click
        re-extracts only that source; other sources retain their frozen snapshots.
      </Notice>
      {!complete ? (
        <p className="muted">Complete the audit before replaying original sources.</p>
      ) : (
        <>
          {error && <Notice error>{error}</Notice>}
          <label className="check-label">
            <input
              type="checkbox"
              disabled={!!busy}
              checked={confirm}
              onChange={(e) => setConfirm(e.target.checked)}
            />
            I confirm local re-extraction. Leaving this page stops viewing, not guaranteed server
            cancellation.
          </label>
          <label className="check-label">
            <input
              type="checkbox"
              disabled={!!busy}
              checked={ocr}
              onChange={(e) => setOCR(e.target.checked)}
            />
            Allow local OCR only when it was enabled on the original upload.
          </label>
          <p>
            Sources checked this visit: {Object.keys(reports).length}/{sources.length}. Pending or
            failed sources are not matches. Reports are not saved and clear on navigation.
          </p>
          {sources.map((source) => {
            const report = reports[source.id];
            return (
              <article className="document-result" key={source.id}>
                <strong>{source.name}</strong>
                <button
                  className="button secondary"
                  disabled={!!busy || !confirm}
                  onClick={() => void replay(source.id)}
                >
                  {busy === source.id
                    ? `Re-extracting ${source.name}…`
                    : `Re-extract ${source.name}`}
                </button>
                {report ? (
                  <>
                    <Badge>{report.status}</Badge>
                    <p>
                      Original bytes: {report.original_bytes_match ? 'match' : 'not matched'} ·
                      profile:{' '}
                      {report.profile_matches === null
                        ? 'unchecked'
                        : report.profile_matches
                          ? 'match'
                          : 'changed'}{' '}
                      · frozen snapshot: {report.snapshot_matches ? 'match' : 'not matched'}.
                    </p>
                    <p>
                      Re-extraction completed: {report.extraction_reexecuted ? 'yes' : 'no'} ·
                      numeric checks reproduced: {report.numeric_checks_reproduced} · elapsed:{' '}
                      {report.elapsed_seconds.toFixed(2)} s.
                    </p>
                    {report.numeric_counts && (
                      <p>
                        Original numeric verdicts reproduced: supported{' '}
                        {report.numeric_counts.supported}, refuted {report.numeric_counts.refuted},
                        unresolved {report.numeric_counts.unresolved}.
                      </p>
                    )}
                    {report.runtime_image_id && (
                      <p className="small-text">
                        Resolved runtime image: <code>{report.runtime_image_id}</code>
                      </p>
                    )}
                  </>
                ) : (
                  <p className="muted">
                    {busy === source.id ? 'Running; no result yet.' : 'Unchecked this visit.'}
                  </p>
                )}
              </article>
            );
          })}
        </>
      )}
    </details>
  );
}
