import { useEffect, useRef, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';

type Report = {
  source_id: string;
  audit_checkpoint_fingerprint: string;
  status: string;
  original_bytes_match: boolean;
  probe_executed: boolean;
  engine: string | null;
  requested_cells: number;
  counts: { agree: number; different: number; uncheckable: number };
  cells: {
    region_id: string;
    page: number;
    status: string;
    probe_status: string;
    min_confidence?: number | null;
  }[];
  raster_pages?: number;
  ocr_calls?: number;
};

export function DocumentCellConcordance({
  api,
  setId,
  auditId,
  checkpoint,
  complete,
  sources,
  mode = 'text',
}: {
  api: ApiClient;
  setId: string;
  auditId: string;
  checkpoint: string;
  complete: boolean;
  sources: { id: string; name: string }[];
  mode?: 'text' | 'pixels';
}) {
  const pixels = mode === 'pixels';
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [reports, setReports] = useState<Record<string, Report>>({});
  const lifetime = useRef(new AbortController());
  useEffect(() => {
    const controller = new AbortController();
    lifetime.current = controller;
    setConfirm(false);
    setBusy('');
    setError('');
    setReports({});
    return () => controller.abort();
  }, [checkpoint, setId, auditId, complete, mode]);

  async function crosscheck(sourceId: string) {
    if (!complete || !confirm || busy) return;
    const signal = lifetime.current.signal;
    setBusy(sourceId);
    setError('');
    setReports((previous) => {
      const next = { ...previous };
      delete next[sourceId];
      return next;
    });
    try {
      const value = pixels
        ? await api.crosscheckPixels<Report>(setId, auditId, sourceId, checkpoint, signal)
        : await api.crosscheckCells<Report>(setId, auditId, sourceId, checkpoint, signal);
      if (signal.aborted) return;
      if (value.source_id !== sourceId || value.audit_checkpoint_fingerprint !== checkpoint)
        throw Error('The cell check belongs to a different source or checkpoint.');
      setReports((previous) => ({ ...previous, [sourceId]: value }));
    } catch (e) {
      if (!signal.aborted && !isAbort(e)) setError(`${errorText(e)} No automatic retry was made.`);
    } finally {
      if (!signal.aborted) setBusy('');
    }
  }

  return (
    <details className="claim-audit-report">
      <summary>
        {pixels
          ? 'Cross-check rendered cells with local OCR'
          : 'Cross-check cited cells with PDFium'}
      </summary>
      <p>
        {pixels
          ? 'Render the cited cell rectangles and read their pixels with local English OCR, without reading the PDF text layer or receiving expected text.'
          : 'Read the cited row headers, column headers and values with a different PDF text engine. PDFium receives the original bytes and coordinates, not the expected cell text.'}
      </p>
      <Notice>
        Text agreement is not claim support, publisher verification or approval. The primary parser
        still selects the coordinates.{' '}
        {pixels
          ? 'Up to 12 native cited cells on two pages. Low-confidence OCR is uncheckable, not agreement. OCR confidence is not a probability of correctness.'
          : 'Native PDF table cells only; no OCR or model requests.'}{' '}
        Other sources and uncited cells are not checked. No provider calls are made.
      </Notice>
      {!complete ? (
        <p className="muted">Complete the audit before cross-checking its cited cells.</p>
      ) : (
        <>
          {error && <Notice error>{error}</Notice>}
          <label className="check-label">
            <input
              type="checkbox"
              checked={confirm}
              disabled={!!busy}
              onChange={(e) => setConfirm(e.target.checked)}
            />
            {pixels
              ? 'I consent to local OCR of rendered cells, separately from the original upload setting.'
              : 'I confirm the local PDF text cross-check.'}
          </label>
          <p>
            Sources checked this visit: {Object.keys(reports).length}/{sources.length}. Reports are
            not saved. Leaving stops viewing, not guaranteed server cancellation.
          </p>
          {sources.map((source) => {
            const report = reports[source.id];
            return (
              <article className="document-result" key={source.id}>
                <strong>{source.name}</strong>
                <button
                  className="button secondary"
                  disabled={!confirm || !!busy}
                  onClick={() => void crosscheck(source.id)}
                >
                  {busy === source.id
                    ? `Checking ${source.name}…`
                    : pixels
                      ? `Check pixels in ${source.name}`
                      : `Cross-check ${source.name}`}
                </button>
                {report ? (
                  <>
                    <Badge>{report.status}</Badge>
                    <p>
                      Original bytes: {report.original_bytes_match ? 'match' : 'not matched'} ·
                      probe completed: {report.probe_executed ? 'yes' : 'no'} · requested cells:{' '}
                      {report.requested_cells}.
                    </p>
                    <p>
                      Agree {report.counts.agree} · different {report.counts.different} ·
                      uncheckable {report.counts.uncheckable}.
                    </p>
                    {report.engine && (
                      <p>
                        {pixels ? 'OCR engine' : 'Text engine'}: <code>{report.engine}</code>
                      </p>
                    )}
                    {pixels && (
                      <p>
                        Rendered pages: {report.raster_pages ?? 0} · local OCR calls:{' '}
                        {report.ocr_calls ?? 0}.
                      </p>
                    )}
                    <ul>
                      {report.cells.map((cell) => (
                        <li key={cell.region_id}>
                          Page {cell.page} · <code>{cell.region_id.slice(0, 12)}</code> ·{' '}
                          {cell.status} ({cell.probe_status})
                          {pixels && cell.min_confidence != null && (
                            <> · minimum word confidence {cell.min_confidence.toFixed(2)}</>
                          )}
                        </li>
                      ))}
                    </ul>
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
