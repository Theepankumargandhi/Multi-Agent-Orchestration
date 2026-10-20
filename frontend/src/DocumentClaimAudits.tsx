import { useEffect, useRef, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';
import { DocumentClaimReplay } from './DocumentClaimReplay';
import { DocumentClaimStress } from './DocumentClaimStress';
import { DocumentClaimBundle } from './DocumentClaimBundle';
import { DocumentSourceReplay } from './DocumentSourceReplay';
import { DocumentCellConcordance } from './DocumentCellConcordance';

type Claim = { claim_id: string; label: string; expected_value: string; expected_unit: string };
type Draft = { result_id: string; claim: Claim };
type Saved = {
  id: string;
  payload: {
    receipt: {
      status: string;
      operation?: string;
      query?: { row_label: string; column_label: string };
    };
  };
};
type Observation = {
  source_id: string;
  status: string;
  citations: { page: number; region_id: string; quote: string }[];
};
type Audit = {
  id: string;
  name: string;
  total: number;
  processed: number;
  complete: boolean;
  all_claims_supported: boolean;
  checkpoint_fingerprint: string;
  counts: { supported: number; refuted: number; unresolved: number };
  items: (Draft & {
    evidence_status: string;
    observations: Observation[];
    check: {
      status: string;
      reason: string;
      observed_value: string | null;
      observed_unit: string | null;
    } | null;
  })[];
};

export function DocumentClaimAudits({
  api,
  setId,
  fingerprint,
  results,
  sources,
  onInspect,
}: {
  api: ApiClient;
  setId: string;
  fingerprint: string;
  results: Saved[];
  sources: { id: string; name: string }[];
  onInspect: (documentId: string, page: number, regionId: string) => void;
}) {
  const base = `/workbench/documents/reconciliations/${setId}/audits`;
  const counter = useRef(1);
  const empty = (id: number): Draft => ({
    result_id: '',
    claim: { claim_id: `claim-${id}`, label: '', expected_value: '', expected_unit: 'scalar' },
  });
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [drafts, setDrafts] = useState<Draft[]>([empty(1)]);
  const [audits, setAudits] = useState<{ id: string; name: string }[]>([]);
  const [selected, setSelected] = useState('');
  const [report, setReport] = useState<Audit | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [exportConsent, setExportConsent] = useState(false);
  const [deleting, setDeleting] = useState(false);
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    void api
      .request<{ audits: { id: string; name: string }[] }>(base, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setAudits(value.audits);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, open, refresh]);
  useEffect(() => {
    setExportConsent(false);
    setDeleting(false);
  }, [selected]);
  useEffect(() => {
    setReport(null);
    if (!open || !selected) return;
    const controller = new AbortController();
    void api
      .request<Audit>(`${base}/${selected}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setReport(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, open, selected, refresh]);
  async function act(work: () => Promise<unknown>, reload = true) {
    setBusy(true);
    setError('');
    try {
      await work();
      if (reload) setRefresh((value) => value + 1);
    } catch (e) {
      if (!isAbort(e))
        setError(`${errorText(e)} Refresh the audit before retrying an uncertain action.`);
    } finally {
      setBusy(false);
    }
  }
  function edit(index: number, field: keyof Claim | 'result_id', value: string) {
    setDrafts((previous) =>
      previous.map((item, i) =>
        i !== index
          ? item
          : field === 'result_id'
            ? { ...item, result_id: value }
            : { ...item, claim: { ...item.claim, [field]: value } },
      ),
    );
  }
  const valid =
    name.trim().length >= 3 &&
    drafts.every(
      (item) =>
        results.some((r) => r.id === item.result_id) &&
        item.claim.label.trim() &&
        /^[+-]?(?:0|[1-9][0-9]{0,59})(?:\.[0-9]{1,60})?$/.test(item.claim.expected_value) &&
        item.claim.expected_value.length <= 100,
    );
  async function download() {
    await act(async () => {
      const data = await api.request<unknown>(`${base}/${selected}/export`);
      const text = JSON.stringify(data, null, 2);
      const blob = new Blob([text], { type: 'application/json;charset=utf-8' });
      if (blob.size > 6_000_000) throw Error('The formatted report exceeds the download budget.');
      const url = URL.createObjectURL(blob);
      try {
        const link = document.createElement('a');
        link.href = url;
        link.download = 'agentforge-numeric-claim-audit.json';
        document.body.append(link);
        link.click();
        link.remove();
      } finally {
        URL.revokeObjectURL(url);
      }
    }, false);
  }
  return (
    <section className="claim-audit-panel">
      <h3>Numeric claim audits</h3>
      <p>
        Check explicit numeric assertions against saved facts or calculations. Labels are
        display-only: this does not verify prose, infer scope or approve an answer.
      </p>
      <button className="button secondary" disabled={busy} onClick={() => setOpen((v) => !v)}>
        {open ? 'Close claim audits' : 'Open claim audits'}
      </button>
      {open && (
        <>
          {error && <Notice error>{error}</Notice>}
          <details open={!selected}>
            <summary>Create a frozen claim audit</summary>
            <label>
              Audit name
              <input
                value={name}
                minLength={3}
                maxLength={80}
                disabled={busy}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            {drafts.map((item, index) => (
              <fieldset className="claim-draft" key={item.claim.claim_id}>
                <legend>Claim {index + 1}</legend>
                <label>
                  Saved evidence for claim {index + 1}
                  <select
                    value={item.result_id}
                    disabled={busy}
                    onChange={(e) => edit(index, 'result_id', e.target.value)}
                  >
                    <option value="">Choose an exact saved result</option>
                    {results.map((result) => (
                      <option value={result.id} key={result.id}>
                        {result.payload.receipt.query
                          ? `${result.payload.receipt.query.row_label} / ${result.payload.receipt.query.column_label}`
                          : result.payload.receipt.operation}{' '}
                        · {result.payload.receipt.status} · {result.id.slice(0, 8)}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Display label for claim {index + 1}
                  <input
                    value={item.claim.label}
                    maxLength={160}
                    disabled={busy}
                    onChange={(e) => edit(index, 'label', e.target.value)}
                  />
                </label>
                <label>
                  Expected value for claim {index + 1}
                  <input
                    value={item.claim.expected_value}
                    maxLength={100}
                    inputMode="decimal"
                    disabled={busy}
                    onChange={(e) => edit(index, 'expected_value', e.target.value)}
                  />
                </label>
                <label>
                  Expected unit for claim {index + 1}
                  <select
                    value={item.claim.expected_unit}
                    disabled={busy}
                    onChange={(e) => edit(index, 'expected_unit', e.target.value)}
                  >
                    {['scalar', '%', 'USD', 'EUR', 'GBP'].map((unit) => (
                      <option value={unit} key={unit}>
                        {unit}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="text-button"
                  disabled={busy || drafts.length === 1}
                  onClick={() => setDrafts((value) => value.filter((_, i) => i !== index))}
                >
                  Remove claim {index + 1}
                </button>
              </fieldset>
            ))}
            <p className="small-text muted">
              Exact decimal strings only: no commas, exponent notation, rounding tolerance or
              automatic scale conversion. A percent value of 25 means 25%, not 0.25.
            </p>
            <div className="button-row">
              <button
                className="button secondary"
                disabled={busy || drafts.length >= 10}
                onClick={() => {
                  const next = empty(++counter.current);
                  setDrafts((value) => [...value, next]);
                }}
              >
                Add claim ({drafts.length}/10)
              </button>
              <button
                className="button"
                disabled={busy || !valid}
                onClick={() =>
                  void act(async () => {
                    const value = await api.request<{ id: string }>(base, {
                      method: 'POST',
                      body: {
                        name: name.trim(),
                        collection_fingerprint: fingerprint,
                        claims: drafts.map((item) => ({
                          ...item,
                          claim: { ...item.claim, label: item.claim.label.trim() },
                        })),
                      },
                    });
                    setSelected(value.id);
                  })
                }
              >
                Create claim audit
              </button>
            </div>
          </details>
          <div className="button-row">
            <label>
              Saved claim audits
              <select
                value={selected}
                disabled={busy}
                onChange={(e) => setSelected(e.target.value)}
              >
                <option value="">Choose an audit</option>
                {audits.map((audit) => (
                  <option key={audit.id} value={audit.id}>
                    {audit.name}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => setRefresh((v) => v + 1)}
            >
              Refresh claim audits
            </button>
          </div>
          {report && (
            <div className="claim-audit-report">
              <h4>{report.name}</h4>
              <Badge>{report.complete ? 'Complete' : 'Partial — unchecked claims remain'}</Badge>
              <p>
                Checked {report.processed}/{report.total} · supported {report.counts.supported} ·
                refuted {report.counts.refuted} · unresolved {report.counts.unresolved}
              </p>
              <Notice>
                {report.all_claims_supported
                  ? 'All listed numeric assertions match the selected evidence. This is not semantic entailment, source truth or release authorization.'
                  : 'Not all claims are supported. Partial, refuted or unresolved evidence must not be presented as a passing audit.'}
              </Notice>
              <button
                className="button"
                disabled={busy || report.complete}
                onClick={() =>
                  void act(() =>
                    api.request(`${base}/${selected}/advance`, {
                      method: 'POST',
                      body: { checkpoint_fingerprint: report.checkpoint_fingerprint },
                    }),
                  )
                }
              >
                Check next two claims
              </button>
              {report.items.map((item) => (
                <article className="document-result" key={item.claim.claim_id}>
                  <strong>{item.claim.label}</strong>
                  <p>
                    Expected: {item.claim.expected_value} {item.claim.expected_unit} · evidence:{' '}
                    {item.evidence_status}
                  </p>
                  <p>
                    <Badge>{item.check?.status ?? 'pending'}</Badge>{' '}
                    {item.check?.reason ?? 'Not checked yet'}
                  </p>
                  {item.check?.observed_value != null && (
                    <p>
                      Verified observed value: {item.check.observed_value}{' '}
                      {item.check.observed_unit}
                    </p>
                  )}
                  {item.observations.map((observation, index) => (
                    <details key={index}>
                      <summary>
                        {sources.find((s) => s.id === observation.source_id)?.name ??
                          observation.source_id}{' '}
                        · {observation.status}
                      </summary>
                      {observation.citations.map((citation, i) => (
                        <button
                          className="citation-button"
                          key={i}
                          onClick={() =>
                            onInspect(observation.source_id, citation.page, citation.region_id)
                          }
                        >
                          Page {citation.page}: {citation.quote}
                        </button>
                      ))}
                    </details>
                  ))}
                </article>
              ))}
              <DocumentClaimReplay
                key={selected}
                api={api}
                setId={setId}
                auditId={selected}
                sourceCheckpoint={report.checkpoint_fingerprint}
                complete={report.complete}
                onInspect={onInspect}
              />
              <DocumentClaimStress
                key={'stress-' + selected}
                api={api}
                setId={setId}
                auditId={selected}
                sourceCheckpoint={report.checkpoint_fingerprint}
                complete={report.complete}
                onInspect={onInspect}
              />
              <DocumentSourceReplay
                key={'original-' + selected}
                api={api}
                setId={setId}
                auditId={selected}
                checkpoint={report.checkpoint_fingerprint}
                complete={report.complete}
                sources={sources}
              />
              <DocumentCellConcordance
                key={'cells-' + selected}
                api={api}
                setId={setId}
                auditId={selected}
                checkpoint={report.checkpoint_fingerprint}
                complete={report.complete}
                sources={sources}
              />
              <DocumentCellConcordance
                key={'pixels-' + selected}
                api={api}
                setId={setId}
                auditId={selected}
                checkpoint={report.checkpoint_fingerprint}
                complete={report.complete}
                sources={sources}
                mode="pixels"
              />
              <DocumentClaimBundle
                key={'bundle-' + selected}
                api={api}
                setId={setId}
                auditId={selected}
                checkpoint={report.checkpoint_fingerprint}
                complete={report.complete}
              />
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={exportConsent}
                  disabled={busy}
                  onChange={(e) => setExportConsent(e.target.checked)}
                />
                I understand the JSON contains private claim text and evidence excerpts. Downloaded
                copies cannot be revoked by this application.
              </label>
              <button
                className="button secondary"
                disabled={busy || !exportConsent}
                onClick={() => void download()}
              >
                Download private audit JSON
              </button>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={deleting}
                  disabled={busy}
                  onChange={(e) => setDeleting(e.target.checked)}
                />
                Confirm deletion of this audit and dependent replay and stress reports; saved
                evidence and sources will remain.
              </label>
              <button
                className="button secondary"
                disabled={busy || !deleting}
                onClick={() =>
                  void act(async () => {
                    await api.request(`${base}/${selected}`, { method: 'DELETE' });
                    setSelected('');
                  })
                }
              >
                Delete claim audit
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
