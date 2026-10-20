import { useEffect, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';
import { DocumentClaimAudits } from './DocumentClaimAudits';

type Source = {
  id: string;
  name: string;
  status: string;
  digest: string;
  manifest_fingerprint?: string | null;
};
type Citation = { page: number; region_id: string; quote: string };
type Observation = {
  source_id: string;
  status: string;
  value?: string | null;
  unit?: string | null;
  citations: Citation[];
};
type Receipt = {
  status: string;
  reason?: string;
  reasons?: string[];
  value?: string | null;
  unit?: string | null;
  query?: { row_label: string; column_label: string };
  distinct_artifacts?: number;
  operation?: string;
  observations?: Observation[];
};
type Result = {
  id: string;
  payload: { digest: string; kind: 'fact' | 'calculation'; receipt: Receipt };
  review: { owner: string; decision: string; note: string } | null;
  attention_reason: string | null;
};
type SetDetail = {
  id: string;
  name: string;
  collection_fingerprint: string;
  scope_label: string;
  sources: Source[];
  catalog: { row_label: string; column_label: string }[];
  catalog_truncated: boolean;
  results: Result[];
};

const base = '/workbench/documents/reconciliations';

export function DocumentReconciliation({
  api,
  documents,
  onInspect,
  onUpdated,
}: {
  api: ApiClient;
  documents: Source[];
  onInspect: (documentId: string, page: number, regionId: string) => void;
  onUpdated: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [scope, setScope] = useState({ entity: '', period: '', basis: '' });
  const [ids, setIds] = useState<string[]>([]);
  const [confirmed, setConfirmed] = useState(false);
  const [sets, setSets] = useState<{ id: string; name: string }[]>([]);
  const [selected, setSelected] = useState('');
  const [detail, setDetail] = useState<SetDetail | null>(null);
  const [row, setRow] = useState('');
  const [column, setColumn] = useState('');
  const [facts, setFacts] = useState<string[]>([]);
  const [operation, setOperation] = useState('percent_change');
  const [deleting, setDeleting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    void api
      .request<{ sets: { id: string; name: string }[] }>(base, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setSets(value.sets);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, open, refresh]);
  useEffect(() => {
    setFacts([]);
    setDeleting(false);
    setRow('');
    setColumn('');
  }, [selected]);
  useEffect(() => {
    setDetail(null);
    if (!selected || !open) return;
    const controller = new AbortController();
    void api
      .request<SetDetail>(`${base}/${selected}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setDetail(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, open, selected, refresh]);
  async function action(work: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await work();
      setRefresh((value) => value + 1);
    } catch (e) {
      if (!isAbort(e))
        setError(`${errorText(e)} Refresh this set before retrying an uncertain action.`);
    } finally {
      setBusy(false);
    }
  }
  async function create() {
    await action(async () => {
      const value = await api.request<{ id: string }>(base, {
        method: 'POST',
        body: {
          name: name.trim(),
          scope: {
            entity: scope.entity.trim(),
            period: scope.period.trim(),
            basis: scope.basis.trim(),
          },
          scope_confirmed: confirmed,
          sources: ids.map((id) => ({
            document_id: id,
            fingerprint: documents.find((doc) => doc.id === id)!.manifest_fingerprint,
          })),
        },
      });
      setSelected(value.id);
      setConfirmed(false);
      onUpdated();
    });
  }
  const eligible = documents.filter(
    (doc) => ['ready', 'held'].includes(doc.status) && doc.manifest_fingerprint,
  );
  const attention = detail?.results.filter((result) => result.attention_reason) ?? [];
  return (
    <section className="panel reconciliation-panel">
      <div className="section-title">
        <h2>Cross-document fact checks</h2>
        <Badge>Exact labels · no model calls</Badge>
      </div>
      <p>
        Compare numeric table facts across explicitly comparable reports. Agreement is not proof of
        truth; conflicts never select a winner.
      </p>
      <button
        className="button secondary"
        disabled={busy}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? 'Close reconciliation workspace' : 'Open reconciliation workspace'}
      </button>
      {open && (
        <>
          {error && <Notice error>{error}</Notice>}
          <details className="reconciliation-create" open={!selected}>
            <summary>Create a frozen evidence set</summary>
            <div className="change-controls">
              <label>
                Evidence set name
                <input
                  value={name}
                  minLength={3}
                  maxLength={80}
                  disabled={busy}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              {(
                [
                  ['entity', 'Entity', 60],
                  ['period', 'Reporting period', 40],
                  ['basis', 'Metric definition, units and scale', 80],
                ] as const
              ).map(([key, label, max]) => (
                <label key={key}>
                  {label}
                  <input
                    value={scope[key]}
                    maxLength={max}
                    disabled={busy}
                    onChange={(e) => {
                      setScope((value) => ({ ...value, [key]: e.target.value }));
                      setConfirmed(false);
                    }}
                  />
                </label>
              ))}
              <fieldset disabled={busy}>
                <legend>Source versions ({ids.length}/5; minimum two)</legend>
                {eligible.length === 0 && <p>Upload and extract source documents first.</p>}
                {eligible.map((doc) => (
                  <label className="check-label change-saved-item" key={doc.id}>
                    <input
                      type="checkbox"
                      checked={ids.includes(doc.id)}
                      disabled={!ids.includes(doc.id) && ids.length >= 5}
                      onChange={(e) => {
                        setIds((value) =>
                          e.target.checked
                            ? [...value, doc.id]
                            : value.filter((id) => id !== doc.id),
                        );
                        setConfirmed(false);
                      }}
                    />
                    {doc.name} · {doc.status} · {doc.digest.slice(0, 8)}
                  </label>
                ))}
              </fieldset>
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={confirmed}
                  disabled={busy}
                  onChange={(e) => setConfirmed(e.target.checked)}
                />
                I confirm these exact source versions describe comparable facts for the entity,
                period, units and scale above. The application has not inferred or authenticated
                this scope.
              </label>
              <button
                className="button"
                disabled={
                  busy ||
                  !confirmed ||
                  name.trim().length < 3 ||
                  ids.length < 2 ||
                  !Object.values(scope).every((v) => v.trim()) ||
                  ids.some((id) => !eligible.some((doc) => doc.id === id))
                }
                onClick={() => void create()}
              >
                Create evidence set
              </button>
            </div>
          </details>
          <div className="button-row">
            <label>
              Saved evidence sets
              <select
                aria-label="Saved evidence sets"
                value={selected}
                disabled={busy}
                onChange={(e) => setSelected(e.target.value)}
              >
                <option value="">Choose a set</option>
                {sets.map((set) => (
                  <option value={set.id} key={set.id}>
                    {set.name}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => setRefresh((value) => value + 1)}
            >
              Refresh evidence set
            </button>
          </div>
          {detail && (
            <div className="reconciliation-report">
              <h3>{detail.name}</h3>
              <p>Owner-confirmed scope: {detail.scope_label}</p>
              <Notice>
                Policy: at least two distinct raw file hashes and complete in-scope coverage.
                Different hashes do not establish independent sources. Missing is not zero; scalar
                units do not establish compatible metric meanings. No voting, latest-report
                preference or conversion.
              </Notice>
              <p className="small-text muted">
                Sources are frozen to these versions:{' '}
                {detail.sources.map((s) => `${s.name} (${s.digest.slice(0, 8)})`).join(', ')}. New
                uploads do not silently replace them.
              </p>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void action(() =>
                    api.request(`${base}/${selected}/facts`, {
                      method: 'POST',
                      body: {
                        collection_fingerprint: detail.collection_fingerprint,
                        row_label: row.trim(),
                        column_label: column.trim(),
                      },
                    }),
                  );
                }}
                className="change-controls"
              >
                <label>
                  Discovered table labels
                  <select
                    aria-label="Discovered table labels"
                    value=""
                    disabled={busy}
                    onChange={(e) => {
                      const q = detail.catalog[Number(e.target.value)];
                      if (q) {
                        setRow(q.row_label);
                        setColumn(q.column_label);
                      }
                    }}
                  >
                    <option value="">Choose literal labels, or enter them below</option>
                    {detail.catalog.map((q, index) => (
                      <option key={index} value={index}>
                        {q.row_label} / {q.column_label}
                      </option>
                    ))}
                  </select>
                </label>
                {detail.catalog_truncated && (
                  <p>
                    Only the first 200 label pairs are shown. Exact labels can still be entered
                    explicitly.
                  </p>
                )}
                <label>
                  Exact row label
                  <input
                    value={row}
                    maxLength={200}
                    required
                    disabled={busy}
                    onChange={(e) => setRow(e.target.value)}
                  />
                </label>
                <label>
                  Exact column label
                  <input
                    value={column}
                    maxLength={200}
                    required
                    disabled={busy}
                    onChange={(e) => setColumn(e.target.value)}
                  />
                </label>
                <button className="button" disabled={busy || !row.trim() || !column.trim()}>
                  Check numeric fact
                </button>
              </form>
              <h3>Attention queue · {attention.length}</h3>
              {attention.map((result) => (
                <p className="change-queue-item" key={result.id}>
                  {result.payload.receipt.query
                    ? `${result.payload.receipt.query.row_label} / ${result.payload.receipt.query.column_label}`
                    : result.payload.receipt.operation}{' '}
                  · {result.payload.receipt.status} · {result.attention_reason}
                </p>
              ))}
              <details>
                <summary>Calculate from saved facts ({facts.length}/8)</summary>
                <p>
                  Selection order is operand order. Unresolved facts block calculation. Percent
                  change is old then new.
                </p>
                <ol>
                  {facts.map((id) => {
                    const fact = detail.results.find((r) => r.id === id)?.payload.receipt;
                    return (
                      <li key={id}>
                        {fact?.query?.row_label} / {fact?.query?.column_label}
                      </li>
                    );
                  })}
                </ol>
                <label>
                  Fact calculation operation
                  <select
                    value={operation}
                    disabled={busy}
                    onChange={(e) => setOperation(e.target.value)}
                  >
                    <option value="percent_change">Percent change</option>
                    <option value="sum">Sum</option>
                    <option value="difference">Difference</option>
                    <option value="ratio">Ratio</option>
                  </select>
                </label>
                <div className="button-row">
                  <button
                    className="button secondary"
                    disabled={busy || !facts.length || (operation !== 'sum' && facts.length !== 2)}
                    onClick={() =>
                      void action(() =>
                        api.request(`${base}/${selected}/calculations`, {
                          method: 'POST',
                          body: {
                            collection_fingerprint: detail.collection_fingerprint,
                            operation,
                            fact_ids: facts,
                          },
                        }),
                      )
                    }
                  >
                    Calculate reconciled facts
                  </button>
                  <button className="text-button" disabled={busy} onClick={() => setFacts([])}>
                    Clear fact selection
                  </button>
                </div>
              </details>
              <h3>Saved evidence · {detail.results.length}/20</h3>
              {detail.results.map((result) => {
                const receipt = result.payload.receipt;
                return (
                  <article className="document-result" key={result.id}>
                    <strong>
                      {receipt.query
                        ? `${receipt.query.row_label} / ${receipt.query.column_label}`
                        : receipt.operation}
                    </strong>
                    <p>
                      <Badge>{receipt.status}</Badge>{' '}
                      {receipt.reasons?.join(', ') || receipt.reason}
                    </p>
                    <p>
                      Released value:{' '}
                      {receipt.value == null ? 'none' : `${receipt.value} ${receipt.unit ?? ''}`}
                    </p>
                    {receipt.distinct_artifacts != null && (
                      <p className="muted small-text">
                        {receipt.distinct_artifacts} distinct observed raw artifacts—not a
                        source-independence or confidence score.
                      </p>
                    )}
                    {result.payload.kind === 'fact' && (
                      <label className="check-label">
                        <input
                          type="checkbox"
                          aria-label={`Select fact ${receipt.query?.row_label} ${receipt.query?.column_label}`}
                          checked={facts.includes(result.id)}
                          disabled={busy || (!facts.includes(result.id) && facts.length >= 8)}
                          onChange={(e) =>
                            setFacts((value) =>
                              e.target.checked
                                ? [...value, result.id]
                                : value.filter((id) => id !== result.id),
                            )
                          }
                        />
                        Use as a calculation operand
                      </label>
                    )}
                    {receipt.observations?.map((observation) => (
                      <div className="fact-observation" key={observation.source_id}>
                        <strong>
                          {
                            detail.sources.find((source) => source.id === observation.source_id)
                              ?.name
                          }
                        </strong>
                        <p>
                          {observation.status} · {observation.value ?? 'not observed'}{' '}
                          {observation.unit}
                        </p>
                        {observation.citations.map((citation, index) => (
                          <button
                            className="citation-button"
                            key={index}
                            onClick={() =>
                              onInspect(observation.source_id, citation.page, citation.region_id)
                            }
                          >
                            Page {citation.page}: {citation.quote}
                          </button>
                        ))}
                      </div>
                    ))}
                    <EvidenceReview
                      result={result}
                      busy={busy}
                      submit={(decision, note) =>
                        action(() =>
                          api.request(`${base}/${selected}/results/${result.id}/reviews`, {
                            method: 'POST',
                            body: { digest: result.payload.digest, decision, note },
                          }),
                        )
                      }
                    />
                  </article>
                );
              })}
              <DocumentClaimAudits
                key={selected}
                api={api}
                setId={selected}
                fingerprint={detail.collection_fingerprint}
                results={detail.results}
                sources={detail.sources}
                onInspect={onInspect}
              />
              <div className="reconciliation-delete">
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={deleting}
                    disabled={busy}
                    onChange={(e) => setDeleting(e.target.checked)}
                  />
                  Confirm deletion of this set, saved evidence, claim audits and its reviews; source
                  documents will remain.
                </label>
                <button
                  className="button secondary"
                  disabled={busy || !deleting}
                  onClick={() =>
                    void action(async () => {
                      await api.request(`${base}/${selected}`, { method: 'DELETE' });
                      setSelected('');
                      onUpdated();
                    })
                  }
                >
                  Delete evidence set
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function EvidenceReview({
  result,
  busy,
  submit,
}: {
  result: Result;
  busy: boolean;
  submit: (decision: string, note: string) => Promise<void>;
}) {
  const [decision, setDecision] = useState('needs_review');
  const [note, setNote] = useState('');
  const [checked, setChecked] = useState(false);
  if (result.review)
    return (
      <p className="small-text">
        Immutable owner review: {result.review.decision} · {result.review.owner}. This does not
        change the evidence status or resolve a conflict.
      </p>
    );
  return (
    <details>
      <summary>Record a scoped evidence review</summary>
      <label>
        Evidence review decision
        <select value={decision} disabled={busy} onChange={(e) => setDecision(e.target.value)}>
          <option value="needs_review">Needs further review</option>
          <option value="inspected">I inspected these observations</option>
          <option value="flagged">Flag for attention</option>
        </select>
      </label>
      <label>
        Evidence review note
        <textarea
          value={note}
          maxLength={1000}
          disabled={busy}
          onChange={(e) => setNote(e.target.value)}
        />
      </label>
      <label className="check-label">
        <input
          type="checkbox"
          checked={checked}
          disabled={busy}
          onChange={(e) => setChecked(e.target.checked)}
        />
        I inspected the scoped evidence. This is not an independent approval or conflict resolution.
      </label>
      <button
        className="button secondary"
        disabled={busy || !checked}
        onClick={() => void submit(decision, note)}
      >
        Record evidence review
      </button>
    </details>
  );
}
