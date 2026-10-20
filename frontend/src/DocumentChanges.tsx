import { useEffect, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';

export type Version = {
  id: string;
  name: string;
  digest: string;
  parent_id?: string | null;
  family_id?: string;
};
type SavedResult = {
  id: string;
  payload: { request: { question?: string; operation?: string }; receipt: { status: string } };
};
type Source = { manifest: { fingerprint: string } | null; results: SavedResult[] };
type RunSummary = { id: string; processed: number; total: number; complete: boolean };
type Entry = {
  item_id: string;
  status: string;
  reason: string;
  result_changed: boolean | null;
  impact: { status: string; reasons: string[]; requires_reissue: boolean };
  answer?: { answer: string; status: string } | null;
  replay?: { result?: { value: string | null; unit: string | null } | null } | null;
};
type Run = RunSummary & {
  checkpoint_fingerprint: string;
  before_id: string;
  after_id: string;
  config: { refresh_answers: boolean; replay_calculations: boolean; max_answer_attempts: number };
  changes: {
    status: string;
    reason: string;
    metadata_changes: string[];
    regions: {
      relation: string;
      flags: string[];
      before: { page: number } | null;
      after: { page: number } | null;
    }[];
  };
  entries: Entry[];
  items: {
    source_result_id: string;
    target_result_id: string | null;
    title: string;
    old_value?: string | null;
    old_unit?: string | null;
    review: { owner: string; decision: string } | null;
  }[];
  review_queue: { item_id: string; category: string; reason: string }[];
};

function ancestors(target: Version, versions: Version[]) {
  const found: Version[] = [];
  let parent = target.parent_id;
  for (let step = 0; parent && step < 10; step++) {
    const version = versions.find((v) => v.id === parent);
    if (!version) break;
    found.push(version);
    parent = version.parent_id;
  }
  return found;
}

export function DocumentChanges({
  api,
  target,
  versions,
  fingerprint,
  onUpdated,
  onOpenResult,
}: {
  api: ApiClient;
  target: Version;
  versions: Version[];
  fingerprint: string;
  onUpdated: () => Promise<void>;
  onOpenResult: (id: string) => void;
}) {
  const [before, setBefore] = useState('');
  const [source, setSource] = useState<Source | null>(null);
  const [ids, setIds] = useState<string[]>([]);
  const [refreshLocal, setRefreshLocal] = useState(false);
  const [attempts, setAttempts] = useState(10);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selected, setSelected] = useState('');
  const [run, setRun] = useState<Run | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const choices = ancestors(target, versions);
  const base = `/workbench/documents/${target.id}/comparisons`;
  useEffect(() => {
    if (!target.parent_id) return;
    const controller = new AbortController();
    void api
      .request<{ comparisons: RunSummary[] }>(base, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setRuns(value.comparisons);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, refresh, target.parent_id]);
  useEffect(() => {
    setSource(null);
    setIds([]);
    if (!before) return;
    const controller = new AbortController();
    void api
      .request<Source>(`/workbench/documents/${before}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setSource(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, before]);
  useEffect(() => {
    setRun(null);
    if (!selected) return;
    const controller = new AbortController();
    void api
      .request<Run>(`${base}/${selected}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setRun(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, selected, refresh]);
  async function action(work: () => Promise<void>) {
    setBusy(true);
    setError('');
    try {
      await work();
      setRefresh((value) => value + 1);
    } catch (e) {
      if (!isAbort(e))
        setError(`${errorText(e)} Refresh this comparison before retrying an uncertain step.`);
    } finally {
      setBusy(false);
    }
  }
  async function create() {
    if (!source?.manifest || !ids.length) return;
    await action(async () => {
      const value = await api.request<{ id: string }>(base, {
        method: 'POST',
        body: {
          before_id: before,
          before_fingerprint: source.manifest!.fingerprint,
          after_fingerprint: fingerprint,
          result_ids: ids,
          refresh_local: refreshLocal,
          max_answer_attempts: attempts,
        },
      });
      setSelected(value.id);
    });
  }
  return (
    <section className="panel document-change-panel">
      <div className="section-title">
        <h2>Version impact & revalidation</h2>
        <Badge>Local · explicit steps</Badge>
      </div>
      <p>
        New evidence can invalidate an old answer. Compare an explicitly linked ancestor, then
        assess or refresh up to ten saved results. Old owner reviews never transfer.
      </p>
      {!target.parent_id ? (
        <p className="muted">
          To start, upload a new report with “New version of” set to this document. Existing
          independent uploads are not silently relabeled.
        </p>
      ) : (
        <>
          {error && <Notice error>{error}</Notice>}
          <div className="change-controls">
            <label>
              Earlier version
              <select
                aria-label="Earlier version"
                value={before}
                disabled={busy}
                onChange={(e) => setBefore(e.target.value)}
              >
                <option value="">Choose an ancestor version</option>
                {choices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} · {v.digest.slice(0, 8)}
                  </option>
                ))}
              </select>
            </label>
            {source && (
              <fieldset disabled={busy}>
                <legend>Saved results to check ({ids.length}/10)</legend>
                {source.results.length === 0 && <p>No saved result in this version.</p>}
                {source.results.map((result) => (
                  <label className="check-label change-saved-item" key={result.id}>
                    <input
                      type="checkbox"
                      checked={ids.includes(result.id)}
                      disabled={!ids.includes(result.id) && ids.length >= 10}
                      onChange={(e) =>
                        setIds((value) =>
                          e.target.checked
                            ? [...value, result.id]
                            : value.filter((id) => id !== result.id),
                        )
                      }
                    />
                    {result.payload.request.question || result.payload.request.operation} ·{' '}
                    {result.payload.receipt.status}
                  </label>
                ))}
              </fieldset>
            )}
            <label className="check-label">
              <input
                type="checkbox"
                checked={refreshLocal}
                disabled={busy}
                onChange={(e) => setRefreshLocal(e.target.checked)}
              />{' '}
              Enable fresh local retrieval and aligned calculation replay for this comparison
            </label>
            <label>
              Maximum fresh answer attempts
              <input
                aria-label="Maximum fresh answer attempts"
                type="number"
                min={0}
                max={10}
                value={attempts}
                disabled={busy}
                onChange={(e) => setAttempts(Number(e.target.value))}
              />
            </label>
            <p className="muted small-text">
              Unchecked means assessment only. Creation performs no refresh. Each explicit advance
              checks at most two items. These are local retrieval attempts, not provider requests or
              a billing cap.
            </p>
            <button
              className="button secondary"
              disabled={
                busy ||
                !source?.manifest ||
                !ids.length ||
                !Number.isInteger(attempts) ||
                attempts < 0 ||
                attempts > 10
              }
              onClick={() => void create()}
            >
              Create comparison
            </button>
          </div>
          <div className="button-row">
            <label>
              Saved comparisons
              <select
                aria-label="Saved comparisons"
                value={selected}
                disabled={busy}
                onChange={(e) => setSelected(e.target.value)}
              >
                <option value="">Choose a comparison</option>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id.slice(0, 8)} · {r.processed}/{r.total} ·{' '}
                    {r.complete ? 'complete' : 'partial'}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => setRefresh((value) => value + 1)}
            >
              Refresh comparison
            </button>
          </div>
          {run && (
            <div className="change-report">
              <h3>
                {run.processed}/{run.total} checked ·{' '}
                {run.complete ? 'Complete' : 'Partial — unchecked items remain'}
              </h3>
              <p>
                <Badge>{run.changes.status}</Badge> {run.changes.reason}. Extraction-level alignment
                is not a semantic or pixel diff.
              </p>
              <p>
                Fresh retrieval {run.config.refresh_answers ? 'enabled' : 'disabled'} · calculation
                replay {run.config.replay_calculations ? 'enabled' : 'disabled'} · answer-attempt
                cap {run.config.max_answer_attempts} · 0 model calls.
              </p>
              <details>
                <summary>
                  Extraction changes (
                  {run.changes.regions.filter((r) => r.relation !== 'unchanged').length})
                </summary>
                <p>Metadata: {run.changes.metadata_changes.join(', ') || 'unchanged'}</p>
                <ol>
                  {run.changes.regions
                    .filter((r) => r.relation !== 'unchanged')
                    .map((r, index) => (
                      <li key={index}>
                        {r.relation} · page {r.before?.page ?? '—'} → {r.after?.page ?? '—'} ·{' '}
                        {r.flags.join(', ') || 'no paired region'}
                      </li>
                    ))}
                </ol>
              </details>
              <button
                className="button"
                disabled={busy || run.complete}
                onClick={() =>
                  void action(async () => {
                    await api.request(`${base}/${run.id}/advance`, {
                      method: 'POST',
                      body: { checkpoint_fingerprint: run.checkpoint_fingerprint },
                    });
                    await onUpdated();
                  })
                }
              >
                {busy ? 'Checking…' : 'Advance next two items'}
              </button>
              <h3>Review queue · {run.review_queue.length} processed items need attention</h3>
              {!run.complete && (
                <p className="muted">
                  This queue covers processed items only. It is not an “all clear.”
                </p>
              )}
              {run.review_queue.map((item) => (
                <p className="change-queue-item" key={item.item_id}>
                  {run.items.find((i) => i.source_result_id === item.item_id)?.title} ·{' '}
                  {item.category} · {item.reason}
                </p>
              ))}
              {run.entries.map((entry) => {
                const item = run.items.find((i) => i.source_result_id === entry.item_id);
                return (
                  <article className="document-result" key={entry.item_id}>
                    <strong>{item?.title}</strong>
                    <p>
                      <Badge>{entry.status}</Badge> {entry.reason}
                    </p>
                    <p>
                      Dependency impact: {entry.impact.status} · {entry.impact.reasons.join(', ')}
                    </p>
                    <p>
                      Result changed:{' '}
                      {entry.result_changed === null
                        ? 'not established'
                        : entry.result_changed
                          ? 'yes'
                          : 'no'}
                      . A verified refresh is not an approval.
                    </p>
                    {entry.answer && <p className="preserve-text">{entry.answer.answer}</p>}
                    {entry.replay?.result && (
                      <p>
                        Value: {item?.old_value ?? 'unknown'} {item?.old_unit} →{' '}
                        {entry.replay.result.value ?? 'abstained'} {entry.replay.result.unit}
                      </p>
                    )}
                    {item?.target_result_id && (
                      <button
                        className="text-button"
                        onClick={() => onOpenResult(item.target_result_id!)}
                      >
                        Inspect fresh result and review
                      </button>
                    )}
                    {item?.review && (
                      <p>
                        New-version owner review: {item.review.decision} · {item.review.owner}
                      </p>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </>
      )}
    </section>
  );
}
