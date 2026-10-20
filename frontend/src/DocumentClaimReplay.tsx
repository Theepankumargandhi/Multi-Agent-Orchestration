import { useEffect, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';

type Check = {
  status: string;
  reason: string;
  observed_value: string | null;
  observed_unit: string | null;
};
type Replay = {
  id: string;
  name: string;
  target_set_id: string;
  target_sources: { id: string; name: string }[];
  checkpoint_fingerprint: string;
  complete: boolean;
  processed: number;
  total: number;
  all_claims_supported: boolean;
  support_lost: number;
  planned_result_upper_bound: number;
  items: {
    claim: { claim_id: string; label: string; expected_value: string; expected_unit: string };
    before: Check;
    after: Check | null;
    comparison: { transition: string; observed_value_changed: boolean | null } | null;
    observations: {
      source_id: string;
      status: string;
      citations: { page: number; region_id: string; quote: string }[];
    }[];
  }[];
};

export function DocumentClaimReplay({
  api,
  setId,
  auditId,
  sourceCheckpoint,
  complete,
  onInspect,
}: {
  api: ApiClient;
  setId: string;
  auditId: string;
  sourceCheckpoint: string;
  complete: boolean;
  onInspect: (documentId: string, page: number, regionId: string) => void;
}) {
  const base = `/workbench/documents/reconciliations/${setId}/audits/${auditId}/replays`;
  const [open, setOpen] = useState(false);
  const [targets, setTargets] = useState<{ id: string; name: string }[]>([]);
  const [target, setTarget] = useState('');
  const [context, setContext] = useState<{
    collection_fingerprint: string;
    scope_label: string;
  } | null>(null);
  const [name, setName] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [history, setHistory] = useState<{ id: string; name: string }[]>([]);
  const [selected, setSelected] = useState('');
  const [report, setReport] = useState<Replay | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!open || !complete) return;
    const controller = new AbortController();
    void Promise.all([
      api.request<{ sets: { id: string; name: string }[] }>(
        '/workbench/documents/reconciliations',
        { signal: controller.signal },
      ),
      api.request<{ replays: { id: string; name: string }[] }>(base, { signal: controller.signal }),
    ])
      .then(([sets, runs]) => {
        if (!controller.signal.aborted) {
          setTargets(sets.sets.filter((s) => s.id !== setId));
          setHistory(runs.replays);
        }
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, setId, complete, open, refresh]);
  useEffect(() => {
    setContext(null);
    setConfirmed(false);
    if (!target || !open) return;
    const controller = new AbortController();
    void api
      .request<{ collection_fingerprint: string; scope_label: string }>(
        `/workbench/documents/reconciliations/${target}`,
        { signal: controller.signal },
      )
      .then((value) => {
        if (!controller.signal.aborted) setContext(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, open, target]);
  useEffect(() => {
    setDeleting(false);
  }, [selected]);
  useEffect(() => {
    setReport(null);
    if (!selected || !open) return;
    const controller = new AbortController();
    void api
      .request<Replay>(`${base}/${selected}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setReport(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, selected, open, refresh]);
  async function act(work: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await work();
      setRefresh((v) => v + 1);
    } catch (e) {
      if (!isAbort(e))
        setError(
          `${errorText(e)} Refresh before retrying an uncertain step. Completed earlier steps remain saved.`,
        );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="claim-replay-panel">
      <h4>Replay against changed evidence</h4>
      <p>
        Rerun these exact assertions on another frozen evidence set. Expected values, units, literal
        labels and calculation operand order do not change.
      </p>
      {!complete ? (
        <p className="muted">Complete the source claim audit before replaying it.</p>
      ) : (
        <>
          <button className="button secondary" disabled={busy} onClick={() => setOpen((v) => !v)}>
            {open ? 'Close audit replays' : 'Open audit replays'}
          </button>
          {open && (
            <>
              {error && <Notice error>{error}</Notice>}
              <div className="change-controls">
                <label>
                  Replay name
                  <input
                    value={name}
                    minLength={3}
                    maxLength={80}
                    disabled={busy}
                    onChange={(e) => setName(e.target.value)}
                  />
                </label>
                <label>
                  Target frozen evidence set
                  <select
                    value={target}
                    disabled={busy}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    <option value="">Choose a different evidence set</option>
                    {targets.map((s) => (
                      <option value={s.id} key={s.id}>
                        {s.name}
                      </option>
                    ))}
                  </select>
                </label>
                {context && <p>Target declared scope: {context.scope_label}</p>}
                <label className="check-label">
                  <input
                    type="checkbox"
                    disabled={busy || !context}
                    checked={confirmed}
                    onChange={(e) => setConfirmed(e.target.checked)}
                  />
                  I inspected the target versions and confirm the same entity, period, metric
                  definition, units and scale. Declared scope must match exactly; the application
                  does not verify these semantics.
                </label>
                <button
                  className="button"
                  disabled={busy || name.trim().length < 3 || !target || !context || !confirmed}
                  onClick={() =>
                    void act(async () => {
                      const result = await api.request<{ id: string }>(base, {
                        method: 'POST',
                        body: {
                          name: name.trim(),
                          target_set_id: target,
                          source_checkpoint_fingerprint: sourceCheckpoint,
                          target_collection_fingerprint: context!.collection_fingerprint,
                          scope_confirmed: confirmed,
                        },
                      });
                      setSelected(result.id);
                      setConfirmed(false);
                    })
                  }
                >
                  Create audit replay
                </button>
              </div>
              <div className="button-row">
                <label>
                  Saved audit replays
                  <select
                    value={selected}
                    disabled={busy}
                    onChange={(e) => setSelected(e.target.value)}
                  >
                    <option value="">Choose a replay</option>
                    {history.map((run) => (
                      <option key={run.id} value={run.id}>
                        {run.name}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="button secondary"
                  disabled={busy}
                  onClick={() => setRefresh((v) => v + 1)}
                >
                  Refresh audit replays
                </button>
              </div>
              {report && (
                <div className="claim-replay-report">
                  <h4>{report.name}</h4>
                  <Badge>
                    {report.complete
                      ? 'Replay complete'
                      : 'Partial replay — unchecked claims remain'}
                  </Badge>
                  <p>
                    Replayed {report.processed}/{report.total} · lost support: {report.support_lost}
                  </p>
                  <Notice>
                    {report.all_claims_supported
                      ? 'All replayed numeric assertions match this target evidence. This is not source truth, semantic entailment or independent approval.'
                      : 'Not all assertions have support on the target. Complete does not mean passed; partial, refuted and unresolved results remain visible.'}
                  </Notice>
                  <p className="small-text muted">
                    Fresh results are stored in the target set, without copying owner reviews.
                    Worst-case planned saved results: {report.planned_result_upper_bound}/20;
                    existing target results also consume capacity.
                  </p>
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
                    Replay next two claims
                  </button>
                  {report.items.map((item) => (
                    <article className="document-result" key={item.claim.claim_id}>
                      <strong>{item.claim.label}</strong>
                      <p>
                        Unchanged assertion: {item.claim.expected_value} {item.claim.expected_unit}
                      </p>
                      <p>
                        Before: {item.before.status} ·{' '}
                        {item.before.observed_value ?? 'no released value'}{' '}
                        {item.before.observed_unit}
                      </p>
                      <p>
                        After: {item.after?.status ?? 'pending'} ·{' '}
                        {item.after?.observed_value ?? 'no released value'}{' '}
                        {item.after?.observed_unit}
                      </p>
                      <p>
                        {item.comparison?.transition ?? 'Not replayed yet'}
                        {item.comparison?.observed_value_changed === true
                          ? ' · verified observed value changed'
                          : ''}
                      </p>
                      {item.after && (
                        <p className="small-text muted">
                          {item.after.reason}. A transition describes this assertion only, not
                          overall answer quality.
                        </p>
                      )}
                      {item.observations.map((observation, index) => (
                        <details key={index}>
                          <summary>
                            {report.target_sources.find((s) => s.id === observation.source_id)
                              ?.name ?? observation.source_id}{' '}
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
                              Target page {citation.page}: {citation.quote}
                            </button>
                          ))}
                        </details>
                      ))}
                    </article>
                  ))}
                  <label className="check-label">
                    <input
                      type="checkbox"
                      disabled={busy}
                      checked={deleting}
                      onChange={(e) => setDeleting(e.target.checked)}
                    />
                    Confirm deletion of this replay report only; its saved target results, source
                    audit and documents will remain.
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
                    Delete audit replay
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}
