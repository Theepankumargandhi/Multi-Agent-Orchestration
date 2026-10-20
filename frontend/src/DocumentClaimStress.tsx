import { useEffect, useState } from 'react';
import type { ApiClient } from './api';
import { Badge, Notice, errorText, isAbort } from './components';

type Check = {
  status: string;
  reason: string;
  observed_value: string | null;
  observed_unit: string | null;
};
type Variant = {
  omitted_source_id: string;
  check: Check | null;
  effect: string;
  observations: {
    source_id: string;
    status: string;
    citations: { page: number; region_id: string; quote: string }[];
  }[];
};
type Stress = {
  id: string;
  name: string;
  sources: { id: string; name: string }[];
  checkpoint_fingerprint: string;
  processed: number;
  total: number;
  complete: boolean;
  evaluated_checks: number;
  support_lost_cases: number;
  support_appeared_cases: number;
  claims: {
    claim: { claim_id: string; label: string; expected_value: string; expected_unit: string };
    baseline: Check;
    variants: Variant[];
    stable_numeric_support: boolean | null;
  }[];
};

export function DocumentClaimStress({
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
  const base = `/workbench/documents/reconciliations/${setId}/audits/${auditId}/stress-tests`;
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [history, setHistory] = useState<{ id: string; name: string }[]>([]);
  const [selected, setSelected] = useState('');
  const [report, setReport] = useState<Stress | null>(null);
  const [inspection, setInspection] = useState<{ claimId: string; omitted: string } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!open || !complete) return;
    const controller = new AbortController();
    void api
      .request<{ stress_tests: { id: string; name: string }[] }>(base, {
        signal: controller.signal,
      })
      .then((value) => {
        if (!controller.signal.aborted) setHistory(value.stress_tests);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, open, complete, refresh]);
  useEffect(() => {
    setInspection(null);
    setDeleting(false);
  }, [selected]);
  useEffect(() => {
    setReport(null);
    if (!open || !selected) return;
    const controller = new AbortController();
    void api
      .request<Stress>(`${base}/${selected}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setReport(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, base, open, selected, refresh]);
  async function act(work: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await work();
      setRefresh((v) => v + 1);
    } catch (e) {
      if (!isAbort(e)) setError(`${errorText(e)} Refresh before retrying an uncertain step.`);
    } finally {
      setBusy(false);
    }
  }
  const inspectedClaim = report?.claims.find((c) => c.claim.claim_id === inspection?.claimId);
  const inspectedVariant = inspectedClaim?.variants.find(
    (v) => v.omitted_source_id === inspection?.omitted,
  );
  return (
    <section className="claim-stress-panel">
      <h4>Evidence fragility stress test</h4>
      <p>
        Omit each source in a separate diagnostic view. Minimum artifact count, coverage, assertions
        and calculation order stay unchanged. No source is deleted or excluded from the real audit.
      </p>
      {!complete ? (
        <p className="muted">Complete the source audit before testing evidence fragility.</p>
      ) : (
        <>
          <button className="button secondary" disabled={busy} onClick={() => setOpen((v) => !v)}>
            {open ? 'Close evidence stress tests' : 'Open evidence stress tests'}
          </button>
          {open && (
            <>
              {error && <Notice error>{error}</Notice>}
              <div className="change-controls">
                <label>
                  Stress test name
                  <input
                    value={name}
                    minLength={3}
                    maxLength={80}
                    disabled={busy}
                    onChange={(e) => setName(e.target.value)}
                  />
                </label>
                <button
                  className="button"
                  disabled={busy || name.trim().length < 3}
                  onClick={() =>
                    void act(async () => {
                      const value = await api.request<{ id: string }>(base, {
                        method: 'POST',
                        body: {
                          name: name.trim(),
                          source_checkpoint_fingerprint: sourceCheckpoint,
                        },
                      });
                      setSelected(value.id);
                    })
                  }
                >
                  Create evidence stress test
                </button>
              </div>
              <div className="button-row">
                <label>
                  Saved evidence stress tests
                  <select
                    value={selected}
                    disabled={busy}
                    onChange={(e) => setSelected(e.target.value)}
                  >
                    <option value="">Choose a diagnostic report</option>
                    {history.map((run) => (
                      <option value={run.id} key={run.id}>
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
                  Refresh evidence stress tests
                </button>
              </div>
              {report && (
                <div className="claim-stress-report">
                  <h4>{report.name}</h4>
                  <Badge>
                    {report.complete
                      ? 'Single-source tests complete'
                      : 'Partial — source omissions remain unchecked'}
                  </Badge>
                  <p>
                    Scenarios {report.processed}/{report.total} · numeric checks{' '}
                    {report.evaluated_checks} · support lost {report.support_lost_cases} · support
                    appeared after omission {report.support_appeared_cases}
                  </p>
                  <Notice>
                    Diagnostic only. Support appearing after a source is omitted can expose
                    selective-evidence risk; it never settles the original conflict or authorizes
                    ignoring that source. Different file hashes do not prove independent sources.
                    Stability here is not source truth or answer approval.
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
                    Run next source omission
                  </button>
                  <div className="stress-matrix-scroll">
                    <table className="stress-matrix">
                      <caption>
                        Fixed assertions across single-source omission scenarios. Pending cells are
                        not passes.
                      </caption>
                      <thead>
                        <tr>
                          <th scope="col">Assertion</th>
                          <th scope="col">Original audit</th>
                          {report.sources.map((source) => (
                            <th scope="col" key={source.id}>
                              Without {source.name}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {report.claims.map((claim) => (
                          <tr key={claim.claim.claim_id}>
                            <th scope="row">
                              {claim.claim.label}
                              <span className="small-text">
                                Expected {claim.claim.expected_value} {claim.claim.expected_unit}
                              </span>
                            </th>
                            <td>
                              <Badge>{claim.baseline.status}</Badge>
                              <span className="small-text">
                                {claim.baseline.observed_value ?? 'no released value'}{' '}
                                {claim.baseline.observed_unit}
                              </span>
                            </td>
                            {report.sources.map((source) => {
                              const variant = claim.variants.find(
                                (v) => v.omitted_source_id === source.id,
                              );
                              return (
                                <td key={source.id}>
                                  <Badge>{variant?.check?.status ?? 'pending'}</Badge>
                                  <span className="small-text">{variant?.effect ?? 'pending'}</span>
                                  {variant?.check && (
                                    <>
                                      <span className="small-text">
                                        {variant.check.observed_value ?? 'no released value'}{' '}
                                        {variant.check.observed_unit}
                                      </span>
                                      <button
                                        className="text-button"
                                        disabled={busy}
                                        aria-label={`Inspect ${claim.claim.label} without ${source.name}`}
                                        onClick={() =>
                                          setInspection({
                                            claimId: claim.claim.claim_id,
                                            omitted: source.id,
                                          })
                                        }
                                      >
                                        Inspect retained evidence
                                      </button>
                                    </>
                                  )}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {report.claims.map((claim) => (
                    <p className="small-text" key={claim.claim.claim_id}>
                      {claim.claim.label}:{' '}
                      {claim.stable_numeric_support === true
                        ? 'numeric support survived every tested single-source omission'
                        : claim.stable_numeric_support === false
                          ? 'numeric support did not survive every omission'
                          : report.complete
                            ? 'baseline was not supported; no stability claim'
                            : 'incomplete; no stability claim'}
                      .
                    </p>
                  ))}
                  {inspectedVariant?.check && (
                    <section className="stress-inspection">
                      <h5>Retained evidence: {inspectedClaim?.claim.label}</h5>
                      <p>
                        Omitted:{' '}
                        {
                          report.sources.find((s) => s.id === inspectedVariant.omitted_source_id)
                            ?.name
                        }
                        . {inspectedVariant.check.reason}.
                      </p>
                      {inspectedVariant.observations.map((observation, index) => (
                        <details key={index}>
                          <summary>
                            {report.sources.find((s) => s.id === observation.source_id)?.name ??
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
                              Retained page {citation.page}: {citation.quote}
                            </button>
                          ))}
                        </details>
                      ))}
                    </section>
                  )}
                  <label className="check-label">
                    <input
                      type="checkbox"
                      checked={deleting}
                      disabled={busy}
                      onChange={(e) => setDeleting(e.target.checked)}
                    />
                    Confirm deletion of this stress report only; source audit, saved evidence and
                    documents will remain.
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
                    Delete evidence stress test
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
