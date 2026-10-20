import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClient } from './api';
import type { CodeTask, TaskEvent, TaskSummary, TestResult, WorkbenchConfig } from './types';
import { Badge, Empty, Icon, Notice, PageHeading, errorText, isAbort } from './components';

const activeStatuses = new Set(['queued', 'running', 'retry_wait']);
const statusTone = (status: string) =>
  status === 'approved'
    ? 'green'
    : ['failed', 'dead_letter', 'rejected'].includes(status)
      ? 'red'
      : 'blue';
function TestOutput({ title, result }: { title: string; result?: TestResult | null }) {
  return (
    <details className="test-output">
      <summary>
        {title}
        <Badge tone={result ? (result.exit_code === 0 && !result.timed_out ? 'green' : 'red') : ''}>
          {!result ? 'Not reported' : result.timed_out ? 'Timed out' : `Exit ${result.exit_code}`}
        </Badge>
      </summary>
      {result && (
        <pre>{(result.stdout + '\n' + result.stderr).slice(0, 24000) || 'No output captured.'}</pre>
      )}
    </details>
  );
}
export function Coding({
  api,
  config,
  allowCalls,
}: {
  api: ApiClient;
  config: WorkbenchConfig;
  allowCalls: boolean;
}) {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [selected, setSelected] = useState('');
  const selectedRef = useRef(selected);
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);
  const [task, setTask] = useState<CodeTask | null>(null);
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [repository, setRepository] = useState(config.repositories[0] ?? '');
  const [model, setModel] = useState(config.coding_models[0] ?? '');
  const [issue, setIssue] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [reason, setReason] = useState('');
  const [reviewed, setReviewed] = useState(false);
  const [patch, setPatch] = useState('');
  const [dossier, setDossier] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [openId, setOpenId] = useState('');
  const idempotency = useRef({ payload: '', key: '' });
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  async function loadHistory(signal?: AbortSignal) {
    const data = await api.request<{ tasks: TaskSummary[] }>('/workbench/code/tasks?limit=30', {
      signal,
    });
    if (mounted.current) setTasks(data.tasks);
  }
  useEffect(() => {
    if (!config.code_enabled) return;
    const controller = new AbortController();
    void loadHistory(controller.signal).catch((e) => {
      if (!isAbort(e)) setError(errorText(e));
    });
    return () => controller.abort();
  }, [api, config.code_enabled]);

  useEffect(() => {
    if (!selected || !config.code_enabled) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setTask(null);
    setPatch('');
    setDossier('');
    setEvents([]);
    setReviewed(false);
    setError('');
    const poll = async () => {
      try {
        const [record, timeline] = await Promise.all([
          api.request<CodeTask>(`/code/tasks/${encodeURIComponent(selected)}`, {
            signal: controller.signal,
          }),
          api.request<{ events: TaskEvent[] }>(
            `/code/tasks/${encodeURIComponent(selected)}/events`,
            { signal: controller.signal },
          ),
        ]);
        if (controller.signal.aborted) return;
        setTask(record);
        setEvents(timeline.events);
        if (activeStatuses.has(record.status)) timer = setTimeout(() => void poll(), 5000);
      } catch (e) {
        if (!isAbort(e)) setError(errorText(e));
      }
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [api, selected, refresh, config.code_enabled]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!allowCalls || !repository || !model || issue.trim().length < 10 || busy) return;
    const payload = { repository, model, issue: issue.trim() };
    const canonical = JSON.stringify(payload);
    if (idempotency.current.payload !== canonical)
      idempotency.current = { payload: canonical, key: crypto.randomUUID() };
    setBusy(true);
    setError('');
    try {
      const record = await api.request<CodeTask>('/workbench/code/tasks', {
        method: 'POST',
        body: payload,
        idempotencyKey: idempotency.current.key,
      });
      if (!mounted.current) return;
      setSelected(record.task_id);
      setRefresh((value) => value + 1);
      setIssue('');
      idempotency.current = { payload: '', key: '' };
      await loadHistory();
    } catch (e) {
      if (mounted.current)
        setError(
          errorText(e) +
            ' Submission may have reached the server. Refresh history; an unchanged retry reuses its idempotency key.',
        );
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function decide(approve: boolean) {
    if (!task || task.status !== 'awaiting_approval' || busy || (approve && !reviewed)) return;
    setBusy(true);
    setError('');
    try {
      const record = await api.request<CodeTask>(
        `/code/tasks/${encodeURIComponent(task.task_id)}/decision`,
        { method: 'POST', body: { approve, reason } },
      );
      if (!mounted.current) return;
      setTask(record);
      setRefresh((value) => value + 1);
      setReason('');
      await loadHistory();
    } catch (e) {
      if (mounted.current)
        setError(errorText(e) + ' Refresh task state before making another decision.');
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  async function inspect(kind: 'patch' | 'dossier') {
    if (!task || busy) return;
    const id = task.task_id;
    setBusy(true);
    setError('');
    try {
      const response = await api.request<unknown>(`/code/tasks/${encodeURIComponent(id)}/${kind}`);
      if (!mounted.current || selectedRef.current !== id) return;
      if (kind === 'patch') {
        const value = response as { patch?: unknown };
        if (typeof value.patch !== 'string') throw new Error('Invalid patch response.');
        setPatch(value.patch);
      } else setDossier(JSON.stringify(response, null, 2));
    } catch (e) {
      if (mounted.current) setError(errorText(e));
    } finally {
      if (mounted.current) setBusy(false);
    }
  }
  function downloadPatch() {
    if (!patch || task?.status !== 'approved') return;
    const url = URL.createObjectURL(new Blob([patch], { type: 'text/plain;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'agentforge.patch';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const result = task?.result;
  return (
    <>
      <PageHeading
        eyebrow="CODING WORKSPACE"
        title="A patch you can inspect."
        actions={
          <Badge tone={config.code_enabled ? 'blue' : ''}>
            {config.code_enabled ? config.workflow : 'Requires configuration'}
          </Badge>
        }
      >
        Submit a bounded repair task. Review execution evidence before releasing patch access.
      </PageHeading>
      {!config.code_enabled ? (
        <section className="panel">
          <Empty title="Coding execution is not enabled.">
            An operator must configure the sandbox, provider and repository root, then enable
            CODE_AGENT_ENABLED. The browser cannot enable it or access arbitrary host paths.
          </Empty>
        </section>
      ) : (
        <div className="coding-grid">
          <aside className="coding-sidebar">
            <section className="panel">
              <div className="section-title">
                <h2>New repair task</h2>
                <Icon name="coding" />
              </div>
              <form onSubmit={submit} className="stack-form">
                <label>
                  Repository
                  <select
                    value={repository}
                    onChange={(e) => setRepository(e.target.value)}
                    required
                    disabled={busy}
                    aria-label="Coding repository"
                  >
                    <option value="" disabled>
                      Select an enabled repository
                    </option>
                    {config.repositories.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                {!config.repositories.length && (
                  <Notice>
                    No repositories are enabled for browser submission. Configure
                    WORKBENCH_REPOSITORIES on the server.
                  </Notice>
                )}
                <label>
                  Model
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    required
                    disabled={busy}
                    aria-label="Coding model"
                  >
                    <option value="" disabled>
                      Select a configured model
                    </option>
                    {config.coding_models.map((name) => (
                      <option key={name}>{name}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Issue
                  <textarea
                    rows={6}
                    value={issue}
                    onChange={(e) => setIssue(e.target.value)}
                    minLength={10}
                    maxLength={12000}
                    required
                    placeholder="Describe the failure, expected behavior, and acceptance criteria…"
                    disabled={busy}
                  />
                </label>
                <div className="command">
                  <small>SERVER-OWNED TEST COMMAND</small>
                  <code>{config.test_command.join(' ')}</code>
                </div>
                <button
                  className="button full"
                  disabled={
                    !allowCalls || busy || !repository || !model || issue.trim().length < 10
                  }
                >
                  Submit repair task <Icon name="arrow" size={16} />
                </button>
                <p className="muted small-text">
                  Submitting may use provider quota and launch a sandbox. No patch is applied to the
                  source repository.
                </p>
              </form>
            </section>
            <section className="panel">
              <div className="section-title">
                <h2>Your recent tasks</h2>
                <button
                  className="icon-button"
                  title="Refresh task history"
                  aria-label="Refresh task history"
                  onClick={() => void loadHistory().catch((e) => setError(errorText(e)))}
                >
                  <Icon name="refresh" size={16} />
                </button>
              </div>
              <div className="history-list">
                {tasks.map((item) => (
                  <button
                    className={`history-item ${item.task_id === selected ? 'selected' : ''}`}
                    disabled={busy}
                    key={item.task_id}
                    onClick={() => setSelected(item.task_id)}
                  >
                    <span>{item.issue_preview}</span>
                    <small>
                      {item.repository} · {item.status.replaceAll('_', ' ')}
                    </small>
                  </button>
                ))}
              </div>
              {!tasks.length && (
                <p className="muted small-text">Only tasks owned by your account appear here.</p>
              )}
              <form
                className="open-task"
                onSubmit={(e) => {
                  e.preventDefault();
                  if (!busy && /^[a-zA-Z0-9-]{1,128}$/.test(openId)) setSelected(openId);
                }}
              >
                <label>
                  Open a task by ID
                  <input
                    value={openId}
                    onChange={(e) => setOpenId(e.target.value)}
                    pattern="[a-zA-Z0-9-]{1,128}"
                    maxLength={128}
                    disabled={busy}
                  />
                </label>
                <button className="button secondary small" disabled={busy || !openId}>
                  Open
                </button>
              </form>
            </section>
          </aside>
          <section className="panel task-panel" aria-label="Task evidence">
            {error && <Notice error>{error}</Notice>}
            {!task ? (
              <Empty title={selected ? 'Loading task evidence…' : 'Your next repair starts here.'}>
                {selected
                  ? 'Fetching the owner-scoped record from the service.'
                  : 'Choose a recent task or submit a new issue. Lifecycle events and verification results will appear here.'}
              </Empty>
            ) : (
              <>
                <div className="section-title">
                  <div>
                    <h2>{task.request.repository}</h2>
                    <code className="muted small-text">{task.task_id}</code>
                  </div>
                  <div className="button-row">
                    <Badge tone={statusTone(task.status)}>{task.status.replaceAll('_', ' ')}</Badge>
                    <button
                      className="icon-button"
                      aria-label="Refresh selected task"
                      disabled={busy}
                      onClick={() => setRefresh((value) => value + 1)}
                    >
                      <Icon name="refresh" size={18} />
                    </button>
                  </div>
                </div>
                <p className="task-issue">{task.request.issue}</p>
                {activeStatuses.has(task.status) && (
                  <Notice>
                    Lifecycle status is polled every 5 seconds. Specialist/tool observations are
                    shown only when returned by the backend. Closing this page does not cancel the
                    task.
                  </Notice>
                )}
                {task.error && <Notice error>{task.error}</Notice>}
                <div className="task-metrics">
                  <div>
                    <small>Model calls</small>
                    <strong>{result?.model_calls ?? '—'}</strong>
                  </div>
                  <div>
                    <small>Tool calls</small>
                    <strong>{result?.tool_calls ?? '—'}</strong>
                  </div>
                  <div>
                    <small>Reported tokens</small>
                    <strong>
                      {result?.usage_accounting_complete
                        ? (result.prompt_tokens + result.completion_tokens).toLocaleString()
                        : 'Unknown'}
                    </strong>
                  </div>
                  <div>
                    <small>Verification</small>
                    <strong
                      className={
                        result?.verification?.final_decision === 'verified' ? 'success-text' : ''
                      }
                    >
                      {result?.verification?.final_decision ?? 'Pending'}
                    </strong>
                  </div>
                </div>
                {result && (
                  <>
                    <h3>Execution summary</h3>
                    <p className="preserve">{result.summary}</p>
                    <p className="muted small-text">
                      Changed files: {result.changed_files.join(', ') || 'None reported'}. Dollar
                      cost is not established by this view.
                    </p>
                    {!!result.verification?.blocking_reasons.length && (
                      <Notice error>{result.verification.blocking_reasons.join(' · ')}</Notice>
                    )}
                    <TestOutput title="Baseline tests" result={result.baseline_test} />
                    <TestOutput title="Final tests" result={result.final_test} />
                  </>
                )}
                <h3>Lifecycle events</h3>
                <ol className="timeline horizontal-space">
                  {events.map((event, index) => (
                    <li key={index}>
                      <strong>{event.to_status.replaceAll('_', ' ')}</strong>
                      <small>
                        {event.actor} · {event.occurred_at}
                      </small>
                    </li>
                  ))}
                </ol>
                {!!result?.observations?.length && (
                  <details className="test-output">
                    <summary>
                      Recorded specialist / tool observations
                      <Badge>{result.observations.length}</Badge>
                    </summary>
                    {result.observations.slice(0, 150).map((item, index) => (
                      <div className="observation" key={index}>
                        <Badge tone={item.ok ? 'green' : 'red'}>{item.action}</Badge>{' '}
                        <strong>{item.stage}</strong>
                        <p>{item.summary}</p>
                        {item.output && <pre>{item.output.slice(0, 12000)}</pre>}
                      </div>
                    ))}
                  </details>
                )}
                {task.status === 'awaiting_approval' && (
                  <div className="approval-box">
                    <h3>Review before releasing the patch</h3>
                    <p>
                      Approval unlocks patch access. It does not merge, apply or publish changes.
                      Rejection may delete stored patch artifacts.
                    </p>
                    <label className="check-label">
                      <input
                        type="checkbox"
                        checked={reviewed}
                        onChange={(e) => setReviewed(e.target.checked)}
                      />{' '}
                      I reviewed the execution evidence and want to release patch access.
                    </label>
                    <label>
                      Decision note
                      <textarea
                        value={reason}
                        onChange={(e) => setReason(e.target.value)}
                        rows={2}
                        maxLength={1000}
                      />
                    </label>
                    <div className="button-row">
                      <button
                        className="button"
                        disabled={busy || !reviewed}
                        onClick={() => void decide(true)}
                      >
                        Approve patch access
                      </button>
                      <button
                        className="button danger"
                        disabled={busy}
                        onClick={() => void decide(false)}
                      >
                        Reject patch
                      </button>
                    </div>
                  </div>
                )}
                <div className="button-row artifact-actions">
                  <button
                    className="button secondary"
                    disabled={busy || !result}
                    onClick={() => void inspect('dossier')}
                  >
                    Inspect evidence dossier
                  </button>
                  <button
                    className="button secondary"
                    disabled={busy || task.status !== 'approved'}
                    onClick={() => void inspect('patch')}
                  >
                    View approved patch
                  </button>
                  {patch && (
                    <button className="button" onClick={downloadPatch}>
                      Download patch
                    </button>
                  )}
                </div>
                {dossier && (
                  <details open className="test-output">
                    <summary>Evidence dossier</summary>
                    <pre>{dossier.slice(0, 150000)}</pre>
                  </details>
                )}
                {patch && (
                  <div className="diff">
                    <div className="diff-header">
                      Approved patch · read-only
                      {patch.length > 200000 && (
                        <span>Preview truncated; download for full patch.</span>
                      )}
                    </div>
                    <pre>
                      {patch
                        .slice(0, 200000)
                        .split('\n')
                        .map((line, index) => (
                          <span
                            className={
                              line.startsWith('+')
                                ? 'added'
                                : line.startsWith('-')
                                  ? 'removed'
                                  : line.startsWith('@@')
                                    ? 'hunk'
                                    : ''
                            }
                            key={index}
                          >
                            {line}
                            {'\n'}
                          </span>
                        ))}
                    </pre>
                  </div>
                )}
              </>
            )}
          </section>
        </div>
      )}
    </>
  );
}
