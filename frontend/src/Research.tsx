import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClient } from './api';
import type { Capabilities, ChatMessage, Thread } from './types';
import { Badge, Empty, Icon, Notice, PageHeading, errorText, isAbort } from './components';

type Activity = { label: string; detail: string };
export function Research({ api, capabilities }: { api: ApiClient; capabilities: Capabilities }) {
  const [model, setModel] = useState(capabilities.models[0]?.id ?? '');
  const [thread, setThread] = useState(() => crypto.randomUUID() as string);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [draft, setDraft] = useState('');
  const [partial, setPartial] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const active = useRef<AbortController | null>(null);
  const historyRequest = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const refreshHistory = async (signal?: AbortSignal) => {
    const data = await api.request<{ threads: Thread[] }>('/store/threads?limit=30', { signal });
    setThreads(data.threads);
  };
  useEffect(() => {
    const controller = new AbortController();
    void refreshHistory(controller.signal).catch((e) => {
      if (!isAbort(e)) setError(errorText(e));
    });
    return () => {
      controller.abort();
      active.current?.abort();
      historyRequest.current?.abort();
    };
  }, [api]); // The client is session-scoped; no model request is made by this effect.
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [messages, partial]);

  async function loadThread(id: string) {
    if (busy) return;
    historyRequest.current?.abort();
    const controller = new AbortController();
    historyRequest.current = controller;
    setLoading(true);
    setError('');
    try {
      const data = await api.request<{
        messages: { role: string; content: string; run_id?: string }[];
      }>(`/store/${encodeURIComponent(id)}?limit=200`, { signal: controller.signal });
      if (controller.signal.aborted) return;
      setMessages(
        data.messages
          .slice()
          .reverse()
          .filter((m) => ['human', 'ai', 'tool'].includes(m.role))
          .map((m) => ({
            type: m.role as ChatMessage['type'],
            content: m.content,
            run_id: m.run_id,
          })),
      );
      setThread(id);
      setActivity([]);
      setPartial('');
    } catch (e) {
      if (!isAbort(e)) setError(errorText(e));
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }
  function newThread() {
    historyRequest.current?.abort();
    setLoading(false);
    setThread(crypto.randomUUID());
    setMessages([]);
    setActivity([]);
    setPartial('');
    setError('');
  }
  async function send(text: string) {
    if (busy || loading || !capabilities.models.some((m) => m.id === model) || !text.trim()) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setError('');
    setDraft('');
    setPartial('');
    setMessages((previous) => [...previous, { type: 'human', content: text }]);
    setActivity((previous) => [
      ...previous,
      { label: 'Request submitted', detail: `Configured model: ${model}` },
    ]);
    try {
      await api.stream({ message: text, model, thread_id: thread }, controller.signal, (event) => {
        if (event.type === 'token') setPartial((previous) => previous + event.content);
        else {
          const message = event.content;
          if (message.type === 'ai') setPartial('');
          setMessages((previous) => [...previous, message]);
          for (const tool of message.tool_calls ?? [])
            setActivity((previous) => [
              ...previous,
              { label: tool.name, detail: JSON.stringify(tool.args, null, 2) },
            ]);
          if (message.type === 'tool')
            setActivity((previous) => [
              ...previous,
              { label: 'Tool result', detail: message.content },
            ]);
        }
      });
      setActivity((previous) => [
        ...previous,
        {
          label: 'Stream completed',
          detail: 'The server sent its completion marker. Inspect the answer and evidence.',
        },
      ]);
      await refreshHistory(controller.signal);
    } catch (e) {
      setError(
        isAbort(e)
          ? 'Stopped viewing this stream. Backend work may continue; check history before submitting again.'
          : errorText(e),
      );
    } finally {
      setBusy(false);
      active.current = null;
    }
  }
  const checked = Boolean(capabilities.features.grounding_verification);
  const pendingApproval =
    messages.at(-1)?.type === 'ai' &&
    messages.at(-1)?.content.includes('Human approval required before web-answer generation.');
  const submit = (event: FormEvent) => {
    event.preventDefault();
    void send(draft.trim());
  };
  return (
    <>
      <PageHeading
        eyebrow="CITERIGHT"
        title="Ask. Check the sources."
        actions={<Badge tone="blue">Connected API</Badge>}
      >
        Answers come from your documents and the web, with a citation for each claim.
      </PageHeading>
      <div className="workspace-grid">
        <section className="panel chat-panel" aria-label="Research conversation">
          <div className="panel-toolbar">
            <label className="compact-label">
              Model
              <select
                aria-label="Research model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={busy}
              >
                {capabilities.models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <button className="button secondary small" onClick={newThread} disabled={busy}>
              + New conversation
            </button>
          </div>
          <div className="chat-scroll">
            {!messages.length && (
              <Empty title="A clear question is a good start.">
                Ask about your documents, request a calculation, or research a topic on the web. Web
                searches wait for your approval.
              </Empty>
            )}
            {messages
              .filter((m) => m.type !== 'tool' && m.content)
              .map((message, index) => (
                <article
                  className={`message ${message.type}`}
                  key={`${index}-${message.run_id ?? ''}`}
                >
                  <div className="message-label">
                    {message.type === 'human' ? 'YOU' : 'CITERIGHT'}
                    {message.run_id && (
                      <span title={message.run_id}> · {message.run_id.slice(0, 8)}</span>
                    )}
                  </div>
                  <div className="message-text">{message.content}</div>
                </article>
              ))}
            {partial && (
              <article className="message ai">
                <div className="message-label">
                  {checked ? 'CITERIGHT · CHECKING EACH SENTENCE' : 'CITERIGHT · STREAMING'}
                </div>
                <div className="message-text">{partial}</div>
              </article>
            )}
            {busy && !partial && (
              <p className="working" role="status">
                <span className="pulse" /> {checked ? 'Finding and checking sources…' : 'Working…'}
              </p>
            )}
            <div ref={bottom} />
          </div>
          {error && <Notice error>{error}</Notice>}
          {pendingApproval && (
            <div className="approval-box">
              <strong>Web access is awaiting your decision</strong>
              <p>Approval resumes this conversation and may make provider requests.</p>
              <div className="button-row">
                <button className="button" disabled={busy} onClick={() => void send('approve')}>
                  Approve web access
                </button>
                <button
                  className="button secondary"
                  disabled={busy}
                  onClick={() => void send('reject')}
                >
                  Reject
                </button>
              </div>
            </div>
          )}
          <form className="composer" onSubmit={submit}>
            <label className="sr-only" htmlFor="question">
              Your question
            </label>
            <textarea
              id="question"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              maxLength={20000}
              rows={3}
              placeholder="Ask a question grounded in your evidence…"
              disabled={busy || loading}
            />
            <div className="composer-footer">
              <span>
                {checked
                  ? 'Each sentence appears once its sources support it.'
                  : 'Source checking is off on this server.'}
              </span>
              {busy ? (
                <button
                  type="button"
                  className="button secondary"
                  onClick={() => active.current?.abort()}
                >
                  Stop viewing
                </button>
              ) : (
                <button className="button" disabled={!model || !draft.trim() || loading}>
                  Send question <Icon name="arrow" size={16} />
                </button>
              )}
            </div>
          </form>
        </section>
        <aside className="inspector">
          <section className="panel">
            <div className="section-title">
              <h2>Conversations</h2>
              <Badge>{threads.length} recent</Badge>
            </div>
            {threads.length ? (
              <div className="history-list">
                {threads.map((item) => (
                  <button
                    className={`history-item ${item.thread_id === thread ? 'selected' : ''}`}
                    disabled={busy || loading}
                    key={item.thread_id}
                    onClick={() => void loadThread(item.thread_id)}
                  >
                    <span>{item.last_message_preview || item.thread_id.slice(0, 12)}</span>
                    <small>{item.message_count ?? '—'} messages</small>
                  </button>
                ))}
              </div>
            ) : (
              <p className="muted small-text">Saved conversations will appear here.</p>
            )}
          </section>
          <section className="panel">
            <div className="section-title">
              <h2>Execution activity</h2>
              <span className="tiny-dot" />
            </div>
            <p className="muted small-text">
              Observed actions, not hidden reasoning. This session only.
            </p>
            {activity.length ? (
              <ol className="timeline">
                {activity.slice(-40).map((item, index) => (
                  <li key={index}>
                    <strong>{item.label}</strong>
                    <details>
                      <summary>Inspect event</summary>
                      <pre>{item.detail.slice(0, 12000)}</pre>
                    </details>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="quiet-state">No execution events yet.</div>
            )}
          </section>
          <div className="trust-note">
            <Icon name="shield" />
            <p>Source citations require review. A fluent answer is not a correctness guarantee.</p>
          </div>
        </aside>
      </div>
    </>
  );
}
