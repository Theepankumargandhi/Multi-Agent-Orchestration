import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { ApiClient, ApiError } from './api';
import { AnswerText, citationKey, plainText } from './AnswerText';
import type { Citation } from './AnswerText';
import type { Capabilities, ChatMessage, Passage, Thread, UploadedDocument } from './types';
import { Badge, Empty, Icon, Notice, PageHeading, errorText, isAbort } from './components';

type Activity = { label: string; detail: string };
type OpenSource = { message: number; citation: Citation; passage?: Passage; error?: string };
// "local:" makes the server answer from documents; it is a routing hint, not part of the question.
const LOCAL_PREFIX = /^local:\s*/i;
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
  const uploadsEnabled = Boolean(capabilities.features.document_upload);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [uploading, setUploading] = useState('');
  const [fromDocuments, setFromDocuments] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const [source, setSource] = useState<OpenSource | null>(null);
  const sourceRequest = useRef<AbortController | null>(null);
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
    if (uploadsEnabled)
      void api
        .request<{ documents: UploadedDocument[] }>('/documents', { signal: controller.signal })
        .then((data) => setDocuments(data.documents))
        .catch((e) => {
          if (!isAbort(e)) setError(errorText(e));
        });
    return () => {
      controller.abort();
      active.current?.abort();
      historyRequest.current?.abort();
    };
  }, [api, uploadsEnabled]); // The client is session-scoped; no model request is made by this effect.
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
            content: m.role === 'human' ? m.content.replace(LOCAL_PREFIX, '') : m.content,
            run_id: m.run_id,
          })),
      );
      setThread(id);
      setActivity([]);
      setPartial('');
      setSource(null);
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
    setSource(null);
  }
  async function showSource(message: number, citation: Citation) {
    sourceRequest.current?.abort();
    if (source?.message === message && citationKey(source.citation) === citationKey(citation)) {
      setSource(null);
      return;
    }
    const controller = new AbortController();
    sourceRequest.current = controller;
    setSource({ message, citation });
    try {
      const passage = await api.request<Passage>(
        `/sources/passage?source=${encodeURIComponent(citation.source)}&page=${citation.page}`,
        { signal: controller.signal },
      );
      setSource({ message, citation, passage });
    } catch (e) {
      if (isAbort(e)) return;
      setSource({
        message,
        citation,
        error:
          e instanceof ApiError && e.status === 404
            ? 'This page is not in your documents any more.'
            : errorText(e),
      });
    }
  }
  async function upload(file: File) {
    setUploading(file.name);
    setError('');
    try {
      const receipt = await api.upload(file);
      setDocuments((previous) => [
        receipt,
        ...previous.filter((d) => d.file_name !== receipt.file_name),
      ]);
      setFromDocuments(true);
      setActivity((previous) => [
        ...previous,
        {
          label: 'Document indexed',
          detail: `${receipt.file_name}: ${receipt.pages} page(s), ${receipt.chunk_count} passages`,
        },
      ]);
    } catch (e) {
      if (!isAbort(e)) setError(errorText(e));
    } finally {
      setUploading('');
      if (fileInput.current) fileInput.current.value = '';
    }
  }
  async function removeDocument(document: UploadedDocument) {
    setError('');
    try {
      await api.request(`/documents/${document.document_id}`, { method: 'DELETE' });
      const remaining = documents.filter((d) => d.document_id !== document.document_id);
      setDocuments(remaining);
      if (!remaining.length) setFromDocuments(false);
    } catch (e) {
      setError(errorText(e));
    }
  }
  async function send(text: string, fromDocs = false) {
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
      const message = fromDocs ? `local: ${text}` : text;
      await api.stream({ message, model, thread_id: thread }, controller.signal, (event) => {
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
    void send(draft.trim(), fromDocuments && documents.length > 0);
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
                {uploadsEnabled
                  ? 'Upload a document and ask about it, request a calculation, or research a topic on the web.'
                  : 'Ask about your documents, request a calculation, or research a topic on the web.'}{' '}
                Web searches wait for your approval.
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
                  <div className="message-text">
                    {message.type === 'ai' ? (
                      <AnswerText
                        text={message.content}
                        selected={source?.message === index ? citationKey(source.citation) : ''}
                        onCite={(citation) => void showSource(index, citation)}
                      />
                    ) : (
                      message.content
                    )}
                  </div>
                  {source?.message === index && (
                    <aside className="source-passage" aria-live="polite">
                      <div className="source-passage-head">
                        <span>
                          <strong>{source.citation.source}</strong> · page {source.citation.page}
                          {source.passage?.section &&
                          source.passage.section !== `Page ${source.citation.page}`
                            ? ` · ${source.passage.section}`
                            : ''}
                        </span>
                        <button className="link-button" onClick={() => setSource(null)}>
                          Close
                        </button>
                      </div>
                      {source.error ? (
                        <p className="muted small-text">{source.error}</p>
                      ) : source.passage ? (
                        <blockquote>
                          {source.passage.text}
                          {source.passage.truncated ? ' …' : ''}
                        </blockquote>
                      ) : (
                        <p className="muted small-text">Loading the cited page…</p>
                      )}
                    </aside>
                  )}
                </article>
              ))}
            {partial && (
              <article className="message ai">
                <div className="message-label">
                  {checked ? 'CITERIGHT · CHECKING EACH SENTENCE' : 'CITERIGHT · STREAMING'}
                </div>
                <div className="message-text">
                  <AnswerText text={partial} />
                </div>
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
              <div className="composer-options">
                {documents.length > 0 && (
                  <label className="toggle">
                    <input
                      type="checkbox"
                      checked={fromDocuments}
                      onChange={(e) => setFromDocuments(e.target.checked)}
                      disabled={busy}
                    />
                    Answer from my documents
                  </label>
                )}
                <span>
                  {checked
                    ? 'Each sentence appears once its sources support it.'
                    : 'Source checking is off on this server.'}
                </span>
              </div>
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
          {uploadsEnabled && (
            <section className="panel">
              <div className="section-title">
                <h2>Your documents</h2>
                <Badge>
                  {documents.length} {documents.length === 1 ? 'file' : 'files'}
                </Badge>
              </div>
              <p className="muted small-text">
                PDF, Markdown or text. Only you can search the files you upload.
              </p>
              {documents.length > 0 && (
                <ul className="document-list">
                  {documents.map((document) => (
                    <li key={document.document_id}>
                      <span>
                        <strong title={document.file_name}>{document.file_name}</strong>
                        <small>
                          {document.pages} {document.pages === 1 ? 'page' : 'pages'}
                        </small>
                      </span>
                      <button
                        className="link-button"
                        aria-label={`Remove ${document.file_name}`}
                        onClick={() => void removeDocument(document)}
                        disabled={Boolean(uploading)}
                      >
                        Remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}
              <input
                ref={fileInput}
                type="file"
                accept=".pdf,.md,.markdown,.txt"
                aria-label="Document file"
                hidden
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) void upload(file);
                }}
              />
              <button
                className="button secondary small"
                onClick={() => fileInput.current?.click()}
                disabled={Boolean(uploading)}
              >
                {uploading ? `Indexing ${uploading}…` : '+ Upload a document'}
              </button>
            </section>
          )}
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
                    <span>
                      {plainText(item.last_message_preview?.replace(LOCAL_PREFIX, '') ?? '') ||
                        item.thread_id.slice(0, 12)}
                    </span>
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
