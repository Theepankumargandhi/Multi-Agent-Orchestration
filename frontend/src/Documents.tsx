import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import type { ApiClient } from './api';
import { Badge, Empty, Notice, PageHeading, errorText, isAbort } from './components';
import { DocumentChanges } from './DocumentChanges';
import { DocumentReconciliation } from './DocumentReconciliation';

type Region = {
  region_id: string;
  kind: string;
  text: string;
  box: { x0: number; y0: number; x1: number; y1: number };
  table_id?: string;
  row?: number;
  column?: number;
};
type Page = { page: number; status: string; reason: string; method: string; regions: Region[] };
type Document = {
  id: string;
  name: string;
  status: string;
  reason: string;
  digest: string;
  version: string;
  family_id?: string;
  parent_id?: string | null;
  manifest_fingerprint?: string | null;
};
type Receipt = {
  status: string;
  reason: string;
  answer?: string;
  value?: string;
  unit?: string;
  citations: { page: number; region_id: string; quote: string }[];
};
type Result = {
  id: string;
  payload: {
    digest: string;
    request: { question?: string; kind: string; operation?: string };
    receipt: Receipt;
  };
  review?: { decision: string; owner: string } | null;
};
type Detail = {
  document: Document;
  manifest: { fingerprint: string; extractor: string; ocr_backend: string; pages: Page[] } | null;
  results: Result[];
  versions?: Document[];
  evidence_set_count?: number;
};

export function Documents({ api, enabled }: { api: ApiClient; enabled: boolean }) {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selected, setSelected] = useState('');
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const [detail, setDetail] = useState<Detail | null>(null);
  const [page, setPage] = useState(1);
  const [image, setImage] = useState('');
  const [highlight, setHighlight] = useState('');
  const [cells, setCells] = useState<string[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [ocr, setOCR] = useState(false);
  const [parent, setParent] = useState('');
  const [question, setQuestion] = useState('');
  const [operation, setOperation] = useState('sum');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [deleting, setDeleting] = useState(false);
  const [citation, setCitation] = useState<{
    documentId: string;
    page: number;
    regionId: string;
  } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    void api
      .request<{ documents: Document[] }>('/workbench/documents', { signal: controller.signal })
      .then((value) => setDocuments(value.documents))
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, enabled, refresh]);
  useEffect(() => {
    setDetail(null);
    setPage(1);
    setImage('');
    setCells([]);
    setHighlight('');
    setDeleting(false);
    if (!selected) return;
    const controller = new AbortController();
    void api
      .request<Detail>(`/workbench/documents/${selected}`, { signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setDetail(value);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, selected, refresh]);
  useEffect(() => {
    if (!citation || detail?.document.id !== citation.documentId) return;
    const sourcePage = detail.manifest?.pages.find((p) => p.page === citation.page);
    if (sourcePage?.regions.some((r) => r.region_id === citation.regionId)) {
      setPage(citation.page);
      setHighlight(citation.regionId);
      document
        .getElementById('document-source')
        ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else setError('The selected citation is unavailable in this frozen source.');
    setCitation(null);
  }, [citation, detail]);
  useEffect(() => {
    setImage('');
    if (!selected || !detail?.manifest) return;
    const controller = new AbortController();
    void api
      .request<{ image: string }>(`/workbench/documents/${selected}/pages/${page}`, {
        signal: controller.signal,
      })
      .then((value) => {
        if (
          !controller.signal.aborted &&
          /^data:image\/png;base64,[A-Za-z0-9+/=]+$/.test(value.image)
        )
          setImage(value.image);
      })
      .catch((e) => {
        if (!isAbort(e)) setError(errorText(e));
      });
    return () => controller.abort();
  }, [api, selected, detail, page]);
  async function act(action: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await action();
      setRefresh((value) => value + 1);
    } catch (e) {
      if (!isAbort(e))
        setError(`${errorText(e)} Refresh the library before retrying an uncertain upload.`);
    } finally {
      setBusy(false);
    }
  }
  async function upload(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    await act(async () => {
      const result = parent
        ? await api.uploadDocument(file, ocr, parent)
        : await api.uploadDocument(file, ocr);
      setSelected(result.id);
    });
  }
  async function query(kind: 'question' | 'calculation') {
    if (!detail?.manifest) return;
    await act(() =>
      api.request(`/workbench/documents/${selected}/queries`, {
        method: 'POST',
        body: {
          kind,
          document_fingerprint: detail.manifest!.fingerprint,
          ...(kind === 'question' ? { question } : { operation, region_ids: cells }),
        },
      }),
    );
  }
  const current = detail?.manifest?.pages.find((p) => p.page === page);
  const region = current?.regions.find((r) => r.region_id === highlight);
  const hasChildVersions =
    detail?.versions?.some((version) => version.parent_id === selected) ?? false;
  if (!enabled)
    return (
      <>
        <PageHeading eyebrow="DOCUMENT EVIDENCE" title="Document Workbench">
          Local document processing is disabled.
        </PageHeading>
        <section className="panel">
          <Empty title="Enable the isolated document runtime first">
            Build docker/Dockerfile.document-workbench, then set WORKBENCH_DOCUMENTS_ENABLED=true
            and restart the backend. This screen does not call a model.
          </Empty>
        </section>
      </>
    );
  return (
    <>
      <PageHeading
        eyebrow="DOCUMENT EVIDENCE / 0.11"
        title="Read the source. Check the claim."
        actions={<Badge tone="green">Local · no model calls</Badge>}
      >
        Private PDF and image evidence, literal retrieval, explicit table calculations and
        version-bound owner review.
      </PageHeading>
      <Notice>
        Lineage verification checks where evidence came from—not factual truth or answer relevance.
        Any held page blocks document-wide answers. Reviews are your attestation, not independent
        approval.
      </Notice>
      {error && <Notice error>{error}</Notice>}
      <div className="document-library panel">
        <form onSubmit={upload} className="document-upload">
          <label>
            Source document (PDF, PNG or JPEG; 3 MB, 10 pages maximum)
            <input
              aria-label="Source document"
              type="file"
              accept=".pdf,.png,.jpg,.jpeg"
              disabled={busy}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            New version of (optional)
            <select
              aria-label="New version of"
              value={parent}
              disabled={busy}
              onChange={(e) => setParent(e.target.value)}
            >
              <option value="">Independent document</option>
              {documents
                .filter((doc) => ['ready', 'held'].includes(doc.status))
                .map((doc) => (
                  <option value={doc.id} key={doc.id}>
                    {doc.name} · {doc.digest.slice(0, 8)}
                  </option>
                ))}
            </select>
          </label>
          <label className="check-label">
            <input
              type="checkbox"
              checked={ocr}
              disabled={busy}
              onChange={(e) => setOCR(e.target.checked)}
            />{' '}
            Enable local English OCR for images/scans
          </label>
          <button className="button" disabled={busy || !file}>
            {busy ? 'Processing…' : 'Upload and extract'}
          </button>
        </form>
        <div className="button-row">
          <label>
            Private library
            <select
              aria-label="Private library"
              value={selected}
              disabled={busy}
              onChange={(e) => setSelected(e.target.value)}
            >
              <option value="">Choose a document</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.name} · {doc.status} · {doc.digest.slice(0, 8)}
                </option>
              ))}
            </select>
          </label>
          <button
            className="button secondary"
            disabled={busy}
            onClick={() => setRefresh((value) => value + 1)}
          >
            Refresh library
          </button>
        </div>
        <p className="muted small-text">
          Immutable uploads; choose a parent explicitly to link versions. Up to 10
          versions/documents per owner. Processing may take 90 seconds. Navigating away does not
          cancel server extraction.
        </p>
      </div>
      <DocumentReconciliation
        api={api}
        documents={documents}
        onInspect={(documentId, page, regionId) => {
          setCitation({ documentId, page, regionId });
          setSelected(documentId);
        }}
        onUpdated={() => {
          void api
            .request<{ documents: Document[] }>('/workbench/documents')
            .then((value) => setDocuments(value.documents))
            .catch((e) => {
              if (!isAbort(e)) setError(errorText(e));
            });
          const id = selectedRef.current;
          if (id)
            void api
              .request<Detail>(`/workbench/documents/${id}`)
              .then((value) => {
                if (selectedRef.current === id) setDetail(value);
              })
              .catch((e) => {
                if (!isAbort(e)) setError(errorText(e));
              });
        }}
      />
      {detail && (
        <>
          <section className="panel document-meta">
            <div>
              <h2>{detail.document.name}</h2>
              <Badge>{detail.document.status}</Badge> <span>{detail.document.reason}</span>
              <p className="mono small-text">SHA-256 {detail.document.digest}</p>
            </div>
            <label className="check-label">
              <input
                type="checkbox"
                checked={deleting}
                onChange={(e) => setDeleting(e.target.checked)}
                disabled={busy}
              />{' '}
              Confirm deletion of this document and its results
            </label>
            <button
              className="button secondary"
              disabled={
                busy ||
                !deleting ||
                detail.document.status === 'processing' ||
                hasChildVersions ||
                !!detail.evidence_set_count
              }
              onClick={() =>
                void act(async () => {
                  await api.request(`/workbench/documents/${selected}`, { method: 'DELETE' });
                  setSelected('');
                })
              }
            >
              Delete document
            </button>
            {hasChildVersions && (
              <p className="muted small-text">
                This version has children. If deletion is intended, remove child-version leaves
                first.
              </p>
            )}
            {!!detail.evidence_set_count && (
              <p className="muted small-text">
                This version belongs to {detail.evidence_set_count} frozen evidence set(s). Delete
                those sets before deleting the source.
              </p>
            )}
          </section>
          {detail.manifest && (
            <DocumentChanges
              key={selected}
              api={api}
              target={detail.document}
              versions={detail.versions ?? [detail.document]}
              fingerprint={detail.manifest.fingerprint}
              onUpdated={async () => {
                const id = selected;
                const value = await api.request<Detail>(`/workbench/documents/${id}`);
                if (selectedRef.current === id) setDetail(value);
              }}
              onOpenResult={(id) =>
                document
                  .getElementById('document-result-' + id)
                  ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
              }
            />
          )}
          {detail.manifest && (
            <div className="document-grid">
              <section className="panel document-source" id="document-source">
                <div className="section-title">
                  <h2>Source inspection</h2>
                  <select
                    aria-label="Source page"
                    value={page}
                    onChange={(e) => {
                      setPage(Number(e.target.value));
                      setHighlight('');
                    }}
                  >
                    {detail.manifest.pages.map((p) => (
                      <option key={p.page} value={p.page}>
                        Page {p.page} · {p.method} · {p.status}
                      </option>
                    ))}
                  </select>
                </div>
                <p>
                  {current?.reason}. Highlight boxes enclose the entire extracted region, not just
                  the quoted words.
                </p>
                <div className="document-raster">
                  {image ? (
                    <>
                      <img src={image} alt={`Sanitized source preview, page ${page}`} />
                      {region && current?.status === 'ready' && (
                        <svg
                          className="document-overlay"
                          viewBox="0 0 1 1"
                          preserveAspectRatio="none"
                          aria-label="Selected evidence region"
                        >
                          <rect
                            x={region.box.x0}
                            y={region.box.y0}
                            width={region.box.x1 - region.box.x0}
                            height={region.box.y1 - region.box.y0}
                          />
                        </svg>
                      )}
                    </>
                  ) : (
                    <p className="muted">Preview unavailable or loading.</p>
                  )}
                </div>
                <details>
                  <summary>Extraction and regions ({current?.regions.length ?? 0})</summary>
                  <p className="small-text mono">
                    {detail.manifest.extractor}
                    <br />
                    {detail.manifest.ocr_backend}
                  </p>
                  {current?.regions.map((r) => (
                    <div className="document-region" key={r.region_id}>
                      <button className="text-button" onClick={() => setHighlight(r.region_id)}>
                        {r.text}
                      </button>
                      {r.kind === 'table_cell' && (
                        <label className="check-label">
                          <input
                            type="checkbox"
                            aria-label={`Select cell ${r.text}`}
                            checked={cells.includes(r.region_id)}
                            disabled={busy || (!cells.includes(r.region_id) && cells.length >= 16)}
                            onChange={(e) =>
                              setCells((value) =>
                                e.target.checked
                                  ? [...value, r.region_id]
                                  : value.filter((id) => id !== r.region_id),
                              )
                            }
                          />{' '}
                          Cell row {r.row}, column {r.column}
                        </label>
                      )}
                    </div>
                  ))}
                </details>
              </section>
              <section className="panel document-results">
                <h2>Evidence questions</h2>
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void query('question');
                  }}
                >
                  <label>
                    Question
                    <textarea
                      value={question}
                      maxLength={2000}
                      required
                      onChange={(e) => setQuestion(e.target.value)}
                      placeholder="Find the source passage that answers your question"
                    />
                  </label>
                  <button className="button" disabled={busy || !question.trim()}>
                    Retrieve literal evidence
                  </button>
                </form>
                <details>
                  <summary>Calculate from selected table cells ({cells.length})</summary>
                  <p>
                    Selection order is operand order. Difference: first minus second; percent
                    change: old then new. OCR numbers are not calculation-ready.
                  </p>
                  <ol>
                    {cells.map((id) => (
                      <li key={id}>
                        {
                          detail.manifest?.pages
                            .flatMap((p) => p.regions)
                            .find((r) => r.region_id === id)?.text
                        }
                      </li>
                    ))}
                  </ol>
                  <label>
                    Operation
                    <select value={operation} onChange={(e) => setOperation(e.target.value)}>
                      <option value="sum">Sum</option>
                      <option value="difference">Difference</option>
                      <option value="ratio">Ratio</option>
                      <option value="percent_change">Percent change</option>
                    </select>
                  </label>
                  <div className="button-row">
                    <button
                      className="button secondary"
                      disabled={
                        busy || !cells.length || (operation !== 'sum' && cells.length !== 2)
                      }
                      onClick={() => void query('calculation')}
                    >
                      Calculate selected cells
                    </button>
                    <button className="text-button" disabled={busy} onClick={() => setCells([])}>
                      Clear selection
                    </button>
                  </div>
                </details>
                <h3>Saved results · {detail.results.length}/20</h3>
                {detail.results.length === 0 && (
                  <p className="muted">No result yet. Uploading and reading do not call a model.</p>
                )}
                {detail.results.map((result) => (
                  <article
                    key={result.id}
                    id={'document-result-' + result.id}
                    className="document-result"
                  >
                    <strong>
                      {result.payload.request.question || result.payload.request.operation}
                    </strong>
                    <p>
                      <Badge>{result.payload.receipt.status}</Badge> {result.payload.receipt.reason}
                    </p>
                    <p className="preserve-text">
                      {result.payload.receipt.answer ??
                        (result.payload.receipt.value != null
                          ? `${result.payload.receipt.value} ${result.payload.receipt.unit ?? ''}`
                          : 'No validated calculation available.')}
                    </p>
                    <p className="small-text muted">
                      SDK lineage verified · 0 model calls · not a truth guarantee
                    </p>
                    {result.payload.receipt.citations.map((citation, i) => (
                      <button
                        className="citation-button"
                        key={i}
                        onClick={() => {
                          setPage(citation.page);
                          setHighlight(citation.region_id);
                        }}
                      >
                        [{i + 1}] Page {citation.page}: {citation.quote}
                      </button>
                    ))}
                    <ResultReview
                      result={result}
                      busy={busy}
                      submit={(decision, note) =>
                        act(() =>
                          api.request(
                            `/workbench/documents/${selected}/results/${result.id}/reviews`,
                            {
                              method: 'POST',
                              body: { digest: result.payload.digest, decision, note },
                            },
                          ),
                        )
                      }
                    />
                  </article>
                ))}
              </section>
            </div>
          )}
        </>
      )}
    </>
  );
}

function ResultReview({
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
        Owner review: {result.review.decision} · {result.review.owner}. Immutable, bound to this
        result and document version.
      </p>
    );
  return (
    <details>
      <summary>Review this exact result</summary>
      <label>
        Decision
        <select value={decision} onChange={(e) => setDecision(e.target.value)}>
          <option value="needs_review">Needs review</option>
          <option value="supported">Supported by the inspected source</option>
          <option value="unsupported">Unsupported or incorrect</option>
        </select>
      </label>
      <label>
        Review note
        <textarea value={note} maxLength={1000} onChange={(e) => setNote(e.target.value)} />
      </label>
      <label className="check-label">
        <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} /> I
        inspected the source. This records my review, not an independent approval.
      </label>
      <button
        className="button secondary"
        disabled={busy || !checked}
        onClick={() => void submit(decision, note)}
      >
        Record immutable review
      </button>
    </details>
  );
}
