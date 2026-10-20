import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { Documents } from '../Documents';
import { ApiClient } from '../api';

const doc = {
  id: 'doc1',
  name: 'report.pdf',
  status: 'ready',
  reason: 'extracted',
  digest: 'a'.repeat(64),
  version: 'v1',
};
const region = {
  region_id: 'b'.repeat(64),
  kind: 'table_cell',
  text: '20',
  row: 1,
  column: 1,
  box: { x0: 0.1, y0: 0.2, x1: 0.4, y1: 0.3 },
};
function apiFixture() {
  const api = new ApiClient('authored-session');
  const saved = {
    id: 'result1',
    payload: {
      digest: 'c'.repeat(64),
      request: { kind: 'question', question: 'Revenue?' },
      receipt: {
        status: 'evidence_found',
        reason: 'literal_evidence',
        answer: '<script>literal source</script>',
        citations: [{ page: 1, region_id: region.region_id, quote: '20' }],
      },
    },
  };
  const request = vi.spyOn(api, 'request').mockImplementation(async (path, options) => {
    if (path === '/workbench/documents') return { documents: [doc] };
    if (path.endsWith('/pages/1')) return { image: 'data:image/png;base64,AAAA' };
    if (path.endsWith('/queries')) return { id: 'result1' };
    if (path.endsWith('/reviews')) return { recorded: true };
    if (path.endsWith('/doc1')) {
      if (options?.method === 'DELETE') return { deleted: true };
      return {
        document: doc,
        manifest: {
          fingerprint: 'd'.repeat(64),
          extractor: 'authored-fixture',
          ocr_backend: 'none',
          pages: [
            { page: 1, status: 'ready', method: 'native', reason: 'extracted', regions: [region] },
          ],
        },
        results: [saved],
      };
    }
    throw Error('Unexpected test endpoint');
  });
  const upload = vi.spyOn(api, 'uploadDocument').mockResolvedValue({ id: 'doc1' });
  return { api, request, upload };
}

describe('Document Workbench authored UI contracts', () => {
  it('does not read private records when disabled', () => {
    const { api, request } = apiFixture();
    render(<Documents api={api} enabled={false} />);
    expect(screen.getByText(/Enable the isolated document runtime/)).toBeTruthy();
    expect(request).not.toHaveBeenCalled();
  });

  it('reads sources, highlights literal citations and binds review to the exact result', async () => {
    const { api, request, upload } = apiFixture();
    const user = userEvent.setup();
    render(<Documents api={api} enabled />);
    await screen.findByRole('option', { name: /report.pdf.*ready/ });
    await user.selectOptions(screen.getByLabelText('Private library'), 'doc1');
    await screen.findByText('<script>literal source</script>');
    expect(document.querySelector('script')).toBeNull();
    await user.click(screen.getByRole('button', { name: /Page 1: 20/ }));
    await screen.findByLabelText('Selected evidence region');
    await user.click(screen.getByText('Review this exact result'));
    const save = screen.getByRole('button', { name: 'Record immutable review' });
    expect((save as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole('checkbox', { name: /I inspected the source/ }));
    await user.click(save);
    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('/workbench/documents/doc1/results/result1/reviews', {
        method: 'POST',
        body: { digest: 'c'.repeat(64), decision: 'needs_review', note: '' },
      }),
    );
    expect(upload).not.toHaveBeenCalled();
    expect(request.mock.calls.every(([path]) => !String(path).includes('/stream'))).toBe(true);
  });

  it('requires an explicit upload action and sends OCR off by default', async () => {
    const { api, upload } = apiFixture();
    const user = userEvent.setup();
    render(<Documents api={api} enabled />);
    const file = new File(['%PDF-test'], 'report.pdf', { type: 'application/pdf' });
    await user.upload(screen.getByLabelText('Source document'), file);
    expect(upload).not.toHaveBeenCalled();
    await user.click(screen.getByRole('button', { name: 'Upload and extract' }));
    await waitFor(() => expect(upload).toHaveBeenCalledWith(file, false));
  });

  it('submits selected-cell arithmetic without a shell or model plan', async () => {
    const { api, request } = apiFixture();
    const user = userEvent.setup();
    render(<Documents api={api} enabled />);
    await screen.findByRole('option', { name: /report.pdf.*ready/ });
    await user.selectOptions(screen.getByLabelText('Private library'), 'doc1');
    await screen.findByText('<script>literal source</script>');
    await user.click(screen.getByText(/Extraction and regions/));
    await user.click(screen.getByLabelText('Select cell 20'));
    await user.click(screen.getByText(/Calculate from selected table cells/));
    await user.click(screen.getByRole('button', { name: 'Calculate selected cells' }));
    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('/workbench/documents/doc1/queries', {
        method: 'POST',
        body: {
          kind: 'calculation',
          document_fingerprint: 'd'.repeat(64),
          operation: 'sum',
          region_ids: [region.region_id],
        },
      }),
    );
  });

  it('navigates a collection citation to another version and preserves its page highlight', async () => {
    const { api, request } = apiFixture();
    const previous = request.getMockImplementation()!;
    const other = { ...doc, id: 'doc2', name: 'other.pdf', manifest_fingerprint: 'e'.repeat(64) };
    request.mockImplementation(async (path, options) => {
      if (path === '/workbench/documents') return { documents: [doc, other] };
      if (path === '/workbench/documents/reconciliations')
        return { sets: [{ id: 'set1', name: 'Saved check' }] };
      if (path === '/workbench/documents/reconciliations/set1')
        return {
          id: 'set1',
          name: 'Saved check',
          scope_label: 'Atlas | Q1 | raw counts',
          collection_fingerprint: 'f'.repeat(64),
          sources: [other],
          catalog: [],
          results: [
            {
              id: 'fact1',
              payload: {
                kind: 'fact',
                digest: 'a'.repeat(64),
                receipt: {
                  status: 'conflict',
                  value: null,
                  observations: [
                    {
                      source_id: 'doc2',
                      status: 'observed',
                      value: '20',
                      citations: [
                        { page: 2, region_id: region.region_id, quote: '20 in other source' },
                      ],
                    },
                  ],
                },
              },
              review: null,
              attention_reason: 'unresolved_evidence',
            },
          ],
        };
      if (path === '/workbench/documents/doc2')
        return {
          document: other,
          evidence_set_count: 1,
          manifest: {
            fingerprint: 'e'.repeat(64),
            extractor: 'authored',
            ocr_backend: 'none',
            pages: [
              { page: 1, status: 'ready', reason: 'extracted', method: 'native', regions: [] },
              {
                page: 2,
                status: 'ready',
                reason: 'extracted',
                method: 'native',
                regions: [region],
              },
            ],
          },
          results: [],
        };
      if (path === '/workbench/documents/doc2/pages/2')
        return { image: 'data:image/png;base64,AAAA' };
      return previous(path, options);
    });
    const user = userEvent.setup();
    render(<Documents api={api} enabled />);
    await screen.findByRole('option', { name: /report.pdf.*ready/ });
    await user.selectOptions(screen.getByLabelText('Private library'), 'doc1');
    await screen.findByText('<script>literal source</script>');
    await user.click(screen.getByRole('button', { name: 'Open reconciliation workspace' }));
    await screen.findByRole('option', { name: 'Saved check' });
    await user.selectOptions(screen.getByLabelText('Saved evidence sets'), 'set1');
    await user.click(await screen.findByRole('button', { name: 'Page 2: 20 in other source' }));
    await screen.findByLabelText('Selected evidence region');
    expect((screen.getByLabelText('Private library') as HTMLSelectElement).value).toBe('doc2');
    expect((screen.getByLabelText('Source page') as HTMLSelectElement).value).toBe('2');
    await screen.findByAltText('Sanitized source preview, page 2');
    expect(screen.getByRole('heading', { name: 'Saved check' })).toBeTruthy();
    await user.click(screen.getByRole('checkbox', { name: /Confirm deletion of this document/ }));
    expect(
      (screen.getByRole('button', { name: 'Delete document' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});
