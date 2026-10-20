import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../api';
import { DocumentCellConcordance } from '../DocumentCellConcordance';

const result = {
  source_id: 'doc1',
  audit_checkpoint_fingerprint: 'a'.repeat(64),
  status: 'all_cells_agree',
  original_bytes_match: true,
  probe_executed: true,
  engine: 'tesseract-tsv-psm6:eng',
  requested_cells: 3,
  counts: { agree: 3, different: 0, uncheckable: 0 },
  raster_pages: 1,
  ocr_calls: 3,
  cells: [
    {
      region_id: 'c'.repeat(64),
      page: 2,
      status: 'agree',
      probe_status: 'observed',
      min_confidence: 0.95,
    },
  ],
};
function fixture(complete = true) {
  const api = new ApiClient('authored');
  const request = vi.spyOn(api, 'crosscheckPixels').mockResolvedValue(result);
  const native = vi.spyOn(api, 'crosscheckCells');
  const props = {
    api,
    setId: 'set1',
    auditId: 'audit1',
    checkpoint: 'a'.repeat(64),
    complete,
    sources: [
      { id: 'doc1', name: 'first.pdf' },
      { id: 'doc2', name: 'second.pdf' },
    ],
  };
  return {
    request,
    native,
    props,
    ...render(<DocumentCellConcordance {...props} mode="pixels" />),
    user: userEvent.setup(),
  };
}
async function consent(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByText('Cross-check rendered cells with local OCR'));
  await view.user.click(
    screen.getByRole('checkbox', { name: /I consent to local OCR of rendered cells/ }),
  );
}
describe('Rendered-cell local OCR', () => {
  it('requires a complete audit and does no work on mount', () => {
    const view = fixture(false);
    expect(view.request).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: /Check pixels/ })).toBeNull();
    expect(screen.getByText(/Complete the audit/)).toBeTruthy();
  });
  it('requires separate local OCR consent and checks exactly one selected original', async () => {
    const view = fixture();
    await view.user.click(screen.getByText('Cross-check rendered cells with local OCR'));
    const button = screen.getByRole('button', { name: 'Check pixels in first.pdf' });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(screen.getByRole('checkbox'));
    await view.user.click(button);
    await screen.findByText('all_cells_agree');
    expect(view.request).toHaveBeenCalledExactlyOnceWith(
      'set1',
      'audit1',
      'doc1',
      'a'.repeat(64),
      expect.any(AbortSignal),
    );
    expect(view.native).not.toHaveBeenCalled();
    expect(screen.getByText('Unchecked this visit.')).toBeTruthy();
    expect(screen.getByText(/local OCR calls: 3/)).toBeTruthy();
    expect(screen.getByText(/confidence 0.95/)).toBeTruthy();
    expect(screen.getByText(/not a probability of correctness/)).toBeTruthy();
  });
  it('keeps low-confidence cells uncheckable without promoting support', async () => {
    const view = fixture();
    view.request.mockResolvedValue({
      ...result,
      status: 'inconclusive',
      counts: { agree: 2, different: 0, uncheckable: 1 },
      cells: [
        {
          ...result.cells[0],
          status: 'uncheckable',
          probe_status: 'low_confidence',
          min_confidence: 0.4,
        },
      ],
    });
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Check pixels in first.pdf' }));
    await screen.findByText('inconclusive');
    expect(screen.getByText(/uncheckable 1/)).toBeTruthy();
    expect(screen.getByText(/low_confidence/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: /approve/i })).toBeNull();
  });
  it('does not retry an uncertain failure or invent a report', async () => {
    const view = fixture();
    view.request.mockRejectedValue(Error('Authored worker unavailable'));
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Check pixels in first.pdf' }));
    await screen.findByText(/No automatic retry was made/);
    expect(view.request).toHaveBeenCalledOnce();
    expect(screen.getByText(/Sources checked this visit: 0\/2/)).toBeTruthy();
  });
  it('clears consent and discards a late response when mode changes', async () => {
    const view = fixture();
    let resolve!: (value: unknown) => void;
    view.request.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Check pixels in first.pdf' }));
    const signal = view.request.mock.calls[0][4] as AbortSignal;
    view.rerender(<DocumentCellConcordance {...view.props} mode="text" />);
    expect(signal.aborted).toBe(true);
    resolve(result);
    await waitFor(() => expect(screen.queryByText('all_cells_agree')).toBeNull());
    expect(
      (screen.getByRole('button', { name: 'Cross-check first.pdf' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
  it('aborts viewing on unmount without guaranteeing server cancellation', async () => {
    const view = fixture();
    view.request.mockImplementation(() => new Promise(() => undefined));
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Check pixels in first.pdf' }));
    view.unmount();
    expect((view.request.mock.calls[0][4] as AbortSignal).aborted).toBe(true);
  });
});
