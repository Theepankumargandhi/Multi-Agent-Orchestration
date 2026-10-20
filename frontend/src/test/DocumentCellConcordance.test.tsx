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
  engine: 'pypdfium2:5.14.0',
  requested_cells: 3,
  counts: { agree: 3, different: 0, uncheckable: 0 },
  cells: [{ region_id: 'c'.repeat(64), page: 2, status: 'agree', probe_status: 'observed' }],
};
function fixture(complete = true) {
  const api = new ApiClient('authored');
  const request = vi.spyOn(api, 'crosscheckCells').mockResolvedValue(result);
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
    props,
    ...render(<DocumentCellConcordance {...props} />),
    user: userEvent.setup(),
  };
}
async function consent(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByText('Cross-check cited cells with PDFium'));
  await view.user.click(
    screen.getByRole('checkbox', { name: /I confirm the local PDF text cross-check/ }),
  );
}
describe('PDFium cited-cell checks', () => {
  it('does not run on mount or offer execution for incomplete audits', () => {
    const view = fixture(false);
    expect(view.request).not.toHaveBeenCalled();
    expect(screen.getByText(/Complete the audit before cross-checking/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Cross-check first.pdf' })).toBeNull();
  });
  it('requires confirmation, checks only one source and keeps other sources visibly unchecked', async () => {
    const view = fixture();
    await view.user.click(screen.getByText('Cross-check cited cells with PDFium'));
    const button = screen.getByRole('button', { name: 'Cross-check first.pdf' });
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
    expect(screen.getByText(/Sources checked this visit: 1\/2/)).toBeTruthy();
    expect(screen.getByText('Unchecked this visit.')).toBeTruthy();
    expect(screen.getByText(/Text agreement is not claim support/)).toBeTruthy();
    expect(screen.getByText(/Page 2/)).toBeTruthy();
  });
  it('shows disagreement and uncheckable coverage without approving claims', async () => {
    const view = fixture();
    view.request.mockResolvedValue({
      ...result,
      status: 'disagreement',
      counts: { agree: 1, different: 1, uncheckable: 1 },
    });
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Cross-check first.pdf' }));
    await screen.findByText('disagreement');
    expect(screen.getByText(/Agree 1 · different 1 · uncheckable 1/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: /approve/i })).toBeNull();
  });
  it('does not retry an uncertain error or invent a checked result', async () => {
    const view = fixture();
    view.request.mockRejectedValue(Error('Authored unavailable worker'));
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Cross-check first.pdf' }));
    await screen.findByText(/No automatic retry was made/);
    expect(view.request).toHaveBeenCalledOnce();
    expect(screen.getByText(/Sources checked this visit: 0\/2/)).toBeTruthy();
  });
  it('aborts the old checkpoint and discards its late response', async () => {
    const view = fixture();
    let resolve!: (value: unknown) => void;
    view.request.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Cross-check first.pdf' }));
    const signal = view.request.mock.calls[0][4] as AbortSignal;
    view.rerender(<DocumentCellConcordance {...view.props} checkpoint={'b'.repeat(64)} />);
    expect(signal.aborted).toBe(true);
    resolve(result);
    await waitFor(() => expect(screen.queryByText('all_cells_agree')).toBeNull());
    expect(screen.getByText(/Sources checked this visit: 0\/2/)).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Cross-check first.pdf' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
  it('aborts on unmount without implying server cancellation', async () => {
    const view = fixture();
    view.request.mockImplementation(() => new Promise(() => undefined));
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Cross-check first.pdf' }));
    view.unmount();
    expect((view.request.mock.calls[0][4] as AbortSignal).aborted).toBe(true);
  });
});
