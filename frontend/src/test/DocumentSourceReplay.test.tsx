import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../api';
import { DocumentSourceReplay } from '../DocumentSourceReplay';

function fixture(complete = true) {
  const api = new ApiClient('authored-session');
  const request = vi.spyOn(api, 'replayOriginal').mockResolvedValue({
    source_id: 'doc1',
    audit_checkpoint_fingerprint: 'a'.repeat(64),
    status: 'exact_snapshot_match',
    original_bytes_match: true,
    profile_matches: true,
    snapshot_matches: true,
    extraction_reexecuted: true,
    numeric_checks_reproduced: 3,
    numeric_counts: { supported: 2, refuted: 1, unresolved: 0 },
    runtime_image_id: null,
    elapsed_seconds: 1.2,
  });
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
  const view = render(<DocumentSourceReplay {...props} />);
  return { request, props, ...view, user: userEvent.setup() };
}
async function confirm(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByText('Replay against original source bytes'));
  await view.user.click(screen.getByRole('checkbox', { name: /I confirm local re-extraction/ }));
}

describe('Source-backed replay', () => {
  it('discards a response arriving after its checkpoint changed', async () => {
    const view = fixture();
    let resolve!: (value: unknown) => void;
    view.request.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await confirm(view);
    await view.user.click(screen.getByRole('button', { name: 'Re-extract first.pdf' }));
    const previousSignal = view.request.mock.calls[0][5] as AbortSignal;
    view.rerender(<DocumentSourceReplay {...view.props} checkpoint={'b'.repeat(64)} />);
    expect(previousSignal.aborted).toBe(true);
    resolve({
      source_id: 'doc1',
      audit_checkpoint_fingerprint: 'a'.repeat(64),
      status: 'exact_snapshot_match',
    });
    await Promise.resolve();
    expect(screen.queryByText('exact_snapshot_match')).toBeNull();
    expect(screen.getByText(/Sources checked this visit: 0\/2/)).toBeTruthy();
  });
  it('does no work on mount and requires a completed audit', () => {
    const view = fixture(false);
    expect(view.request).not.toHaveBeenCalled();
    expect(screen.getByText(/Complete the audit before replaying/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Re-extract first.pdf' })).toBeNull();
  });
  it('requires explicit confirmation and re-extracts one source without enabling OCR or retries', async () => {
    const view = fixture();
    await view.user.click(screen.getByText('Replay against original source bytes'));
    const button = screen.getByRole('button', { name: 'Re-extract first.pdf' });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(screen.getByRole('checkbox', { name: /I confirm local re-extraction/ }));
    await view.user.click(button);
    await screen.findByText('exact_snapshot_match');
    expect(view.request).toHaveBeenCalledExactlyOnceWith(
      'set1',
      'audit1',
      'doc1',
      'a'.repeat(64),
      false,
      expect.any(AbortSignal),
    );
    expect(screen.getByText(/Sources checked this visit: 1\/2/)).toBeTruthy();
    expect(screen.getByText(/refuted 1/)).toBeTruthy();
    expect(screen.getByText('Unchecked this visit.')).toBeTruthy();
    expect(screen.getByText(/Same parser, not an independent extractor/)).toBeTruthy();
    view.rerender(<DocumentSourceReplay {...view.props} checkpoint={'b'.repeat(64)} />);
    await waitFor(() => expect(screen.queryByText('exact_snapshot_match')).toBeNull());
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });
  it('requires separate local OCR consent and labels failures without inventing a result', async () => {
    const view = fixture();
    view.request.mockRejectedValue(Error('Authored unavailable runtime'));
    await confirm(view);
    await view.user.click(screen.getByRole('checkbox', { name: /Allow local OCR only/ }));
    await view.user.click(screen.getByRole('button', { name: 'Re-extract first.pdf' }));
    await screen.findByText(/No automatic retry was made/);
    expect(view.request).toHaveBeenCalledExactlyOnceWith(
      'set1',
      'audit1',
      'doc1',
      'a'.repeat(64),
      true,
      expect.any(AbortSignal),
    );
    expect(screen.queryByText('exact_snapshot_match')).toBeNull();
    expect(screen.getByText(/Sources checked this visit: 0\/2/)).toBeTruthy();
  });
  it('aborts on unmount and cannot accept a late response', async () => {
    const view = fixture();
    let resolve!: (value: unknown) => void;
    view.request.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await confirm(view);
    await view.user.click(screen.getByRole('button', { name: 'Re-extract first.pdf' }));
    view.unmount();
    expect((view.request.mock.calls[0][5] as AbortSignal).aborted).toBe(true);
    resolve({ source_id: 'doc1', audit_checkpoint_fingerprint: 'a'.repeat(64) });
    await Promise.resolve();
    expect(view.request).toHaveBeenCalledOnce();
  });
});
