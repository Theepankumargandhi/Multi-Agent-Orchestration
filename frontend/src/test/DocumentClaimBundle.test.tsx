import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { DocumentClaimBundle } from '../DocumentClaimBundle';
import { ApiClient } from '../api';

function fixture(complete = true) {
  const api = new ApiClient('authored-session');
  const request = vi.spyOn(api, 'requestJSONText').mockResolvedValue(
    JSON.stringify({
      protocol: 'agentforge-claim-replay-bundle-v1',
      fingerprint: 'b'.repeat(64),
      private_snapshot: 'uncited text',
    }),
  );
  const props = { api, setId: 'set1', auditId: 'audit1', checkpoint: 'a'.repeat(64), complete };
  const view = render(<DocumentClaimBundle {...props} />);
  return { request, props, ...view, user: userEvent.setup() };
}
function downloads() {
  const create = vi.fn((_blob: Blob) => 'blob:authored-bundle'),
    revoke = vi.fn();
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: create });
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
  return { create, revoke, click };
}
async function consent(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByText('Portable offline replay bundle'));
  await view.user.click(
    screen.getByRole('checkbox', {
      name: /I authorize downloading the full private extraction snapshot/,
    }),
  );
}

describe('Private portable replay export', () => {
  it('downloads the original UTF-8 JSON rather than re-serializing numeric geometry', async () => {
    const view = fixture();
    const events = downloads();
    const raw = `{"protocol":"agentforge-claim-replay-bundle-v1","fingerprint":"${'b'.repeat(64)}","coordinate":-0.0,"label":"café"}`;
    view.request.mockResolvedValue(raw);
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Download private replay bundle' }));
    await waitFor(() => expect(events.create).toHaveBeenCalledOnce());
    const actual = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsText(events.create.mock.calls[0][0]);
    });
    expect(actual).toBe(raw);
  });
  it('does no work on mount and requires completion, not just supported claims', () => {
    const view = fixture(false);
    expect(view.request).not.toHaveBeenCalled();
    expect(screen.getByText(/Complete the audit before exporting/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Download private replay bundle' })).toBeNull();
  });

  it('requires separate full-snapshot consent and binds the expected checkpoint', async () => {
    const view = fixture();
    const events = downloads();
    await view.user.click(screen.getByText('Portable offline replay bundle'));
    const button = screen.getByRole('button', { name: 'Download private replay bundle' });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(
      screen.getByRole('checkbox', {
        name: /I authorize downloading the full private extraction snapshot/,
      }),
    );
    await view.user.click(button);
    await waitFor(() => expect(events.click).toHaveBeenCalledOnce());
    expect(view.request).toHaveBeenCalledWith(
      '/workbench/documents/reconciliations/set1/audits/audit1/replay-bundle',
      {
        method: 'POST',
        signal: expect.any(AbortSignal),
        body: { checkpoint_fingerprint: 'a'.repeat(64), include_extraction_text: true },
      },
    );
    expect(events.create).toHaveBeenCalledOnce();
    expect(events.revoke).toHaveBeenCalledWith('blob:authored-bundle');
    expect(screen.getByText(/Exit 0 means replay verified/)).toBeTruthy();
    expect(screen.getByText(/--expected-fingerprint/)).toBeTruthy();
    view.rerender(<DocumentClaimBundle {...view.props} checkpoint={'c'.repeat(64)} />);
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(true));
    expect(screen.queryByText(/--expected-fingerprint/)).toBeNull();
  });

  it('rejects an over-budget bundle without creating a download', async () => {
    const view = fixture();
    const events = downloads();
    view.request.mockResolvedValue(
      JSON.stringify({
        protocol: 'agentforge-claim-replay-bundle-v1',
        fingerprint: 'b'.repeat(64),
        text: 'x'.repeat(3_000_001),
      }),
    );
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Download private replay bundle' }));
    await screen.findByText(/exceeds its 3 MB budget/);
    expect(events.create).not.toHaveBeenCalled();
  });

  it('does not download a response arriving after the component is unmounted', async () => {
    const view = fixture();
    const events = downloads();
    let resolve!: (value: string) => void;
    view.request.mockImplementation(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    await consent(view);
    await view.user.click(screen.getByRole('button', { name: 'Download private replay bundle' }));
    view.unmount();
    resolve(
      JSON.stringify({
        protocol: 'agentforge-claim-replay-bundle-v1',
        fingerprint: 'b'.repeat(64),
      }),
    );
    await Promise.resolve();
    expect(events.create).not.toHaveBeenCalled();
    expect((view.request.mock.calls[0][1]?.signal as AbortSignal).aborted).toBe(true);
  });
});
