import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { DocumentClaimReplay } from '../DocumentClaimReplay';
import { ApiClient } from '../api';

const base = '/workbench/documents/reconciliations/source-set/audits/audit1/replays';
function fixture(complete = true) {
  const api = new ApiClient('authored-session');
  let created = false,
    advanced = false;
  const request = vi.spyOn(api, 'request').mockImplementation(async (path, options) => {
    if (path === '/workbench/documents/reconciliations')
      return {
        sets: [
          { id: 'source-set', name: 'Original' },
          { id: 'target-set', name: 'Corrected reports' },
          { id: 'another-set', name: 'Other reports' },
        ],
      };
    if (path === base) {
      if (options?.method === 'POST') {
        created = true;
        return { id: 'replay1' };
      }
      return { replays: created ? [{ id: 'replay1', name: 'Corrected report replay' }] : [] };
    }
    if (path.endsWith('/advance')) {
      advanced = true;
      return { id: 'replay1' };
    }
    if (
      path === '/workbench/documents/reconciliations/target-set' ||
      path === '/workbench/documents/reconciliations/another-set'
    )
      return { collection_fingerprint: 'b'.repeat(64), scope_label: 'Atlas | Q1 | raw counts' };
    if (path === base + '/replay1') {
      if (options?.method === 'DELETE') {
        created = false;
        return { deleted: true };
      }
      return {
        id: 'replay1',
        name: 'Corrected report replay',
        target_set_id: 'target-set',
        target_sources: [{ id: 'new-document-row', name: 'corrected.pdf' }],
        checkpoint_fingerprint: advanced ? 'd'.repeat(64) : 'c'.repeat(64),
        processed: advanced ? 1 : 0,
        total: 1,
        complete: advanced,
        all_claims_supported: false,
        support_lost: advanced ? 1 : 0,
        planned_result_upper_bound: 3,
        items: [
          {
            claim: {
              claim_id: 'growth',
              label: '<script>growth claim</script>',
              expected_value: '25',
              expected_unit: '%',
            },
            before: {
              status: 'supported',
              reason: 'exact_value_match',
              observed_value: '25',
              observed_unit: '%',
            },
            after: advanced
              ? {
                  status: 'refuted',
                  reason: 'numeric_value_differs',
                  observed_value: '50',
                  observed_unit: '%',
                }
              : null,
            comparison: advanced
              ? { transition: 'support_lost', observed_value_changed: true }
              : null,
            observations: advanced
              ? [
                  {
                    source_id: 'new-document-row',
                    status: 'observed',
                    citations: [{ page: 2, region_id: 'e'.repeat(64), quote: '150' }],
                  },
                ]
              : [],
          },
        ],
      };
    }
    throw Error('Unexpected authored endpoint');
  });
  const inspect = vi.fn();
  render(
    <DocumentClaimReplay
      api={api}
      setId="source-set"
      auditId="audit1"
      sourceCheckpoint={'a'.repeat(64)}
      complete={complete}
      onInspect={inspect}
    />,
  );
  return { request, inspect, user: userEvent.setup() };
}
async function create(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByRole('button', { name: 'Open audit replays' }));
  await view.user.type(screen.getByLabelText('Replay name'), 'Corrected report replay');
  await screen.findByRole('option', { name: 'Corrected reports' });
  await view.user.selectOptions(screen.getByLabelText('Target frozen evidence set'), 'target-set');
  await screen.findByText('Target declared scope: Atlas | Q1 | raw counts');
  await view.user.click(screen.getByRole('checkbox', { name: /I inspected the target versions/ }));
  await view.user.click(screen.getByRole('button', { name: 'Create audit replay' }));
  await screen.findByRole('heading', { name: 'Corrected report replay' });
}

describe('Change-aware audit replay UI contracts', () => {
  it('does no work on mount and does not replay an incomplete source audit', () => {
    const view = fixture(false);
    expect(screen.getByText(/Complete the source claim audit/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Open audit replays' })).toBeNull();
    expect(view.request).not.toHaveBeenCalled();
  });

  it('requires explicit comparable target confirmation and resets it on target changes', async () => {
    const view = fixture();
    expect(view.request).not.toHaveBeenCalled();
    await view.user.click(screen.getByRole('button', { name: 'Open audit replays' }));
    await screen.findByRole('option', { name: 'Corrected reports' });
    expect(screen.queryByRole('option', { name: 'Original' })).toBeNull();
    await view.user.selectOptions(
      screen.getByLabelText('Target frozen evidence set'),
      'target-set',
    );
    await screen.findByText(/Target declared scope/);
    const check = screen.getByRole('checkbox', { name: /I inspected the target versions/ });
    await view.user.click(check);
    expect((check as HTMLInputElement).checked).toBe(true);
    await view.user.selectOptions(
      screen.getByLabelText('Target frozen evidence set'),
      'another-set',
    );
    await waitFor(() => expect((check as HTMLInputElement).checked).toBe(false));
    expect(view.request.mock.calls.every(([, options]) => !options?.method)).toBe(true);
  });

  it('freezes source and target fingerprints without silently advancing or changing assertions', async () => {
    const view = fixture();
    await create(view);
    expect(view.request).toHaveBeenCalledWith(base, {
      method: 'POST',
      body: {
        name: 'Corrected report replay',
        target_set_id: 'target-set',
        source_checkpoint_fingerprint: 'a'.repeat(64),
        target_collection_fingerprint: 'b'.repeat(64),
        scope_confirmed: true,
      },
    });
    expect(
      view.request.mock.calls.some(
        ([path]) => path.endsWith('/advance') || path.includes('/stream'),
      ),
    ).toBe(false);
    expect(screen.getByText(/Replayed 0\/1/)).toBeTruthy();
    expect(screen.getByText('Partial replay — unchecked claims remain')).toBeTruthy();
    expect(screen.getByText('Unchanged assertion: 25 %')).toBeTruthy();
    expect(document.querySelector('script')).toBeNull();
  });

  it('shows support loss, navigates actual target citations and deletes only the replay report', async () => {
    const view = fixture();
    await create(view);
    await view.user.click(screen.getByRole('button', { name: 'Replay next two claims' }));
    await screen.findByText('Replay complete');
    expect(view.request).toHaveBeenCalledWith(base + '/replay1/advance', {
      method: 'POST',
      body: { checkpoint_fingerprint: 'c'.repeat(64) },
    });
    expect(screen.getByText(/support_lost/)).toBeTruthy();
    expect(screen.getByText('Before: supported · 25 %')).toBeTruthy();
    expect(screen.getByText('After: refuted · 50 %')).toBeTruthy();
    expect(screen.queryByText(/All replayed numeric assertions match/)).toBeNull();
    await view.user.click(screen.getByText('corrected.pdf · observed'));
    await view.user.click(screen.getByRole('button', { name: 'Target page 2: 150' }));
    expect(view.inspect).toHaveBeenCalledWith('new-document-row', 2, 'e'.repeat(64));
    const remove = screen.getByRole('button', { name: 'Delete audit replay' });
    expect((remove as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(
      screen.getByRole('checkbox', { name: /Confirm deletion of this replay report only/ }),
    );
    await view.user.click(remove);
    await waitFor(() =>
      expect(view.request).toHaveBeenCalledWith(base + '/replay1', { method: 'DELETE' }),
    );
    expect(
      view.request.mock.calls.filter(([, options]) => options?.method === 'DELETE'),
    ).toHaveLength(1);
  });
});
