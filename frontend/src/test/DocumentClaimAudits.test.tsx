import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { DocumentClaimAudits } from '../DocumentClaimAudits';
import { ApiClient } from '../api';

const base = '/workbench/documents/reconciliations/set1/audits';
const resultId = 'a'.repeat(32);
const results = [
  {
    id: resultId,
    payload: {
      receipt: { status: 'agreed', query: { row_label: 'Requests', column_label: 'Current' } },
    },
  },
];
function fixture() {
  const api = new ApiClient('authored-session');
  let created = false,
    processed = 0;
  const request = vi.spyOn(api, 'request').mockImplementation(async (path, options) => {
    if (path === base) {
      if (options?.method === 'POST') {
        created = true;
        return { id: 'audit1' };
      }
      return { audits: created ? [{ id: 'audit1', name: 'Quarterly claims' }] : [] };
    }
    if (path.endsWith('/advance')) {
      processed = Math.min(processed + 2, 3);
      return { id: 'audit1' };
    }
    if (path.endsWith('/export'))
      return {
        complete: processed === 3,
        all_claims_supported: false,
        private_evidence: 'Literal source excerpts',
      };
    if (path === base + '/audit1') {
      if (options?.method === 'DELETE') {
        created = false;
        return { deleted: true };
      }
      return {
        id: 'audit1',
        name: 'Quarterly claims',
        total: 3,
        processed,
        complete: processed === 3,
        all_claims_supported: false,
        checkpoint_fingerprint: processed === 0 ? 'b'.repeat(64) : 'c'.repeat(64),
        counts: {
          supported: processed ? 1 : 0,
          refuted: processed ? 1 : 0,
          unresolved: processed === 3 ? 1 : 0,
        },
        items: [0, 1, 2].map((index) => ({
          result_id: resultId,
          claim: {
            claim_id: `claim-${index + 1}`,
            label: index === 0 ? '<script>literal claim label</script>' : `Claim ${index + 1}`,
            expected_value: '125',
            expected_unit: 'scalar',
          },
          evidence_status: 'agreed',
          observations: [
            {
              source_id: 'document-row',
              status: 'observed',
              citations: [{ page: 2, region_id: 'd'.repeat(64), quote: '125' }],
            },
          ],
          check:
            index < processed
              ? {
                  status: ['supported', 'refuted', 'unresolved'][index],
                  reason: 'exact_numeric_only',
                  observed_value: '125',
                  observed_unit: 'scalar',
                }
              : null,
        })),
      };
    }
    throw Error('Unexpected authored endpoint');
  });
  const inspect = vi.fn();
  render(
    <DocumentClaimAudits
      api={api}
      setId="set1"
      fingerprint={'f'.repeat(64)}
      results={results}
      sources={[{ id: 'document-row', name: 'report.pdf' }]}
      onInspect={inspect}
    />,
  );
  return { request, inspect, user: userEvent.setup() };
}
async function create(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByRole('button', { name: 'Open claim audits' }));
  await view.user.type(screen.getByLabelText('Audit name'), 'Quarterly claims');
  await view.user.selectOptions(screen.getByLabelText('Saved evidence for claim 1'), resultId);
  await view.user.type(screen.getByLabelText('Display label for claim 1'), 'Requests are 125');
  await view.user.type(screen.getByLabelText('Expected value for claim 1'), '125.0');
  await view.user.click(screen.getByRole('button', { name: 'Create claim audit' }));
  await screen.findByRole('heading', { name: 'Quarterly claims' });
}

describe('Numeric claim audit UI contracts', () => {
  it('loads no audit or model work until opened, and requires explicit saved evidence mapping', async () => {
    const view = fixture();
    expect(view.request).not.toHaveBeenCalled();
    await view.user.click(screen.getByRole('button', { name: 'Open claim audits' }));
    expect(
      (screen.getByRole('button', { name: 'Create claim audit' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await view.user.type(screen.getByLabelText('Audit name'), 'Quarterly claims');
    await view.user.type(
      screen.getByLabelText('Display label for claim 1'),
      'Not semantically checked',
    );
    await view.user.type(screen.getByLabelText('Expected value for claim 1'), '1e3');
    await view.user.selectOptions(screen.getByLabelText('Saved evidence for claim 1'), resultId);
    expect(
      (screen.getByRole('button', { name: 'Create claim audit' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await waitFor(() => expect(view.request).toHaveBeenCalledTimes(1));
  });

  it('freezes the exact numeric request without starting a step automatically', async () => {
    const view = fixture();
    await create(view);
    expect(view.request).toHaveBeenCalledWith(base, {
      method: 'POST',
      body: {
        name: 'Quarterly claims',
        collection_fingerprint: 'f'.repeat(64),
        claims: [
          {
            result_id: resultId,
            claim: {
              claim_id: 'claim-1',
              label: 'Requests are 125',
              expected_value: '125.0',
              expected_unit: 'scalar',
            },
          },
        ],
      },
    });
    expect(
      view.request.mock.calls.some(
        ([path]) =>
          path.endsWith('/advance') || path.endsWith('/export') || path.includes('/stream'),
      ),
    ).toBe(false);
    expect(screen.getByText(/Checked 0\/3/)).toBeTruthy();
    expect(document.querySelector('script')).toBeNull();
  });

  it('advances explicitly with the checkpoint, labels partial audits and routes source citations', async () => {
    const view = fixture();
    await create(view);
    await view.user.click(screen.getByRole('button', { name: 'Check next two claims' }));
    await screen.findByText(/Checked 2\/3/);
    expect(view.request).toHaveBeenCalledWith(base + '/audit1/advance', {
      method: 'POST',
      body: { checkpoint_fingerprint: 'b'.repeat(64) },
    });
    expect(screen.getByText('Partial — unchecked claims remain')).toBeTruthy();
    expect(screen.getAllByText(/Not all claims are supported/)).toHaveLength(1);
    await view.user.click(screen.getAllByText(/report.pdf · observed/)[0]);
    await view.user.click(screen.getAllByRole('button', { name: 'Page 2: 125' })[0]);
    expect(view.inspect).toHaveBeenCalledWith('document-row', 2, 'd'.repeat(64));
    await view.user.click(screen.getByRole('button', { name: 'Check next two claims' }));
    await screen.findByText('Complete');
    expect(
      (screen.getByRole('button', { name: 'Check next two claims' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(screen.queryByText(/All listed numeric assertions match/)).toBeNull();
  });

  it('requires explicit private export consent and limits deletion to the selected audit', async () => {
    const view = fixture();
    await create(view);
    const makeURL = vi.fn(() => 'blob:authored-audit'),
      revoke = vi.fn();
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: makeURL });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revoke });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined);
    const download = screen.getByRole('button', { name: 'Download private audit JSON' });
    expect((download as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(
      screen.getByRole('checkbox', { name: /I understand the JSON contains private/ }),
    );
    await view.user.click(download);
    await waitFor(() => expect(view.request).toHaveBeenCalledWith(base + '/audit1/export'));
    expect(makeURL).toHaveBeenCalledOnce();
    expect(click).toHaveBeenCalledOnce();
    expect(revoke).toHaveBeenCalledWith('blob:authored-audit');
    const remove = screen.getByRole('button', { name: 'Delete claim audit' });
    expect((remove as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(
      screen.getByRole('checkbox', { name: /Confirm deletion of this audit and dependent replay/ }),
    );
    await view.user.click(remove);
    await waitFor(() =>
      expect(view.request).toHaveBeenCalledWith(base + '/audit1', { method: 'DELETE' }),
    );
    expect(
      view.request.mock.calls.filter(([, options]) => options?.method === 'DELETE'),
    ).toHaveLength(1);
  });
});
