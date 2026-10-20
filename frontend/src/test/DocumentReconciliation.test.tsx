import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { DocumentReconciliation } from '../DocumentReconciliation';
import { ApiClient } from '../api';

const base = '/workbench/documents/reconciliations';
const sources = ['a', 'b'].map((letter, i) => ({
  id: letter.repeat(32),
  name: `report-${i}.pdf`,
  status: 'ready',
  digest: letter.repeat(64),
  manifest_fingerprint: String(i + 1).repeat(64),
}));
const context = 'f'.repeat(64);
const beforeId = 'c'.repeat(32),
  conflictId = 'd'.repeat(32);
function fixture() {
  const api = new ApiClient('authored-session');
  let created = false;
  let reviewed = false;
  let calculated = false;
  const receipt = (column: string, conflict: boolean) => ({
    status: conflict ? 'conflict' : 'agreed',
    value: conflict ? null : '100',
    unit: conflict ? null : 'scalar',
    query: { row_label: 'Requests', column_label: column },
    reasons: conflict ? ['numeric_values_disagree'] : ['all_observations_agree'],
    distinct_artifacts: 2,
    observations: sources.map((source, index) => ({
      source_id: source.id,
      status: 'observed',
      value: conflict && index === 1 ? '150' : '100',
      citations: [
        { page: 2, region_id: 'e'.repeat(64), quote: '<script>literal numeric source</script>' },
      ],
    })),
  });
  const request = vi.spyOn(api, 'request').mockImplementation(async (path, options) => {
    if (path === base) {
      if (options?.method === 'POST') {
        created = true;
        return { id: 'set1' };
      }
      return { sets: created ? [{ id: 'set1', name: 'Atlas reconciliation' }] : [] };
    }
    if (path.endsWith('/calculations')) {
      calculated = true;
      return { id: 'calculation1' };
    }
    if (path.endsWith('/reviews')) {
      reviewed = true;
      return { recorded: true };
    }
    if (path === base + '/set1') {
      if (options?.method === 'DELETE') {
        created = false;
        return { deleted: true };
      }
      return {
        id: 'set1',
        name: 'Atlas reconciliation',
        scope_label: 'Atlas | 2026-Q1 | raw counts',
        collection_fingerprint: context,
        sources,
        catalog: [{ row_label: 'Requests', column_label: 'Current' }],
        catalog_truncated: false,
        results: [
          {
            id: beforeId,
            payload: { digest: '1'.repeat(64), kind: 'fact', receipt: receipt('Previous', false) },
            review: null,
            attention_reason: 'unreviewed',
          },
          {
            id: conflictId,
            payload: { digest: '2'.repeat(64), kind: 'fact', receipt: receipt('Current', true) },
            review: reviewed ? { decision: 'inspected', owner: 'alice', note: '' } : null,
            attention_reason: 'unresolved_evidence',
          },
          ...(calculated
            ? [
                {
                  id: 'calculation1',
                  payload: {
                    digest: '3'.repeat(64),
                    kind: 'calculation',
                    receipt: {
                      status: 'abstained',
                      reason: 'unresolved_fact',
                      operation: 'percent_change',
                      value: null,
                    },
                  },
                  review: null,
                  attention_reason: 'unresolved_evidence',
                },
              ]
            : []),
        ],
      };
    }
    if (path.endsWith('/facts')) return { id: conflictId };
    throw Error('Unexpected authored endpoint');
  });
  const inspect = vi.fn(),
    updated = vi.fn();
  render(
    <DocumentReconciliation
      api={api}
      documents={sources}
      onInspect={inspect}
      onUpdated={updated}
    />,
  );
  return { request, inspect, updated, user: userEvent.setup() };
}

async function create(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByRole('button', { name: 'Open reconciliation workspace' }));
  await view.user.type(screen.getByLabelText('Evidence set name'), 'Atlas reconciliation');
  await view.user.type(screen.getByLabelText('Entity'), 'Atlas');
  await view.user.type(screen.getByLabelText('Reporting period'), '2026-Q1');
  await view.user.type(screen.getByLabelText('Metric definition, units and scale'), 'raw counts');
  for (const source of sources)
    await view.user.click(screen.getByRole('checkbox', { name: new RegExp(source.name) }));
  await view.user.click(screen.getByRole('checkbox', { name: /I confirm these exact source/ }));
  await view.user.click(screen.getByRole('button', { name: 'Create evidence set' }));
  await screen.findByRole('heading', { name: 'Atlas reconciliation' });
}

describe('Frozen reconciliation UI contracts (authored, no providers)', () => {
  it('does no work on mount and scope edits invalidate explicit confirmation', async () => {
    const view = fixture();
    expect(view.request).not.toHaveBeenCalled();
    await view.user.click(screen.getByRole('button', { name: 'Open reconciliation workspace' }));
    await waitFor(() => expect(view.request).toHaveBeenCalledTimes(1));
    const confirmation = screen.getByRole('checkbox', { name: /I confirm these exact source/ });
    await view.user.click(confirmation);
    expect((confirmation as HTMLInputElement).checked).toBe(true);
    await view.user.type(screen.getByLabelText('Entity'), 'Atlas');
    expect((confirmation as HTMLInputElement).checked).toBe(false);
    expect(
      (screen.getByRole('button', { name: 'Create evidence set' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(view.request.mock.calls.every(([, options]) => !options?.method)).toBe(true);
  });

  it('creates explicit snapshot bindings and opens citations using application source IDs', async () => {
    const view = fixture();
    await create(view);
    expect(view.request).toHaveBeenCalledWith(base, {
      method: 'POST',
      body: {
        name: 'Atlas reconciliation',
        scope: { entity: 'Atlas', period: '2026-Q1', basis: 'raw counts' },
        scope_confirmed: true,
        sources: sources.map((s) => ({ document_id: s.id, fingerprint: s.manifest_fingerprint })),
      },
    });
    expect(document.querySelector('script')).toBeNull();
    await view.user.click(screen.getAllByRole('button', { name: /Page 2: <script>/ })[0]);
    expect(view.inspect).toHaveBeenCalledWith(sources[0].id, 2, 'e'.repeat(64));
    expect(view.updated).toHaveBeenCalledTimes(1);
    await view.user.selectOptions(screen.getByLabelText('Discovered table labels'), '0');
    await view.user.click(screen.getByRole('button', { name: 'Check numeric fact' }));
    await waitFor(() =>
      expect(view.request).toHaveBeenCalledWith(base + '/set1/facts', {
        method: 'POST',
        body: { collection_fingerprint: context, row_label: 'Requests', column_label: 'Current' },
      }),
    );
    expect(view.request.mock.calls.every(([path]) => !String(path).includes('/stream'))).toBe(true);
  });

  it('sends ordered saved fact IDs and displays the server abstention without resolving a conflict', async () => {
    const view = fixture();
    await create(view);
    await view.user.click(screen.getByLabelText('Select fact Requests Previous'));
    await view.user.click(screen.getByLabelText('Select fact Requests Current'));
    await view.user.click(screen.getByText(/Calculate from saved facts/));
    await view.user.click(screen.getByRole('button', { name: 'Calculate reconciled facts' }));
    await screen.findByText('unresolved_fact');
    expect(view.request).toHaveBeenCalledWith(base + '/set1/calculations', {
      method: 'POST',
      body: {
        collection_fingerprint: context,
        operation: 'percent_change',
        fact_ids: [beforeId, conflictId],
      },
    });
    expect(screen.getAllByText('Released value: none').length).toBe(2);
    expect(
      screen.queryByRole('button', { name: /choose winner|resolve conflict|approve answer/i }),
    ).toBeNull();
  });

  it('requires a digest-bound owner review and leaves conflicting evidence in the attention queue', async () => {
    const view = fixture();
    await create(view);
    const conflict = screen
      .getByText('Requests / Current', { selector: 'strong' })
      .closest('article')!;
    await view.user.click(within(conflict).getByText('Record a scoped evidence review'));
    const save = within(conflict).getByRole('button', { name: 'Record evidence review' });
    expect((save as HTMLButtonElement).disabled).toBe(true);
    await view.user.selectOptions(
      within(conflict).getByLabelText('Evidence review decision'),
      'inspected',
    );
    await view.user.click(
      within(conflict).getByRole('checkbox', { name: /I inspected the scoped/ }),
    );
    await view.user.click(save);
    await screen.findByText(/Immutable owner review: inspected/);
    expect(view.request).toHaveBeenCalledWith(`${base}/set1/results/${conflictId}/reviews`, {
      method: 'POST',
      body: { digest: '2'.repeat(64), decision: 'inspected', note: '' },
    });
    expect(screen.getByText(/Current · conflict · unresolved_evidence/)).toBeTruthy();
    const remove = screen.getByRole('button', { name: 'Delete evidence set' });
    expect((remove as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(screen.getByRole('checkbox', { name: /Confirm deletion of this set/ }));
    await view.user.click(remove);
    await waitFor(() =>
      expect(view.request).toHaveBeenCalledWith(base + '/set1', { method: 'DELETE' }),
    );
    expect(
      view.request.mock.calls.filter(([, options]) => options?.method === 'DELETE'),
    ).toHaveLength(1);
  });
});
