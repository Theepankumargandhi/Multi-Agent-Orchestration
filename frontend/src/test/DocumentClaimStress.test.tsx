import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { DocumentClaimStress } from '../DocumentClaimStress';
import { ApiClient } from '../api';

const base = '/workbench/documents/reconciliations/set1/audits/audit1/stress-tests';
function fixture(complete = true, conflict = false) {
  const api = new ApiClient('authored-session');
  let created = false,
    advanced = false;
  const check = (supported: boolean) => ({
    status: supported ? 'supported' : 'unresolved',
    reason: supported ? 'exact_value_match' : 'unresolved_evidence',
    observed_value: supported ? '25' : null,
    observed_unit: supported ? '%' : null,
  });
  const request = vi.spyOn(api, 'request').mockImplementation(async (path, options) => {
    if (path === base) {
      if (options?.method === 'POST') {
        created = true;
        return { id: 'stress1' };
      }
      return { stress_tests: created ? [{ id: 'stress1', name: 'Authored fragility' }] : [] };
    }
    if (path === base + '/stress1/advance') {
      advanced = true;
      return { id: 'stress1' };
    }
    if (path === base + '/stress1') {
      if (options?.method === 'DELETE') {
        created = false;
        return { deleted: true };
      }
      return {
        id: 'stress1',
        name: 'Authored fragility',
        sources: [{ id: 'application-row', name: 'report.pdf' }],
        checkpoint_fingerprint: (advanced ? 'c' : 'b').repeat(64),
        processed: advanced ? 1 : 0,
        total: 1,
        complete: advanced,
        evaluated_checks: advanced ? 1 : 0,
        support_lost_cases: 0,
        support_appeared_cases: advanced && conflict ? 1 : 0,
        claims: [
          {
            claim: {
              claim_id: 'growth',
              label: '<script>growth</script>',
              expected_value: '25',
              expected_unit: '%',
            },
            baseline: check(!conflict),
            stable_numeric_support: advanced && !conflict ? true : null,
            variants: [
              {
                omitted_source_id: 'application-row',
                check: advanced ? check(true) : null,
                effect: advanced
                  ? conflict
                    ? 'support_appeared_after_omission'
                    : 'unchanged_verdict'
                  : 'pending',
                observations: advanced
                  ? [
                      {
                        source_id: 'retained-application-row',
                        status: 'observed',
                        citations: [{ page: 2, region_id: 'd'.repeat(64), quote: '125' }],
                      },
                    ]
                  : [],
              },
            ],
          },
        ],
      };
    }
    throw Error('Unexpected authored endpoint');
  });
  const inspect = vi.fn();
  render(
    <DocumentClaimStress
      api={api}
      setId="set1"
      auditId="audit1"
      sourceCheckpoint={'a'.repeat(64)}
      complete={complete}
      onInspect={inspect}
    />,
  );
  return { request, inspect, user: userEvent.setup() };
}
async function create(view: ReturnType<typeof fixture>) {
  await view.user.click(screen.getByRole('button', { name: 'Open evidence stress tests' }));
  await view.user.type(screen.getByLabelText('Stress test name'), 'Authored fragility');
  await view.user.click(screen.getByRole('button', { name: 'Create evidence stress test' }));
  await screen.findByRole('heading', { name: 'Authored fragility' });
}

describe('Evidence fragility diagnostic UI', () => {
  it('does no work on mount and requires a completed source audit', () => {
    const view = fixture(false);
    expect(screen.getByText(/Complete the source audit/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Open evidence stress tests' })).toBeNull();
    expect(view.request).not.toHaveBeenCalled();
  });

  it('creates a checkpoint-bound report without choosing sources or automatically advancing', async () => {
    const view = fixture();
    expect(view.request).not.toHaveBeenCalled();
    await create(view);
    expect(view.request).toHaveBeenCalledWith(base, {
      method: 'POST',
      body: { name: 'Authored fragility', source_checkpoint_fingerprint: 'a'.repeat(64) },
    });
    expect(view.request.mock.calls.some(([path]) => path.endsWith('/advance'))).toBe(false);
    expect(screen.getByText('Partial — source omissions remain unchecked')).toBeTruthy();
    expect(screen.getByText(/incomplete; no stability claim/)).toBeTruthy();
    expect(screen.getByRole('columnheader', { name: 'Original audit' })).toBeTruthy();
    expect(screen.getByText(/Pending cells are not passes/)).toBeTruthy();
    expect(document.querySelector('script')).toBeNull();
  });

  it('advances explicitly, navigates retained application citations and deletes only the report', async () => {
    const view = fixture();
    await create(view);
    await view.user.click(screen.getByRole('button', { name: 'Run next source omission' }));
    await screen.findByText('Single-source tests complete');
    expect(view.request).toHaveBeenCalledWith(base + '/stress1/advance', {
      method: 'POST',
      body: { checkpoint_fingerprint: 'b'.repeat(64) },
    });
    expect(
      screen.getByText(/numeric support survived every tested single-source omission/),
    ).toBeTruthy();
    await view.user.click(screen.getByRole('button', { name: /Inspect .* without report.pdf/ }));
    await view.user.click(screen.getByText('retained-application-row · observed'));
    await view.user.click(screen.getByRole('button', { name: 'Retained page 2: 125' }));
    expect(view.inspect).toHaveBeenCalledWith('retained-application-row', 2, 'd'.repeat(64));
    const remove = screen.getByRole('button', { name: 'Delete evidence stress test' });
    expect((remove as HTMLButtonElement).disabled).toBe(true);
    await view.user.click(
      screen.getByRole('checkbox', { name: /Confirm deletion of this stress report only/ }),
    );
    await view.user.click(remove);
    await waitFor(() =>
      expect(view.request).toHaveBeenCalledWith(base + '/stress1', { method: 'DELETE' }),
    );
    expect(
      view.request.mock.calls.filter(([, options]) => options?.method === 'DELETE'),
    ).toHaveLength(1);
  });

  it('labels apparent support after omission as diagnostic and never replaces the original verdict', async () => {
    const view = fixture(true, true);
    await create(view);
    await view.user.click(screen.getByRole('button', { name: 'Run next source omission' }));
    await screen.findByText('support_appeared_after_omission');
    expect(screen.getByText('unresolved')).toBeTruthy();
    expect(screen.getByText(/baseline was not supported; no stability claim/)).toBeTruthy();
    expect(screen.getByText(/it never settles the original conflict/)).toBeTruthy();
    expect(screen.queryByText(/numeric support survived/)).toBeNull();
  });
});
