import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { DocumentChanges } from '../DocumentChanges';
import { ApiClient } from '../api';

const before = { id: 'a'.repeat(32), name: 'atlas-v1.pdf', digest: '1'.repeat(64) };
const target = {
  id: 'b'.repeat(32),
  name: 'atlas-v2.pdf',
  digest: '2'.repeat(64),
  parent_id: before.id,
};
const resultId = 'c'.repeat(32);
const base = `/workbench/documents/${target.id}/comparisons`;
function fixture() {
  const api = new ApiClient('authored-token');
  let created = false;
  let advanced = false;
  let enabled = false;
  const request = vi.spyOn(api, 'request').mockImplementation(async (path, options) => {
    if (path === `/workbench/documents/${before.id}`)
      return {
        manifest: { fingerprint: '3'.repeat(64) },
        results: [
          {
            id: resultId,
            payload: {
              request: { question: 'Atlas monthly metrics' },
              receipt: { status: 'evidence_found' },
            },
          },
        ],
      };
    if (path === base) {
      if (options?.method === 'POST') {
        created = true;
        enabled = Boolean((options.body as { refresh_local: boolean }).refresh_local);
        return { id: 'run1' };
      }
      return {
        comparisons: created
          ? [{ id: 'run1', processed: advanced ? 2 : 0, total: 3, complete: false }]
          : [],
      };
    }
    if (path === base + '/run1/advance') {
      advanced = true;
      return { id: 'run1' };
    }
    if (path === base + '/run1')
      return {
        id: 'run1',
        before_id: before.id,
        after_id: target.id,
        processed: advanced ? 2 : 0,
        total: 3,
        complete: false,
        checkpoint_fingerprint: advanced ? '5'.repeat(64) : 'initial',
        config: { refresh_answers: enabled, replay_calculations: enabled, max_answer_attempts: 10 },
        changes: {
          status: 'complete',
          reason: 'compared',
          metadata_changes: ['file_sha256'],
          regions: [],
        },
        items: [
          {
            source_result_id: resultId,
            target_result_id: advanced && enabled ? 'fresh-result' : null,
            title: 'Atlas monthly metrics',
            review: null,
          },
        ],
        entries: advanced
          ? [
              {
                item_id: resultId,
                status: enabled ? 'refreshed' : 'refresh_required',
                reason: enabled ? 'fresh_answer' : 'refresh_not_enabled',
                result_changed: enabled ? false : null,
                impact: {
                  status: 'dependencies_unchanged',
                  reasons: ['cited_content_unchanged'],
                  requires_reissue: true,
                },
                answer: enabled
                  ? {
                      answer: '<script>literal revised evidence</script>',
                      status: 'evidence_found',
                    }
                  : null,
              },
            ]
          : [],
        review_queue: advanced
          ? [
              {
                item_id: resultId,
                category: enabled ? 'new_version_review' : 'refresh_required',
                reason: enabled ? 'fresh_version_requires_owner_review' : 'refresh_not_enabled',
              },
            ]
          : [],
      };
    throw Error('Unexpected authored endpoint');
  });
  return { api, request };
}

describe('Version impact UI contracts (authored, no model)', () => {
  it('does not infer a version relationship or run work on mount', () => {
    const { api, request } = fixture();
    render(
      <DocumentChanges
        api={api}
        target={before}
        versions={[before]}
        fingerprint={'4'.repeat(64)}
        onUpdated={vi.fn()}
        onOpenResult={vi.fn()}
      />,
    );
    expect(
      screen.getByText(/Existing independent uploads are not silently relabeled/),
    ).toBeTruthy();
    expect(request).not.toHaveBeenCalled();
  });

  it('requires selection, defaults to assessment-only and advances explicitly with the saved fingerprint', async () => {
    const { api, request } = fixture();
    const user = userEvent.setup();
    const updated = vi.fn().mockResolvedValue(undefined);
    const open = vi.fn();
    render(
      <DocumentChanges
        api={api}
        target={target}
        versions={[target, before]}
        fingerprint={'4'.repeat(64)}
        onUpdated={updated}
        onOpenResult={open}
      />,
    );
    await user.selectOptions(screen.getByLabelText('Earlier version'), before.id);
    await screen.findByText(/Atlas monthly metrics.*evidence_found/);
    const create = screen.getByRole('button', { name: 'Create comparison' }) as HTMLButtonElement;
    expect(create.disabled).toBe(true);
    await user.click(screen.getByRole('checkbox', { name: /Atlas monthly metrics/ }));
    await user.click(create);
    await screen.findByText(/0\/3 checked/);
    expect(request).toHaveBeenCalledWith(base, {
      method: 'POST',
      body: {
        before_id: before.id,
        before_fingerprint: '3'.repeat(64),
        after_fingerprint: '4'.repeat(64),
        result_ids: [resultId],
        refresh_local: false,
        max_answer_attempts: 10,
      },
    });
    expect(request.mock.calls.filter(([path]) => path.endsWith('/advance'))).toHaveLength(0);
    await user.click(screen.getByRole('button', { name: 'Advance next two items' }));
    await screen.findByText(/2\/3 checked/);
    expect(request).toHaveBeenCalledWith(base + '/run1/advance', {
      method: 'POST',
      body: { checkpoint_fingerprint: 'initial' },
    });
    expect(updated).toHaveBeenCalledOnce();
    expect(screen.getByText(/Partial — unchecked items remain/)).toBeTruthy();
    expect(screen.getByText(/queue covers processed items only/)).toBeTruthy();
    expect(open).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Inspect fresh result and review' })).toBeNull();
  });

  it('renders fresh evidence as literal text and routes new-version results to review only after opt-in', async () => {
    const { api } = fixture();
    const user = userEvent.setup();
    const open = vi.fn();
    render(
      <DocumentChanges
        api={api}
        target={target}
        versions={[target, before]}
        fingerprint={'4'.repeat(64)}
        onUpdated={vi.fn().mockResolvedValue(undefined)}
        onOpenResult={open}
      />,
    );
    await user.selectOptions(screen.getByLabelText('Earlier version'), before.id);
    await screen.findByText(/Atlas monthly metrics.*evidence_found/);
    await user.click(screen.getByRole('checkbox', { name: /Atlas monthly metrics/ }));
    await user.click(screen.getByRole('checkbox', { name: /Enable fresh local retrieval/ }));
    await user.click(screen.getByRole('button', { name: 'Create comparison' }));
    await screen.findByText(/0\/3 checked/);
    await user.click(screen.getByRole('button', { name: 'Advance next two items' }));
    await screen.findByText('<script>literal revised evidence</script>');
    expect(document.querySelector('script')).toBeNull();
    expect(screen.getByText(/new_version_review/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Inspect fresh result and review' }));
    expect(open).toHaveBeenCalledWith('fresh-result');
  });

  it('makes refresh opt-in and refuses an out-of-range answer budget', async () => {
    const { api, request } = fixture();
    const user = userEvent.setup();
    render(
      <DocumentChanges
        api={api}
        target={target}
        versions={[target, before]}
        fingerprint={'4'.repeat(64)}
        onUpdated={vi.fn()}
        onOpenResult={vi.fn()}
      />,
    );
    await user.selectOptions(screen.getByLabelText('Earlier version'), before.id);
    await screen.findByText(/Atlas monthly metrics.*evidence_found/);
    await user.click(screen.getByRole('checkbox', { name: /Atlas monthly metrics/ }));
    await user.click(screen.getByRole('checkbox', { name: /Enable fresh local retrieval/ }));
    await user.clear(screen.getByLabelText('Maximum fresh answer attempts'));
    await user.type(screen.getByLabelText('Maximum fresh answer attempts'), '11');
    expect(
      (screen.getByRole('button', { name: 'Create comparison' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(request.mock.calls.filter(([, options]) => options?.method === 'POST')).toHaveLength(0);
  });
});
