import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import App from '../App';

const config = {
  version: '0.1.0',
  identity: 'reviewer',
  code_enabled: true,
  repositories: ['example'],
  coding_models: ['authored-model'],
  workflow: 'verified_pr',
  test_command: ['python', '-m', 'pytest', '-q'],
  integrations: {},
};
function server(
  options: {
    disabled?: boolean;
    unauthorized?: boolean;
    failFirstSubmission?: boolean;
    webApproval?: boolean;
  } = {},
) {
  const calls: { path: string; init?: RequestInit }[] = [];
  let taskStatus = 'awaiting_approval';
  let submissions = 0;
  let streams = 0;
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const path = String(input);
    calls.push({ path, init });
    const json = (value: unknown, status = 200) =>
      new Response(JSON.stringify(value), {
        status,
        headers: { 'content-type': 'application/json' },
      });
    if (path === '/api/capabilities')
      return json({
        models: [{ id: 'authored-model', label: 'Authored test model' }],
        features: { user_authentication: true },
      });
    if (path === '/api/auth/login' || path === '/api/auth/register')
      return json({ user_id: 'reviewer', access_token: 'authored-session', expires_in: 3600 });
    if (path === '/api/workbench/config')
      return options.unauthorized
        ? json({}, 401)
        : json({ ...config, code_enabled: !options.disabled });
    if (path.startsWith('/api/store/threads')) return json({ threads: [] });
    if (path.startsWith('/api/workbench/code/tasks?')) return json({ tasks: [] });
    if (path === '/api/workbench/code/tasks' && init?.method === 'POST') {
      submissions += 1;
      if (options.failFirstSubmission && submissions === 1)
        throw new TypeError('Authored network failure');
      return json(
        {
          task_id: 'task-1',
          status: 'queued',
          request: {
            repository: 'example',
            issue: 'Fix the rounding failure',
            model: 'authored-model',
          },
        },
        202,
      );
    }
    const task = {
      task_id: 'task-1',
      status: taskStatus,
      request: {
        repository: 'example',
        issue: 'Fix the rounding failure',
        model: 'authored-model',
      },
      result: {
        summary: 'Authored test result',
        changed_files: ['app.py'],
        model_calls: 1,
        tool_calls: 2,
        prompt_tokens: 0,
        completion_tokens: 0,
        usage_accounting_complete: false,
        verification: { final_decision: 'verified', blocking_reasons: [] },
      },
    };
    if (path === '/api/code/tasks/task-1') return json(task);
    if (path === '/api/code/tasks/task-1/decision') {
      taskStatus = JSON.parse(String(init?.body)).approve ? 'approved' : 'rejected';
      return json({ ...task, status: taskStatus });
    }
    if (path === '/api/code/tasks/task-1/patch')
      return json({
        patch: '--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-return x < 0\n+return x <= 0',
      });
    if (path === '/api/code/tasks/task-1/events') return json({ events: [] });
    if (path === '/api/stream') {
      streams += 1;
      const content =
        options.webApproval && streams === 1
          ? 'Human approval required before web-answer generation. Query: Authored query'
          : '<script>untrusted</script>';
      return new Response(
        'data: ' +
          JSON.stringify({ type: 'message', content: { type: 'ai', content } }) +
          '\n\ndata: [DONE]\n\n',
        { headers: { 'content-type': 'text/event-stream' } },
      );
    }
    throw new Error(`Unexpected test request: ${path}`);
  });
  return calls;
}
async function login() {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText('User ID'), 'reviewer');
  await user.type(screen.getByLabelText('Password'), 'authored-password');
  await waitFor(() =>
    expect(
      (screen.getByRole('button', { name: 'Sign in to Workbench' }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
  await user.click(screen.getByRole('button', { name: 'Sign in to Workbench' }));
  await screen.findByRole('heading', { name: 'Make the work inspectable.' });
  return user;
}
describe('Workbench integration (authored HTTP responses, no provider)', () => {
  it('makes no model requests on mount, sign-in, navigation or history loads', async () => {
    const calls = server();
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Research' }));
    await screen.findByRole('heading', { name: 'Ask. Inspect. Verify.' });
    await user.click(screen.getByRole('button', { name: 'Coding' }));
    await screen.findByRole('heading', { name: 'A patch you can inspect.' });
    expect(calls.filter((c) => c.init?.method === 'POST').map((c) => c.path)).toEqual([
      '/api/auth/login',
    ]);
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });
  it('requires session consent, streams safely as text, and signs out without persisted credentials', async () => {
    const calls = server();
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Research' }));
    await user.type(screen.getByLabelText('Your question'), 'Show the evidence');
    expect(
      (screen.getByRole('button', { name: /Send question/ }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await user.click(screen.getByLabelText('Allow model requests'));
    await user.click(screen.getByRole('button', { name: /Send question/ }));
    await screen.findByText('<script>untrusted</script>');
    expect(document.querySelector('script')).toBeNull();
    expect(calls.filter((c) => c.path === '/api/stream')).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: 'Sign out' }));
    await screen.findByRole('heading', { name: 'Welcome to your workspace' });
    expect((screen.getByLabelText('Password') as HTMLInputElement).value).toBe('');
    expect(localStorage.length + sessionStorage.length).toBe(0);
  });
  it('submits only a model, allowlisted repository and issue; patch access stays gated', async () => {
    const calls = server();
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Coding' }));
    await user.type(screen.getByLabelText('Issue'), 'Fix the rounding failure');
    await user.click(screen.getByLabelText('Allow model requests'));
    await user.click(screen.getByRole('button', { name: /Submit repair task/ }));
    await screen.findByText('Authored test result');
    const submission = calls.find(
      (c) => c.path === '/api/workbench/code/tasks' && c.init?.method === 'POST',
    );
    expect(JSON.parse(String(submission?.init?.body))).toEqual({
      repository: 'example',
      model: 'authored-model',
      issue: 'Fix the rounding failure',
    });
    expect((submission?.init?.headers as Record<string, string>)['Idempotency-Key']).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'View approved patch' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Approve patch access' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(screen.getByText('Unknown')).toBeTruthy();
  });
  it('keeps document ingestion off when the backend has not enabled it', async () => {
    const calls = server();
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Documents' }));
    expect(await screen.findByText(/Enable the isolated document runtime first/)).toBeTruthy();
    expect(calls.filter((c) => c.init?.method === 'POST')).toHaveLength(1);
  });
  it('keeps coding disabled when the backend disables it', async () => {
    server({ disabled: true });
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Coding' }));
    expect(
      await screen.findByRole('heading', { name: 'Coding execution is not enabled.' }),
    ).toBeTruthy();
    expect(screen.queryByRole('button', { name: /Submit repair task/ })).toBeNull();
  });
  it('preserves the idempotency key after an uncertain submission and unlocks patch viewing only after a decision', async () => {
    const calls = server({ failFirstSubmission: true });
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Coding' }));
    await user.type(screen.getByLabelText('Issue'), 'Fix the rounding failure');
    await user.click(screen.getByLabelText('Allow model requests'));
    await user.click(screen.getByRole('button', { name: /Submit repair task/ }));
    await screen.findByText(/Submission may have reached the server/);
    await user.click(screen.getByRole('button', { name: /Submit repair task/ }));
    await screen.findByText('Authored test result');
    const requests = calls.filter(
      (c) => c.path === '/api/workbench/code/tasks' && c.init?.method === 'POST',
    );
    expect(requests).toHaveLength(2);
    expect((requests[0].init?.headers as Record<string, string>)['Idempotency-Key']).toBe(
      (requests[1].init?.headers as Record<string, string>)['Idempotency-Key'],
    );
    expect(calls.some((c) => c.path.endsWith('/patch'))).toBe(false);
    await user.click(
      screen.getByLabelText('I reviewed the execution evidence and want to release patch access.'),
    );
    await user.click(screen.getByRole('button', { name: 'Approve patch access' }));
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'View approved patch' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    await user.click(screen.getByRole('button', { name: 'View approved patch' }));
    await screen.findByText('+return x <= 0');
    expect(calls.filter((c) => c.path.endsWith('/decision'))).toHaveLength(1);
    expect(calls.some((c) => /apply|merge/.test(c.path))).toBe(false);
  });
  it('resumes web approval explicitly within the same conversation', async () => {
    const calls = server({ webApproval: true });
    const user = await login();
    await user.click(screen.getByRole('button', { name: 'Research' }));
    await user.click(screen.getByLabelText('Allow model requests'));
    await user.type(screen.getByLabelText('Your question'), 'Find evidence for this question');
    await user.click(screen.getByRole('button', { name: /Send question/ }));
    await user.click(await screen.findByRole('button', { name: 'Approve web access' }));
    await screen.findByText('<script>untrusted</script>');
    const bodies = calls
      .filter((c) => c.path === '/api/stream')
      .map((c) => JSON.parse(String(c.init?.body)));
    expect(bodies).toHaveLength(2);
    expect(bodies[1].message).toBe('approve');
    expect(bodies[1].thread_id).toBe(bodies[0].thread_id);
  });
});
