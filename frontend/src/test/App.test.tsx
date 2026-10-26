import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import App from '../App';

function server(options: { webApproval?: boolean } = {}) {
  const calls: { path: string; init?: RequestInit }[] = [];
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
    if (path.startsWith('/api/store/threads')) return json({ threads: [] });
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
      (screen.getByRole('button', { name: 'Sign in to CiteRight' }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
  await user.click(screen.getByRole('button', { name: 'Sign in to CiteRight' }));
  await screen.findByRole('heading', { name: 'Ask. Check the sources.' });
  return user;
}
describe('CiteRight UI (authored HTTP responses, no provider)', () => {
  it('makes no model requests on mount, sign-in or history loads', async () => {
    const calls = server();
    await login();
    expect(calls.filter((c) => c.init?.method === 'POST').map((c) => c.path)).toEqual([
      '/api/auth/login',
    ]);
    expect(localStorage.length).toBe(0);
    expect(sessionStorage.length).toBe(0);
  });
  it('streams safely as text and signs out without persisted credentials', async () => {
    const calls = server();
    const user = await login();
    expect(
      (screen.getByRole('button', { name: /Send question/ }) as HTMLButtonElement).disabled,
    ).toBe(true);
    await user.type(screen.getByLabelText('Your question'), 'Show the evidence');
    await user.click(screen.getByRole('button', { name: /Send question/ }));
    await screen.findByText('<script>untrusted</script>');
    expect(document.querySelector('script')).toBeNull();
    expect(calls.filter((c) => c.path === '/api/stream')).toHaveLength(1);
    await user.click(screen.getByRole('button', { name: 'Sign out' }));
    await screen.findByRole('heading', { name: 'Welcome to CiteRight' });
    expect((screen.getByLabelText('Password') as HTMLInputElement).value).toBe('');
    expect(localStorage.length + sessionStorage.length).toBe(0);
  });
  it('resumes web approval explicitly within the same conversation', async () => {
    const calls = server({ webApproval: true });
    const user = await login();
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
