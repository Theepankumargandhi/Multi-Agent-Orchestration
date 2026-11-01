import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import App from '../App';

function server(options: { webApproval?: boolean; uploads?: boolean } = {}) {
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
        features: { user_authentication: true, document_upload: Boolean(options.uploads) },
      });
    if (path === '/api/documents') return json({ count: 0, documents: [] });
    if (path.startsWith('/api/documents?file_name='))
      return json({
        document_id: 'a'.repeat(64),
        file_name: 'Q3_report.pdf',
        pages: 4,
        chunk_count: 9,
        uploaded_at: 1,
      });
    if (path === `/api/documents/${'a'.repeat(64)}`) return json({ chunks_removed: 9 });
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
  it('uploads a document, answers from it, and removes it', async () => {
    const calls = server({ uploads: true });
    const user = await login();
    expect(screen.queryByLabelText('Answer from my documents')).toBeNull();
    await user.upload(
      screen.getByLabelText('Document file'),
      new File(['%PDF-1.4'], 'Q3 report.pdf', { type: 'application/pdf' }),
    );
    await screen.findByText('Q3_report.pdf');
    const sent = calls.find((c) => c.path.startsWith('/api/documents?'));
    expect(sent?.path).toBe('/api/documents?file_name=Q3_report.pdf');
    expect(sent?.init?.body).toBeInstanceOf(File);
    // Answering from documents switches on after an upload.
    expect((screen.getByLabelText('Answer from my documents') as HTMLInputElement).checked).toBe(
      true,
    );
    await user.type(screen.getByLabelText('Your question'), 'What grew in Q3?');
    await user.click(screen.getByRole('button', { name: /Send question/ }));
    await screen.findByText('<script>untrusted</script>');
    const body = JSON.parse(String(calls.find((c) => c.path === '/api/stream')?.init?.body));
    expect(body.message).toBe('local: What grew in Q3?');
    expect(screen.getByText('What grew in Q3?')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Remove Q3_report.pdf' }));
    await waitFor(() => expect(screen.queryByText('Q3_report.pdf')).toBeNull());
    expect(calls.at(-1)?.init?.method).toBe('DELETE');
    expect(screen.queryByLabelText('Answer from my documents')).toBeNull();
  });
});
