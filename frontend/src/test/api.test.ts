import { describe, it, expect, vi } from 'vitest';
import { ApiClient, ApiError, SSEParser, apiPath } from '../api';

describe('same-origin API boundary', () => {
  it('cancels JSON response reads that exceed the byte budget', async () => {
    const cancel = vi.fn();
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(8_000_001));
          },
          cancel,
        }),
      ),
    );
    await expect(new ApiClient('authored-token').request('/store/threads')).rejects.toThrow(
      /display budget/,
    );
    expect(cancel).toHaveBeenCalledOnce();
  });
  it('rejects external, traversal and credential destinations', () => {
    for (const path of [
      'https://outside.invalid',
      '//outside.invalid',
      '/store/../metrics',
      '/code\\tasks',
      '/x#secret',
      '/x@outside',
    ])
      expect(() => apiPath(path)).toThrow();
    expect(apiPath('/store/thread-1?limit=200')).toBe('/api/store/thread-1?limit=200');
  });
  it('sends a bearer only to the fixed local proxy with no cookies or retries', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"ok":true}'));
    const api = new ApiClient('authored-token');
    await expect(api.request('/capabilities')).resolves.toEqual({ ok: true });
    expect(fetcher.mock.calls[0][0]).toBe('/api/capabilities');
    expect(fetcher.mock.calls[0][1]).toMatchObject({
      headers: { Authorization: 'Bearer authored-token' },
      credentials: 'omit',
      cache: 'no-store',
      redirect: 'error',
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
  it('clears an expired session and never reflects private error bodies', async () => {
    const expire = vi.fn();
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('PRIVATE PASSWORD', { status: 401 }),
    );
    const api = new ApiClient('authored-token', expire);
    const error = await api.request('/store/threads').catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(String(error)).not.toContain('PRIVATE');
    expect(expire).toHaveBeenCalledOnce();
  });
});
describe('existing POST SSE protocol', () => {
  it('handles chunk boundaries, CRLF, unicode, comments and completion', () => {
    const receive = vi.fn();
    const parser = new SSEParser(receive);
    const raw =
      ': keepalive\r\ndata: {"type":"token","content":"café"}\r\n\r\ndata: {"type":"message","content":{"type":"ai","content":"Verified excerpt"}}\n\ndata: [DONE]\n\n';
    for (const char of raw) parser.push(char);
    expect(parser.done).toBe(true);
    expect(receive.mock.calls.map((call) => call[0])).toEqual([
      { type: 'token', content: 'café' },
      { type: 'message', content: { type: 'ai', content: 'Verified excerpt' } },
    ]);
  });
  it('does not display provider errors or accept malformed messages', () => {
    const parser = new SSEParser(vi.fn());
    expect(() => parser.push('data: {"type":"error","content":"PRIVATE"}\n\n')).toThrow(
      'Agent execution failed',
    );
    expect(() =>
      new SSEParser(vi.fn()).push('data: {"type":"message","content":{"type":"other"}}\n'),
    ).toThrow('Invalid chat');
    expect(() => new SSEParser(vi.fn()).push('x'.repeat(8_000_001))).toThrow('display budget');
  });
  it('detects early stream termination rather than reporting success', async () => {
    const bytes = new TextEncoder().encode('data: {"type":"token","content":"unfinished"}\n\n');
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(bytes);
            controller.close();
          },
        }),
        { headers: { 'content-type': 'text/event-stream' } },
      ),
    );
    await expect(
      new ApiClient('authored').stream(
        { message: 'Question', model: 'test', thread_id: 'thread' },
        new AbortController().signal,
        vi.fn(),
      ),
    ).rejects.toThrow('before completion');
  });
});
