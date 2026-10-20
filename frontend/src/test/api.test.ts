import { describe, it, expect, vi } from 'vitest';
import { ApiClient, ApiError, SSEParser, apiPath } from '../api';

describe('same-origin API boundary', () => {
  it('sends separate explicit pixel OCR consent with no language, expected text or crop settings', async () => {
    const timeout = vi.spyOn(AbortSignal, 'timeout');
    const fetcher = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{"status":"all_cells_agree"}'));
    await new ApiClient('authored-token').crosscheckPixels(
      'set1',
      'audit1',
      'doc1',
      'a'.repeat(64),
      new AbortController().signal,
    );
    expect(timeout).toHaveBeenCalledWith(180_000);
    expect(fetcher).toHaveBeenCalledExactlyOnceWith(
      '/api/workbench/documents/reconciliations/set1/audits/audit1/sources/doc1/crosscheck-pixels',
      expect.objectContaining({
        method: 'POST',
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        headers: {
          Accept: 'application/json',
          Authorization: 'Bearer authored-token',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ checkpoint_fingerprint: 'a'.repeat(64), confirm_local_ocr: true }),
      }),
    );
  });
  it('cross-checks only one explicit source with no caller coordinates or OCR capability', async () => {
    const timeout = vi.spyOn(AbortSignal, 'timeout');
    const fetcher = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{"status":"all_cells_agree"}'));
    await new ApiClient('authored-token').crosscheckCells(
      'set1',
      'audit1',
      'doc1',
      'a'.repeat(64),
      new AbortController().signal,
    );
    expect(timeout).toHaveBeenCalledWith(180_000);
    expect(fetcher).toHaveBeenCalledExactlyOnceWith(
      '/api/workbench/documents/reconciliations/set1/audits/audit1/sources/doc1/crosscheck-cells',
      expect.objectContaining({
        method: 'POST',
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        headers: {
          Accept: 'application/json',
          Authorization: 'Bearer authored-token',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ checkpoint_fingerprint: 'a'.repeat(64), confirm_crosscheck: true }),
      }),
    );
  });
  it('re-extracts only the explicit authenticated source with a fixed long request deadline', async () => {
    const timeout = vi.spyOn(AbortSignal, 'timeout');
    const fetcher = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{"status":"exact_snapshot_match"}'));
    await new ApiClient('authored-token').replayOriginal(
      'set1',
      'audit1',
      'doc1',
      'a'.repeat(64),
      false,
      new AbortController().signal,
    );
    expect(timeout).toHaveBeenCalledWith(180_000);
    expect(fetcher).toHaveBeenCalledExactlyOnceWith(
      '/api/workbench/documents/reconciliations/set1/audits/audit1/sources/doc1/replay-original',
      expect.objectContaining({
        method: 'POST',
        credentials: 'omit',
        redirect: 'error',
        cache: 'no-store',
        headers: {
          Accept: 'application/json',
          Authorization: 'Bearer authored-token',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          checkpoint_fingerprint: 'a'.repeat(64),
          confirm_reextract: true,
          allow_original_ocr: false,
        }),
      }),
    );
  });
  it('preserves fingerprinted JSON numeric spellings across streamed UTF-8 chunks', async () => {
    const raw = '{"coordinate":-0.0,"label":"café"}';
    const bytes = new TextEncoder().encode(raw);
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        new ReadableStream({
          start(controller) {
            // Deliberately split a multi-byte character across read boundaries.
            for (const value of bytes) controller.enqueue(new Uint8Array([value]));
            controller.close();
          },
        }),
      ),
    );
    const api = new ApiClient('authored-token');
    expect(
      await api.requestJSONText(
        '/workbench/documents/reconciliations/set/audits/audit/replay-bundle',
        {
          method: 'POST',
          body: { include_extraction_text: true },
        },
      ),
    ).toBe(raw);
    expect(fetcher.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      credentials: 'omit',
      redirect: 'error',
      cache: 'no-store',
      headers: { Authorization: 'Bearer authored-token' },
    });
  });
  it('cancels over-budget replay bytes before parsing or download', async () => {
    const cancel = vi.fn();
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(3_000_001));
          },
          cancel,
        }),
      ),
    );
    await expect(
      new ApiClient('authored').requestJSONText('/workbench/config', { method: 'POST' }),
    ).rejects.toThrow(/3 MB budget/);
    expect(cancel).toHaveBeenCalledOnce();
  });
  it('sends an explicit safe parent version without accepting arbitrary paths', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"id":"doc"}'));
    const api = new ApiClient('authored-token');
    const file = new File(['%PDF-fixture'], 'report.pdf');
    await api.uploadDocument(file, false, 'a'.repeat(32));
    expect(fetcher.mock.calls[0][0]).toBe(
      '/api/workbench/documents?name=report.pdf&ocr=false&parent_id=' + 'a'.repeat(32),
    );
    await expect(api.uploadDocument(file, false, '../private')).rejects.toThrow();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
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
    await expect(new ApiClient('authored-token').request('/workbench/documents')).rejects.toThrow(
      /display budget/,
    );
    expect(cancel).toHaveBeenCalledOnce();
  });
  it('uploads raw media only to the authenticated local document endpoint', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"id":"doc1"}'));
    const api = new ApiClient('authored-token');
    const file = new File(['%PDF-fixture'], 'Annual (2026).pdf', { type: 'application/pdf' });
    await expect(api.uploadDocument(file, false)).resolves.toEqual({ id: 'doc1' });
    expect(fetcher.mock.calls[0][0]).toBe(
      '/api/workbench/documents?name=Annual%20%282026%29.pdf&ocr=false',
    );
    expect(fetcher.mock.calls[0][1]).toMatchObject({
      method: 'POST',
      body: file,
      headers: { Authorization: 'Bearer authored-token', 'Content-Type': 'application/pdf' },
      credentials: 'omit',
      cache: 'no-store',
      redirect: 'error',
    });
  });
  it('rejects unsafe upload names, types and sizes before fetching', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch');
    const api = new ApiClient('authored-token');
    for (const file of [
      new File(['x'], '../private.pdf'),
      new File(['x'], 'x.svg'),
      new File([], 'empty.pdf'),
      new File(['x'.repeat(3_000_001)], 'huge.pdf'),
    ]) {
      await expect(api.uploadDocument(file, false)).rejects.toThrow();
    }
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('rejects external, traversal and credential destinations', () => {
    for (const path of [
      'https://outside.invalid',
      '//outside.invalid',
      '/code/../metrics',
      '/code\\tasks',
      '/x#secret',
      '/x@outside',
    ])
      expect(() => apiPath(path)).toThrow();
    expect(apiPath('/code/tasks/task-1/events')).toBe('/api/code/tasks/task-1/events');
  });
  it('sends a bearer only to the fixed local proxy with no cookies or retries', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"ok":true}'));
    const api = new ApiClient('authored-token');
    await expect(api.request('/workbench/config')).resolves.toEqual({ ok: true });
    expect(fetcher.mock.calls[0][0]).toBe('/api/workbench/config');
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
    const error = await api.request('/workbench/config').catch((e) => e);
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
