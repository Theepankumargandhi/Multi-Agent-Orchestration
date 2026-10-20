import type { ChatMessage, Session } from './types';

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
const MAX_JSON = 8_000_000;
type RequestOptions = {
  method?: string;
  body?: unknown;
  signal?: AbortSignal;
  idempotencyKey?: string;
};
async function readJSONText(response: Response, maximum = MAX_JSON): Promise<string> {
  if (!response.body) throw new Error('The service returned an empty response.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
  const parts: string[] = [];
  let total = 0;
  try {
    while (true) {
      const part = await reader.read();
      if (part.done) break;
      total += part.value.byteLength;
      if (total > maximum)
        throw new Error(
          maximum === MAX_JSON
            ? 'Response exceeds the Workbench display budget.'
            : 'The replay bundle exceeds its 3 MB budget.',
        );
      parts.push(decoder.decode(part.value, { stream: true }));
    }
    parts.push(decoder.decode());
    return parts.join('');
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
async function readJSON<T>(response: Response): Promise<T> {
  return JSON.parse(await readJSONText(response)) as T;
}
export const apiPath = (path: string) => {
  if (!/^\/[a-z][a-zA-Z0-9_/?=.&%:-]*$/.test(path) || path.includes('..') || path.includes('//'))
    throw new Error('Invalid API path');
  return '/api' + path;
};

async function checked(response: Response): Promise<Response> {
  if (!response.ok) {
    // Never reflect raw server/proxy errors, submitted passwords or provider response bodies.
    const message =
      response.status === 401
        ? 'Your session is unavailable or expired. Sign in again.'
        : response.status === 403
          ? 'This operation is not permitted for your account or repository.'
          : response.status === 409
            ? 'The state changed. Refresh before trying again.'
            : response.status === 422
              ? 'Check the form values and try again.'
              : response.status === 429
                ? 'The service is busy or a request limit was reached.'
                : response.status === 404
                  ? 'This capability or record is unavailable.'
                  : 'The service could not complete the request. Check backend configuration.';
    throw new ApiError(response.status, message);
  }
  return response;
}

export class ApiClient {
  private lifetime = new AbortController();
  constructor(
    private token = '',
    private onExpire?: () => void,
  ) {}
  dispose() {
    this.lifetime.abort();
  }
  private async check(response: Response) {
    if (response.status === 401 && this.token) this.onExpire?.();
    return checked(response);
  }
  private async response(
    path: string,
    options: RequestOptions,
    timeoutMs = 15_000,
  ): Promise<Response> {
    const headers: Record<string, string> = { Accept: 'application/json' };
    if (this.token) headers.Authorization = `Bearer ${this.token}`;
    if (options.body !== undefined) headers['Content-Type'] = 'application/json';
    if (options.idempotencyKey) headers['Idempotency-Key'] = options.idempotencyKey;
    const response = await this.check(
      await fetch(apiPath(path), {
        method: options.method ?? 'GET',
        headers,
        body: options.body === undefined ? undefined : JSON.stringify(options.body),
        signal: AbortSignal.any([
          this.lifetime.signal,
          AbortSignal.timeout(timeoutMs),
          ...(options.signal ? [options.signal] : []),
        ]),
        credentials: 'omit',
        cache: 'no-store',
        redirect: 'error',
      }),
    );
    return response;
  }
  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    return readJSON<T>(await this.response(path, options));
  }
  async requestJSONText(path: string, options: RequestOptions): Promise<string> {
    // Preserve numeric spellings such as -0.0 in fingerprinted extraction manifests.
    return readJSONText(await this.response(path, options), 3_000_000);
  }
  async replayOriginal<T>(
    setId: string,
    auditId: string,
    sourceId: string,
    checkpoint: string,
    allowOCR: boolean,
    signal: AbortSignal,
  ): Promise<T> {
    return readJSON<T>(
      await this.response(
        `/workbench/documents/reconciliations/${setId}/audits/${auditId}/sources/${sourceId}/replay-original`,
        {
          method: 'POST',
          signal,
          body: {
            checkpoint_fingerprint: checkpoint,
            confirm_reextract: true,
            allow_original_ocr: allowOCR,
          },
        },
        180_000,
      ),
    );
  }
  authenticate(mode: 'login' | 'register', user_id: string, password: string) {
    return this.request<Session>(`/auth/${mode}`, { method: 'POST', body: { user_id, password } });
  }
  async crosscheckCells<T>(
    setId: string,
    auditId: string,
    sourceId: string,
    checkpoint: string,
    signal: AbortSignal,
  ): Promise<T> {
    return readJSON<T>(
      await this.response(
        `/workbench/documents/reconciliations/${setId}/audits/${auditId}/sources/${sourceId}/crosscheck-cells`,
        {
          method: 'POST',
          signal,
          body: { checkpoint_fingerprint: checkpoint, confirm_crosscheck: true },
        },
        180_000,
      ),
    );
  }
  async crosscheckPixels<T>(
    setId: string,
    auditId: string,
    sourceId: string,
    checkpoint: string,
    signal: AbortSignal,
  ): Promise<T> {
    return readJSON<T>(
      await this.response(
        `/workbench/documents/reconciliations/${setId}/audits/${auditId}/sources/${sourceId}/crosscheck-pixels`,
        {
          method: 'POST',
          signal,
          body: { checkpoint_fingerprint: checkpoint, confirm_local_ocr: true },
        },
        180_000,
      ),
    );
  }
  async uploadDocument(file: File, ocr: boolean, parentId?: string): Promise<{ id: string }> {
    if (parentId && !/^[a-f0-9]{32}$/.test(parentId)) throw new Error('Invalid parent version.');
    if (!/^[A-Za-z0-9][A-Za-z0-9 _().-]{0,119}$/.test(file.name) || file.name.includes('..'))
      throw new Error(
        'Rename the document using a short name with letters, numbers, spaces or parentheses.',
      );
    if (!file.size || file.size > 3_000_000) throw new Error('Choose a document up to 3 MB.');
    const suffix = file.name.split('.').pop()?.toLowerCase();
    const type =
      suffix === 'pdf'
        ? 'application/pdf'
        : suffix === 'png'
          ? 'image/png'
          : suffix === 'jpg' || suffix === 'jpeg'
            ? 'image/jpeg'
            : '';
    if (!type) throw new Error('Choose a PDF, PNG or JPEG.');
    const response = await this.check(
      await fetch(
        apiPath(
          `/workbench/documents?name=${encodeURIComponent(file.name).replace(/[()]/g, (c) => '%' + c.charCodeAt(0).toString(16))}&ocr=${ocr}${parentId ? '&parent_id=' + parentId : ''}`,
        ),
        {
          method: 'POST',
          headers: { Authorization: `Bearer ${this.token}`, 'Content-Type': type },
          body: file,
          signal: AbortSignal.any([this.lifetime.signal, AbortSignal.timeout(180_000)]),
          credentials: 'omit',
          cache: 'no-store',
          redirect: 'error',
        },
      ),
    );
    return readJSON<{ id: string }>(response);
  }
  async stream(
    body: { message: string; model: string; thread_id: string },
    signal: AbortSignal,
    receive: (event: StreamEvent) => void,
  ) {
    const response = await this.check(
      await fetch(apiPath('/stream'), {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${this.token}`,
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({ ...body, stream_tokens: true }),
        signal: AbortSignal.any([this.lifetime.signal, signal]),
        credentials: 'omit',
        cache: 'no-store',
        redirect: 'error',
      }),
    );
    if (!response.headers.get('content-type')?.includes('text/event-stream') || !response.body)
      throw new Error('Streaming is unavailable.');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const parser = new SSEParser(receive);
    try {
      while (!parser.done) {
        const part = await reader.read();
        if (part.done) {
          parser.push(decoder.decode());
          break;
        }
        parser.push(decoder.decode(part.value, { stream: true }));
      }
      if (!parser.done)
        throw new Error('The stream ended before completion. Check history before resubmitting.');
    } finally {
      await reader.cancel().catch(() => undefined);
      reader.releaseLock();
    }
  }
}

export type StreamEvent =
  { type: 'token'; content: string } | { type: 'message'; content: ChatMessage };
export class SSEParser {
  private buffer = '';
  private total = 0;
  done = false;
  constructor(private receive: (event: StreamEvent) => void) {}
  push(chunk: string) {
    if (this.done) return;
    this.total += chunk.length;
    if (this.total > MAX_JSON) throw new Error('Stream exceeds the Workbench display budget.');
    this.buffer += chunk;
    let boundary: number;
    while ((boundary = this.buffer.indexOf('\n')) >= 0) {
      const line = this.buffer.slice(0, boundary).replace(/\r$/, '');
      this.buffer = this.buffer.slice(boundary + 1);
      // The existing service emits one complete JSON event per data line.
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trimStart();
      if (data === '[DONE]') {
        this.done = true;
        this.buffer = '';
        break;
      }
      const value: unknown = JSON.parse(data);
      if (!value || typeof value !== 'object') throw new Error('Invalid stream event.');
      const event = value as { type?: string; content?: unknown };
      if (event.type === 'error')
        throw new Error('Agent execution failed. Check history before resubmitting.');
      if (event.type === 'token' && typeof event.content === 'string')
        this.receive({ type: 'token', content: event.content });
      else if (event.type === 'message') {
        const message = event.content as ChatMessage | null;
        if (
          !message ||
          !['human', 'ai', 'tool'].includes(message.type) ||
          typeof message.content !== 'string'
        )
          throw new Error('Invalid chat message.');
        this.receive({ type: 'message', content: message });
      }
    }
  }
}
