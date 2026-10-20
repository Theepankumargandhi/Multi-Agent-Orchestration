import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
Object.defineProperty(Element.prototype, 'scrollIntoView', {
  value: () => undefined,
  configurable: true,
});
