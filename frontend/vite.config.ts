import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Only the server-side proxy reads this variable. Never prefix credentials with VITE_.
const backend = process.env.WORKBENCH_API_URL ?? 'http://127.0.0.1:8000';
const target = new URL(backend);
if (
  !['http:', 'https:'].includes(target.protocol) ||
  target.username ||
  target.password ||
  target.search ||
  target.hash ||
  target.pathname !== '/'
) {
  throw new Error('WORKBENCH_API_URL must be an HTTP(S) origin without credentials or a path');
}

export default defineConfig({
  plugins: [react()],
  envDir: false,
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: target.origin,
        changeOrigin: false,
        rewrite: (path) => path.replace(/^\/api(?=\/|$)/, ''),
      },
    },
  },
  preview: { host: '127.0.0.1', port: 4173, strictPort: true },
  test: { environment: 'jsdom', setupFiles: ['./src/test/setup.ts'], clearMocks: true },
});
