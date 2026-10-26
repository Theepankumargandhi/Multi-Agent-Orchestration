# CiteRight web UI

React + TypeScript single-page app for CiteRight. It talks to the FastAPI service through a
same-origin `/api` proxy, so no provider key or token ever reaches browser storage.

```bash
npm ci --ignore-scripts
npm run dev        # http://127.0.0.1:5173, proxies /api to http://127.0.0.1:8000
npm test           # UI and transport tests with recorded responses
npm run build      # typecheck and production build
```

Point the dev proxy at another backend with `CITERIGHT_API_URL=http://host:port npm run dev`.

Session tokens live in memory only; refreshing the page signs you out. In Docker the built app is
served by nginx (`docker/Dockerfile.web`), which proxies `/api/` to the service.
