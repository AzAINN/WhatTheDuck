# React Frontend (Compact Dashboard)

A React + Vite dashboard that keeps all graphs visible without scrolling on desktop, while controls live in a slim sidebar. Charts are driven by Recharts and gracefully fall back to mock data if the backend API is unavailable.

## Quick start

```bash
cd portfolio_frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

The dev server proxies `/api/*` to `http://localhost:8000` (FastAPI sample). Adjust `vite.config.ts` if your backend runs elsewhere.

## Design notes
- Full-height layout: header + split grid (controls left, charts right) so graphs stay in view on typical laptop screens.
- Four chart tiles share a fixed grid; metrics sit above them without pushing content below the fold.
- Sidebar scrolls independently, keeping visualization region stable.
- Mock data generator keeps the UI populated; real responses are mapped when `/api/optimize` returns data.
