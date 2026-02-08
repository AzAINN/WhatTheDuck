import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import clsx from 'clsx';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Metric = {
  label: string;
  value: string;
  detail?: string;
  tone?: 'good' | 'warn' | 'bad';
};

type Allocation = { name: string; value: number };
type RiskSlice = { name: string; value: number };
type DistributionPoint = { pnl: number; density: number };
type VaRComparisonPoint = {
  confidence: string;
  classical: number;
  quantum: number | null;
};

type VarResult = {
  var_value: number;   // positive fraction: 0.023 = 2.3% loss
  cvar_value: number;  // positive fraction: always >= var_value
  ci_lower?: number;
  ci_upper?: number;
  method?: string;
  metadata?: Record<string, unknown>;
};

type VaRPair = {
  classical: VarResult | null;
  quantum: VarResult | null;
};

type Analysis = {
  solver?: string;
  device?: string;
  qihd_time?: number;
  refinement_time?: number;
  total_time?: number;
  active_assets?: number;
  active_positions?: { asset: string; weight: number }[];
  n_shots?: number;
  n_steps?: number;
  n_samples?: number;
  n_assets?: number;
  cardinality_requested?: number;
  max_weight_cap?: number;
  mode?: string;
  var_method?: string;
  ci?: string;
};

type Snapshot = {
  asOf: string;
  mode: 'cvar' | 'variance';
  metrics: Metric[];
  allocations: Allocation[];
  risk: RiskSlice[];
  distribution: DistributionPoint[];
  varComparison: VaRComparisonPoint[];
  analysis?: Analysis;
};

type BackendHealth = {
  status: string;
  qihd_available: boolean;
  backend_available: boolean;
  yfinance_available: boolean;
  classiq_available?: boolean;
} | null;

type Phase = 'idle' | 'optimizing' | 'optimized' | 'var_running' | 'var_done';

const palette = ['#00d4ff', '#00e68a', '#a855f7', '#ff9f43', '#3b82f6', '#f43f5e', '#facc15', '#14b8a6', '#e879f9'];

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** Format a dollar amount: $1.2K, $2.3M, etc. */
function fmtDollar(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

/** Format VaR as dollar + percentage: "$2.3K (-2.30%)" */
function fmtVaR(fraction: number, portfolioValue: number, horizon: number): string {
  const scaled = fraction * Math.sqrt(horizon);
  const dollars = scaled * portfolioValue;
  return `${fmtDollar(dollars)} (-${(scaled * 100).toFixed(2)}%)`;
}

/** Time horizon label */
function horizonLabel(days: number): string {
  if (days === 1) return '1-Day';
  if (days === 5) return '5-Day';
  if (days === 10) return '10-Day';
  if (days === 21) return '21-Day';
  return `${days}-Day`;
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------
function gaussian(mu: number, sigma: number, x: number) {
  const coeff = 1 / (sigma * Math.sqrt(2 * Math.PI));
  return coeff * Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2));
}

function makeSyntheticDistribution(mu = 0, sigma = 0.014): DistributionPoint[] {
  const pts: DistributionPoint[] = [];
  for (let i = -4; i <= 4; i += 0.2) {
    const x = mu + i * sigma;
    pts.push({ pnl: x, density: gaussian(mu, sigma, x) });
  }
  return pts;
}

function parseCsv(text: string) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (!lines.length) return { error: 'CSV is empty.' };

  const split = (line: string) => line.split(/[;,\t]/).map((c) => c.trim());
  const first = split(lines[0]);
  const firstIsNumeric = first.every((c) => c === '' || !Number.isNaN(Number(c)));

  let names: string[] = [];
  let startIdx = 0;
  if (!firstIsNumeric) {
    names = first.map((c, i) => c || `Asset ${i + 1}`);
    startIdx = 1;
  }

  const rows: number[][] = [];
  for (let i = startIdx; i < lines.length; i++) {
    const cells = split(lines[i]);
    const nums = cells.map((c) => Number(c)).filter((n) => !Number.isNaN(n));
    if (nums.length) rows.push(nums);
  }

  const colCount = names.length || Math.max(0, ...rows.map((r) => r.length));
  if (!colCount || !rows.length) return { error: 'No numeric data found in CSV.' };
  if (!names.length) names = Array.from({ length: colCount }, (_, i) => `Asset ${i + 1}`);

  const data = rows.map((r) => {
    const copy = [...r];
    while (copy.length < colCount) copy.push(0);
    return copy.slice(0, colCount);
  });

  return { data, names };
}

/** Placeholder snapshot for initial render. */
function makePlaceholderSnapshot(mode: 'cvar' | 'variance' = 'cvar'): Snapshot {
  return {
    asOf: new Date().toISOString(),
    mode,
    metrics: [
      { label: 'VaR 95', value: '--', detail: 'Run optimization' },
      { label: 'CVaR 95', value: '--', detail: 'Run optimization' },
      { label: 'Volatility', value: '--', detail: 'Run optimization' },
      { label: 'Sharpe', value: '--', detail: 'Run optimization' },
      { label: 'Expected Return', value: '--', detail: 'Run optimization' },
      { label: 'Active Assets', value: '--', detail: 'Run optimization' },
      { label: 'VaR CI', value: '--', detail: 'Run optimization' },
    ],
    allocations: [],
    risk: [],
    distribution: makeSyntheticDistribution(),
    varComparison: [],
    analysis: undefined,
  };
}

/**
 * Map backend optimize response + VaR results into a Snapshot.
 * All VaR/CVaR values are POSITIVE fractions representing loss magnitude.
 */
function mapApiSnapshot(
  api: any,
  varRes: VarResult | null | undefined,
  varPair: VaRPair | null | undefined,
  portfolioValue: number,
  varHorizon: number,
): Snapshot {
  // Build allocations from active_positions metadata (has correct asset→weight mapping)
  // Fallback to weights + selected_assets if metadata unavailable
  const activePositions: { asset: string; weight: number }[] = api?.metadata?.active_positions ?? [];
  const allocations: Allocation[] = [];
  if (activePositions.length > 0) {
    for (const pos of activePositions) {
      if (pos.weight > 0.001) {
        allocations.push({
          name: pos.asset,
          value: parseFloat((pos.weight * 100).toFixed(2)),
        });
      }
    }
  } else {
    const weights: number[] = api?.weights ?? [];
    const names: string[] = api?.selected_assets ?? [];
    let nameIdx = 0;
    for (let i = 0; i < weights.length; i++) {
      if (weights[i] > 0.001) {
        allocations.push({
          name: names[nameIdx] ?? `Asset ${i + 1}`,
          value: parseFloat((weights[i] * 100).toFixed(2)),
        });
        nameIdx++;
      }
    }
  }
  allocations.sort((a, b) => b.value - a.value);

  const hasVaR = !!varRes;
  // VaR/CVaR as positive fractions (e.g. 0.023 = 2.3% loss)
  const var95 = varRes?.var_value ?? 0;
  const cvar95 = varRes?.cvar_value ?? 0;
  const vol = api?.volatility ?? 0;  // daily, fraction
  const sharpe = api?.sharpe_ratio ?? 0;
  const expectedReturn = api?.expected_return ?? 0;  // daily, fraction
  const hLabel = horizonLabel(varHorizon);
  const sqrtH = Math.sqrt(varHorizon);

  const ciText =
    varRes?.ci_lower != null && varRes?.ci_upper != null
      ? `${fmtDollar(varRes.ci_lower * sqrtH * portfolioValue)} -- ${fmtDollar(varRes.ci_upper * sqrtH * portfolioValue)}`
      : hasVaR ? 'Not available' : '--';

  // VaR comparison: pull multi-confidence values from metadata
  const classicalMeta = varPair?.classical?.metadata ?? varRes?.metadata ?? {};
  const quantumMeta = varPair?.quantum?.metadata;

  let varComparison: VaRComparisonPoint[] = [];
  if (hasVaR) {
    const cVar90 = (classicalMeta.var_90 as number | undefined) ?? var95 * 0.78;
    const cVar95 = (classicalMeta.var_95 as number | undefined) ?? var95;
    const cVar99 = (classicalMeta.var_99 as number | undefined) ?? var95 * 1.35;

    const qVar90 = quantumMeta ? ((quantumMeta.var_90 as number | undefined) ?? (varPair!.quantum!.var_value * 0.78)) : null;
    const qVar95 = varPair?.quantum ? varPair.quantum.var_value : null;
    const qVar99 = quantumMeta ? ((quantumMeta.var_99 as number | undefined) ?? (varPair!.quantum!.var_value * 1.35)) : null;

    // All values in dollars, scaled by time horizon
    varComparison = [
      { confidence: '90%', classical: cVar90 * sqrtH * portfolioValue, quantum: qVar90 != null ? qVar90 * sqrtH * portfolioValue : null },
      { confidence: '95%', classical: cVar95 * sqrtH * portfolioValue, quantum: qVar95 != null ? qVar95 * sqrtH * portfolioValue : null },
      { confidence: '99%', classical: cVar99 * sqrtH * portfolioValue, quantum: qVar99 != null ? qVar99 * sqrtH * portfolioValue : null },
    ];
  }

  // Risk metrics in dollars (VaR/CVaR only — Vol is separate)
  const risk: RiskSlice[] = [];
  if (hasVaR) {
    risk.push({ name: `VaR (${hLabel})`, value: var95 * sqrtH * portfolioValue });
    risk.push({ name: `CVaR (${hLabel})`, value: cvar95 * sqrtH * portfolioValue });
  }

  return {
    asOf: new Date().toISOString(),
    mode: api?.mode === 'variance' ? 'variance' : 'cvar',
    metrics: [
      {
        label: `VaR ${hLabel}`,
        value: hasVaR ? fmtVaR(var95, portfolioValue, varHorizon) : '--',
        detail: hasVaR ? (varRes?.method ?? 'Backend') : 'Calculate VaR',
        tone: hasVaR ? 'warn' : undefined,
      },
      {
        label: `CVaR ${hLabel}`,
        value: hasVaR ? fmtVaR(cvar95, portfolioValue, varHorizon) : '--',
        detail: hasVaR ? 'Expected tail loss' : 'Calculate VaR',
        tone: hasVaR ? 'bad' : undefined,
      },
      {
        label: 'Volatility',
        value: `${(vol * 100).toFixed(2)}%`,
        detail: `Daily std | Ann. ~${(vol * Math.sqrt(252) * 100).toFixed(1)}%`,
        tone: 'good',
      },
      { label: 'Sharpe', value: `${sharpe.toFixed(3)}`, detail: 'Return / Risk', tone: sharpe > 0 ? 'good' : 'bad' },
      {
        label: 'Expected Return',
        value: `${(expectedReturn * 100).toFixed(3)}%`,
        detail: `Daily | Ann. ~${(expectedReturn * 252 * 100).toFixed(1)}%`,
        tone: expectedReturn > 0 ? 'good' : 'bad',
      },
      { label: 'Active Assets', value: `${api?.metadata?.active_assets ?? allocations.length}`, detail: `of ${api?.metadata?.n_assets ?? '?'}` },
      { label: 'VaR CI', value: ciText, detail: hasVaR ? (varRes?.method ?? 'Classical MC') : 'Calculate VaR' },
    ],
    allocations,
    risk,
    distribution: makeSyntheticDistribution(-(var95 || 0.01), Math.max(0.005, vol || 0.013)),
    varComparison,
    analysis: {
      solver: api?.metadata?.solver,
      device: api?.metadata?.device,
      qihd_time: api?.metadata?.qihd_time,
      refinement_time: api?.metadata?.refinement_time,
      total_time: api?.metadata?.total_time,
      active_assets: api?.metadata?.active_assets,
      active_positions: api?.metadata?.active_positions,
      n_shots: api?.metadata?.n_shots,
      n_steps: api?.metadata?.n_steps,
      n_samples: api?.metadata?.n_samples,
      n_assets: api?.metadata?.n_assets,
      cardinality_requested: api?.metadata?.cardinality_requested,
      max_weight_cap: api?.metadata?.max_weight,
      mode: api?.metadata?.mode ?? api?.mode,
      var_method: hasVaR ? (varRes?.method ?? 'Classical MC') : '--',
      ci: ciText,
    },
  };
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------
const ChartCard = ({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) => (
  <div className="chart-card">
    <div className="chart-head">
      <div>
        <p className="chart-title">{title}</p>
        {subtitle && <p className="chart-sub">{subtitle}</p>}
      </div>
    </div>
    <div className="chart-body">{children}</div>
  </div>
);

const MetricCard = ({ metric }: { metric: Metric }) => (
  <div className="metric-card">
    <span className="metric-label">{metric.label}</span>
    <span className="metric-value">{metric.value}</span>
    {metric.detail && <span className="metric-foot">{metric.detail}</span>}
  </div>
);

// ---------------------------------------------------------------------------
// Main UI
// ---------------------------------------------------------------------------
function App() {
  const [mode, setMode] = useState<'cvar' | 'variance'>('cvar');
  const [snapshot, setSnapshot] = useState<Snapshot>(() => makePlaceholderSnapshot('cvar'));
  const [loading, setLoading] = useState(false);
  const [tickers, setTickers] = useState('AAPL, MSFT, NVDA, AMZN, GOOGL');
  const [cardinality, setCardinality] = useState(8);
  const [maxWeight, setMaxWeight] = useState(0.25);
  const [conf, setConf] = useState(0.95);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [isMobile, setIsMobile] = useState(false);
  const [carouselIndex, setCarouselIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [backendHealth, setBackendHealth] = useState<BackendHealth>(null);
  const [varPair, setVarPair] = useState<VaRPair>({ classical: null, quantum: null });

  // Two-phase flow state
  const [phase, setPhase] = useState<Phase>('idle');
  const [optimizeResult, setOptimizeResult] = useState<any>(null);
  const [savedReturns, setSavedReturns] = useState<number[][] | null>(null);

  // IQAE live output
  const [iqaeLog, setIqaeLog] = useState<string[]>([]);
  const [iqaePhase, setIqaePhase] = useState<string>('');

  // Real distribution data (from /api/distribution)
  const [realDistribution, setRealDistribution] = useState<DistributionPoint[] | null>(null);

  // Historical lookback
  const [lookbackDays, setLookbackDays] = useState(252);

  // Portfolio value & time horizon
  const [portfolioValue, setPortfolioValue] = useState(100000);
  const [varHorizon, setVarHorizon] = useState(1);

  // Interactive VaR lines - which confidence level to show on distribution (null = none)
  const [selectedConfidence, setSelectedConfidence] = useState<string | null>(null);

  const consoleEndRef = useRef<HTMLDivElement>(null);

  const allocationData = useMemo(() => snapshot.allocations, [snapshot]);

  // Auto-scroll IQAE console
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [iqaeLog]);

  // Check backend health on mount
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setBackendHealth(data))
      .catch(() => setBackendHealth(null));
  }, []);

  // Responsive breakpoint
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1180px)');
    const handler = () => setIsMobile(mq.matches);
    handler();
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Recompute snapshot when portfolioValue or varHorizon changes (if we have results)
  useEffect(() => {
    if (optimizeResult && phase !== 'idle' && phase !== 'optimizing') {
      const primaryVar = varPair.classical;
      setSnapshot(mapApiSnapshot(optimizeResult, primaryVar, varPair, portfolioValue, varHorizon));
    }
  }, [portfolioValue, varHorizon]); // eslint-disable-line react-hooks/exhaustive-deps

  // Use real distribution when available, else snapshot's synthetic one
  const distributionData = realDistribution ?? snapshot.distribution;
  const sqrtH = Math.sqrt(varHorizon);

  const runOptimization = async () => {
    setLoading(true);
    setPhase('optimizing');
    setError(null);
    setCsvError(null);
    setVarPair({ classical: null, quantum: null });
    setIqaeLog([]);
    setIqaePhase('');
    setRealDistribution(null);

    let returnsPayload: number[][] | undefined;
    let assetNames: string[] | undefined;

    // Step 1: Get returns data
    if (csvFile) {
      try {
        const text = await csvFile.text();
        const parsed = parseCsv(text);
        if (parsed.error) throw new Error(parsed.error);
        returnsPayload = parsed.data;
        assetNames = parsed.names;
      } catch (err: any) {
        setCsvError(err.message ?? 'Failed to parse CSV');
        setLoading(false);
        setPhase('idle');
        return;
      }
    } else {
      try {
        const tickerList = tickers.split(',').map((t) => t.trim()).filter(Boolean);
        if (!tickerList.length) {
          setError('Enter at least one ticker symbol.');
          setLoading(false);
          setPhase('idle');
          return;
        }
        const tickerRes = await fetch('/api/ticker-data', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tickers: tickerList, lookback_days: lookbackDays }),
        });
        if (!tickerRes.ok) {
          const detail = await tickerRes.json().catch(() => ({}));
          throw new Error(detail.detail || `Ticker fetch failed (${tickerRes.status})`);
        }
        const tickerData = await tickerRes.json();
        returnsPayload = tickerData.returns;
        assetNames = tickerData.asset_names;
      } catch (err: any) {
        setError(err.message ?? 'Failed to fetch ticker data. Is the backend running?');
        setLoading(false);
        setPhase('idle');
        return;
      }
    }

    // Step 2: Optimize
    const body = {
      tickers: tickers.split(',').map((t) => t.trim()).filter(Boolean),
      mode,
      cardinality,
      max_weight: maxWeight,
      confidence_level: conf,
      returns: returnsPayload,
      asset_names: assetNames,
    };

    try {
      const res = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Optimization failed (${res.status})`);
      }
      const api = await res.json();

      // Store results for Phase 2
      const returnsForVar = returnsPayload ?? api?.returns;
      const optResult = { ...api, confidence_level: conf, mode };
      setOptimizeResult(optResult);
      setSavedReturns(returnsForVar ?? null);

      // Fetch real distribution
      if (returnsForVar && Array.isArray(api?.weights)) {
        try {
          const distRes = await fetch('/api/distribution', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ weights: api.weights, returns: returnsForVar }),
          });
          if (distRes.ok) {
            const distData = await distRes.json();
            setRealDistribution(distData.points);
          }
        } catch {
          // Non-fatal
        }
      }

      // Build snapshot without VaR
      setSnapshot(mapApiSnapshot(optResult, null, null, portfolioValue, varHorizon));
      setPhase('optimized');
    } catch (err: any) {
      setError(err.message ?? 'Optimization failed. Is the backend running?');
      setPhase('idle');
    } finally {
      setLoading(false);
    }
  };

  const runVaR = async () => {
    if (!optimizeResult || !savedReturns) return;

    setPhase('var_running');
    setIqaeLog([]);
    setIqaePhase('');
    setError(null);

    try {
      const response = await fetch('/api/var-stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          weights: optimizeResult.weights,
          returns: savedReturns,
          confidence_level: conf,
        }),
      });

      if (!response.ok) {
        throw new Error(`VaR stream failed (${response.status})`);
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const newPair: VaRPair = { classical: null, quantum: null };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        let currentEvent = '';
        let currentData = '';

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            currentData = line.slice(6);

            if (currentEvent === 'phase') {
              try {
                setIqaePhase(JSON.parse(currentData));
              } catch {
                setIqaePhase(currentData);
              }
            } else if (currentEvent === 'log') {
              try {
                const logLine = JSON.parse(currentData);
                setIqaeLog((prev) => [...prev, logLine]);
              } catch {
                setIqaeLog((prev) => [...prev, currentData]);
              }
            } else if (currentEvent === 'classical_result') {
              try {
                const result = JSON.parse(currentData);
                newPair.classical = {
                  var_value: result.var_value,
                  cvar_value: result.cvar_value,
                  ci_lower: result.ci_lower,
                  ci_upper: result.ci_upper,
                  method: result.method ?? 'classical_mc',
                  metadata: result,
                };
                setVarPair({ ...newPair });
                setSnapshot(mapApiSnapshot(optimizeResult, newPair.classical, { ...newPair }, portfolioValue, varHorizon));
              } catch {
                // skip
              }
            } else if (currentEvent === 'quantum_result') {
              try {
                const result = JSON.parse(currentData);
                newPair.quantum = {
                  var_value: result.var_value,
                  cvar_value: result.cvar_value,
                  ci_lower: result.ci_lower,
                  ci_upper: result.ci_upper,
                  method: result.method ?? 'iqae',
                  metadata: result,
                };
                setVarPair({ ...newPair });
                setSnapshot(mapApiSnapshot(optimizeResult, newPair.classical, { ...newPair }, portfolioValue, varHorizon));
              } catch {
                // skip
              }
            } else if (currentEvent === 'error') {
              try {
                const msg = JSON.parse(currentData);
                setError(typeof msg === 'string' ? msg : JSON.stringify(msg));
              } catch {
                setError(currentData);
              }
            }

            currentEvent = '';
            currentData = '';
          }
        }
      }

      setVarPair(newPair);
      setSnapshot(mapApiSnapshot(optimizeResult, newPair.classical, newPair, portfolioValue, varHorizon));
      setPhase('var_done');
    } catch (err: any) {
      setError(err.message ?? 'VaR computation failed.');
      setPhase('optimized');
    }
  };

  // Compute VaR reference line positions on the P&L distribution
  // Distribution x-axis is fractional P&L (e.g. -0.023 for 2.3% loss)
  // VaR values are positive fractions, so VaR line is at -var_value on PnL axis
  const classicalVarPnl = varPair.classical ? -varPair.classical.var_value : null;
  const classicalCvarPnl = varPair.classical ? -varPair.classical.cvar_value : null;
  const quantumVarPnl = varPair.quantum ? -varPair.quantum.var_value : null;

  // Get VaR values for each confidence level from comparison data
  const varLines = useMemo(() => {
    if (snapshot.varComparison.length === 0) return [];
    const lines: Array<{ confidence: string; classical: number; quantum: number | null }> = [];
    snapshot.varComparison.forEach((point) => {
      lines.push({
        confidence: point.confidence,
        classical: -point.classical / portfolioValue / sqrtH,  // Convert back to fractional PnL
        quantum: point.quantum != null ? -point.quantum / portfolioValue / sqrtH : null,
      });
    });
    return lines;
  }, [snapshot.varComparison, portfolioValue, sqrtH]);

  const toggleConfidence = (conf: string) => {
    setSelectedConfidence((prev) => (prev === conf ? null : conf));
  };

  // Compute the tail cutoff for the selected confidence on the distribution
  // The tail % is (1 - confidence): 90% -> 10% tail, 95% -> 5% tail, 99% -> 1% tail
  const selectedVarLine = useMemo(() => {
    if (!selectedConfidence || varLines.length === 0) return null;
    return varLines.find((l) => l.confidence === selectedConfidence) ?? null;
  }, [selectedConfidence, varLines]);

  // Get the leftmost PnL value in distribution for shading the tail area
  const distMinPnl = useMemo(() => {
    if (!distributionData.length) return 0;
    return Math.min(...distributionData.map((d) => d.pnl));
  }, [distributionData]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">P</div>
          <div>
            <div>Portfolio Optimizer</div>
            <small style={{ color: '#97a7c0', display: 'block', marginTop: 2 }}>
              Portfolio Optimization (CVaR & Variance) using Quantum Hamiltonian Descent, and VaR calculation on Classical and Quantum
            </small>
            <small style={{ color: '#7ce7ff', display: 'block', marginTop: 2, fontSize: 11 }}>
              Author: Azain Khalid, Computer Science at Purdue University | Skills: Quantum Optimization, Quantum Software Stack, Machine Learning
              {backendHealth ? (
                <span className="status-connected" title="Backend connected" />
              ) : (
                <span className="status-disconnected" title="Backend offline" />
              )}
            </small>
          </div>
        </div>
        <div className="top-actions">
          <div className="badge">As of {new Date(snapshot.asOf).toLocaleString()}</div>
          <button className="button secondary" onClick={() => {
            setSnapshot(makePlaceholderSnapshot(mode));
            setError(null);
            setVarPair({ classical: null, quantum: null });
            setPhase('idle');
            setOptimizeResult(null);
            setSavedReturns(null);
            setRealDistribution(null);
            setIqaeLog([]);
            setIqaePhase('');
          }}>
            Reset view
          </button>
          <button className="button" onClick={runOptimization} disabled={loading || phase === 'var_running'}>
            {phase === 'optimizing' ? 'Optimizing...' : 'Run Optimization'}
          </button>
          {(phase === 'optimized' || phase === 'var_done') && (
            <button className="button quantum" onClick={runVaR}>
              Calculate VaR
            </button>
          )}
          {phase === 'var_running' && (
            <button className="button quantum" disabled>
              Computing VaR...
            </button>
          )}
        </div>
      </header>

      {/* Progress bar */}
      {(phase === 'optimizing' || phase === 'var_running') && (
        <div className="progress-bar">
          <div className="progress-pulse" />
          <span>{phase === 'optimizing' ? 'Optimizing portfolio...' : iqaePhase || 'Computing VaR...'}</span>
        </div>
      )}

      {backendHealth === null && (
        <div className="warn-banner">
          Backend not connected. Start with: <code>PYTHONPATH=OpenPhiSolve:. python portfolio_backend/server.py</code>
        </div>
      )}
      {backendHealth && !backendHealth.qihd_available && (
        <div className="warn-banner">
          QIHD solver not available -- using CVXPY fallback. Install phisolve for GPU optimization.
        </div>
      )}
      {error && (
        <div className="error-banner" onClick={() => setError(null)}>
          {error}
          <span style={{ float: 'right', cursor: 'pointer', opacity: 0.7 }}>dismiss</span>
        </div>
      )}

      <main className="layout">
        <aside className="panel sidebar">
          <div>
            <p className="section-title">Data Source</p>
            <p className="section-note">
              Enter tickers for live yfinance data or upload a returns CSV.
            </p>
            <div className="field">
              <label>Tickers</label>
              <input
                className="input"
                value={tickers}
                onChange={(e) => setTickers(e.target.value)}
                placeholder="AAPL, MSFT, NVDA"
              />
            </div>
            <div className="field">
              <label>Upload CSV (returns)</label>
              <input
                className="upload"
                type="file"
                accept=".csv"
                onChange={(e) => setCsvFile(e.target.files?.[0] ?? null)}
              />
              {csvError && <small style={{ color: '#ff9f66' }}>{csvError}</small>}
            </div>
            <div className="field">
              <label>Portfolio Value (USD)</label>
              <div className="currency-input">
                <span className="currency-prefix">$</span>
                <input
                  className="input currency-field"
                  type="number"
                  min={1000}
                  step={10000}
                  value={portfolioValue}
                  onChange={(e) => setPortfolioValue(Math.max(1, Number(e.target.value) || 0))}
                />
              </div>
            </div>
            <div className="field">
              <label>Lookback Period</label>
              <select
                className="select"
                value={lookbackDays}
                onChange={(e) => setLookbackDays(Number(e.target.value))}
              >
                <option value={30}>30 days</option>
                <option value={63}>63 days (~3 months)</option>
                <option value={126}>126 days (~6 months)</option>
                <option value={252}>252 days (~1 year)</option>
                <option value={500}>500 days (~2 years)</option>
                <option value={750}>750 days (~3 years)</option>
              </select>
              {lookbackDays < 63 && <small className="field-warn">Small sample size may produce unreliable results</small>}
            </div>
            <div className="field">
              <label>VaR Time Horizon</label>
              <select
                className="select"
                value={varHorizon}
                onChange={(e) => setVarHorizon(Number(e.target.value))}
              >
                <option value={1}>1-Day VaR</option>
                <option value={5}>5-Day VaR (1 week)</option>
                <option value={10}>10-Day VaR (Basel III)</option>
                <option value={21}>21-Day VaR (1 month)</option>
              </select>
            </div>
            <div className="field">
              <label>Mode</label>
              <div className="toggle-row">
                {['cvar', 'variance'].map((m) => (
                  <button key={m} className={clsx('pill', { active: mode === m })} onClick={() => setMode(m as any)}>
                    {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div>
            <p className="section-title">Controls</p>
            <div className="field">
              <label>Cardinality: {cardinality}</label>
              <div className="slider-row">
                <input
                  type="range"
                  min={4}
                  max={14}
                  step={1}
                  value={cardinality}
                  onChange={(e) => setCardinality(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="field">
              <label>Max Weight: {(maxWeight * 100).toFixed(0)}%</label>
              <div className="slider-row">
                <input
                  type="range"
                  min={0.05}
                  max={0.4}
                  step={0.01}
                  value={maxWeight}
                  onChange={(e) => setMaxWeight(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="field">
              <label>Confidence: {(conf * 100).toFixed(0)}%</label>
              <div className="slider-row">
                <input
                  type="range"
                  min={0.8}
                  max={0.99}
                  step={0.01}
                  value={conf}
                  onChange={(e) => setConf(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="status-row">
              <span className="status-dot" />
              {backendHealth
                ? `Server: ${backendHealth.qihd_available ? 'QIHD' : 'CVXPY'} | yfinance: ${backendHealth.yfinance_available ? 'OK' : 'N/A'}${backendHealth.classiq_available ? ' | Classiq: OK' : ''}`
                : 'Backend offline'}
            </div>
          </div>
        </aside>

        <section className="dashboard">
          <div className="metric-grid">
            {snapshot.metrics.map((m) => (
              <MetricCard key={m.label} metric={m} />
            ))}
          </div>

          <div className={clsx('chart-grid', { carousel: isMobile })}>
            {(!isMobile || carouselIndex === 0) && (
              <ChartCard title="Allocation" subtitle={allocationData.length ? `${allocationData.length} positions · ${snapshot.analysis?.solver?.toUpperCase() ?? ''} ${snapshot.analysis?.total_time ? `(${snapshot.analysis.total_time.toFixed(1)}s)` : ''}`.trim() : 'Run optimization to see weights'}>
                <ResponsiveContainer>
                  <PieChart>
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0];
                        const name = d.name ?? '';
                        const pct = typeof d.value === 'number' ? d.value : 0;
                        const dollars = (pct / 100) * portfolioValue;
                        return (
                          <div style={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 8, padding: '10px 14px', boxShadow: '0 4px 12px rgba(0,0,0,0.4)' }}>
                            <p style={{ color: '#f9fafb', fontWeight: 600, fontSize: 14, margin: 0 }}>{name}</p>
                            <p style={{ color: '#00d4ff', fontSize: 13, margin: '4px 0 0' }}>{pct.toFixed(2)}% · {fmtDollar(dollars)}</p>
                          </div>
                        );
                      }}
                    />
                    <Pie
                      data={allocationData.length ? allocationData : [{ name: 'None', value: 100 }]}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={42}
                      outerRadius={100}
                      paddingAngle={2}
                      stroke="#0d1117"
                      strokeWidth={2}
                    >
                      {(allocationData.length ? allocationData : [{ name: 'None', value: 100 }]).map((entry, index) => (
                        <Cell key={`cell-${entry.name}`} fill={allocationData.length ? palette[index % palette.length] : '#1f2b3c'} />
                      ))}
                    </Pie>
                    <Legend verticalAlign="bottom" height={28} wrapperStyle={{ color: '#97a7c0', fontSize: 12 }} />
                  </PieChart>
                </ResponsiveContainer>
                {allocationData.length > 0 && snapshot.analysis && (
                  <div style={{ display: 'flex', justifyContent: 'center', gap: 16, padding: '6px 0 2px', flexWrap: 'wrap' }}>
                    {snapshot.analysis.device && (
                      <span style={{ fontSize: 11, color: '#6b7a90', letterSpacing: 0.5 }}>{snapshot.analysis.device.toUpperCase()}</span>
                    )}
                    {snapshot.analysis.mode && (
                      <span style={{ fontSize: 11, color: '#6b7a90', letterSpacing: 0.5 }}>{snapshot.analysis.mode.toUpperCase()}</span>
                    )}
                    {snapshot.analysis.cardinality_requested != null && snapshot.analysis.n_assets != null && (
                      <span style={{ fontSize: 11, color: '#6b7a90', letterSpacing: 0.5 }}>K={snapshot.analysis.cardinality_requested}/{snapshot.analysis.n_assets}</span>
                    )}
                    {snapshot.analysis.max_weight_cap != null && (
                      <span style={{ fontSize: 11, color: '#6b7a90', letterSpacing: 0.5 }}>Cap {(snapshot.analysis.max_weight_cap * 100).toFixed(0)}%</span>
                    )}
                    {snapshot.analysis.n_shots != null && (
                      <span style={{ fontSize: 11, color: '#6b7a90', letterSpacing: 0.5 }}>{snapshot.analysis.n_shots} shots</span>
                    )}
                  </div>
                )}
              </ChartCard>
            )}

            {(!isMobile || carouselIndex === 1) && (
              <ChartCard title={`VaR / CVaR (${horizonLabel(varHorizon)})`} subtitle={snapshot.risk.length ? `On ${fmtDollar(portfolioValue)} portfolio` : 'Calculate VaR to see risk metrics'}>
                {snapshot.risk.length > 0 ? (
                  <ResponsiveContainer>
                    <BarChart data={snapshot.risk} barCategoryGap="20%">
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3c" />
                      <XAxis dataKey="name" stroke="#97a7c0" tickLine={false} />
                      <YAxis stroke="#97a7c0" tickFormatter={(v) => fmtDollar(v)} />
                      <Tooltip formatter={(v: number) => fmtDollar(v)} contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #2a2a4a' }} itemStyle={{ color: '#e0e0e0' }} labelStyle={{ color: '#c0c0c0' }} />
                      <Bar dataKey="value" name="Risk ($)" fill="#7ce7ff" radius={[8, 8, 4, 4]}>
                        <Cell fill="#fbbf24" />
                        <Cell fill="#ff6b6b" />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart-placeholder">Calculate VaR to see risk metrics</div>
                )}
              </ChartCard>
            )}

            {(!isMobile || carouselIndex === 2) && (
              <ChartCard
                title="P&L Distribution"
                subtitle={phase === 'var_done'
                  ? (selectedConfidence
                    ? `Showing ${selectedConfidence} VaR — tail shaded`
                    : 'Click a confidence bar to show VaR lines')
                  : 'Daily portfolio P&L'}
              >
                <ResponsiveContainer>
                  <AreaChart data={distributionData} margin={{ left: -14, right: 0, top: 10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="lossFill" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="5%" stopColor="#7ce7ff" stopOpacity={0.45} />
                        <stop offset="95%" stopColor="#7ce7ff" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3c" />
                    <XAxis
                      dataKey="pnl"
                      type="number"
                      domain={['auto', 'auto']}
                      stroke="#97a7c0"
                      tickFormatter={(v) => typeof v === 'number' ? fmtDollar(v * portfolioValue) : String(v)}
                    />
                    <YAxis stroke="#97a7c0" hide />
                    <Tooltip
                      formatter={(v: number) => v.toFixed(4)}
                      labelFormatter={(pnl) => {
                        const pnlNum = typeof pnl === 'number' ? pnl : parseFloat(String(pnl));
                        return `P&L: ${fmtDollar(pnlNum * portfolioValue)} (${(pnlNum * 100).toFixed(2)}%)`;
                      }}
                      contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #2a2a4a' }}
                      itemStyle={{ color: '#e0e0e0' }}
                      labelStyle={{ color: '#c0c0c0' }}
                    />
                    {/* Tail shading: red area from left edge to VaR line */}
                    {phase === 'var_done' && selectedVarLine && (
                      <ReferenceArea
                        x1={distMinPnl}
                        x2={selectedVarLine.classical}
                        fill="#ff6b6b"
                        fillOpacity={0.18}
                        strokeOpacity={0}
                      />
                    )}
                    {/* Classical VaR line */}
                    {phase === 'var_done' && selectedVarLine && (
                      <ReferenceLine
                        x={selectedVarLine.classical}
                        stroke="#fbbf24"
                        strokeDasharray="6 3"
                        strokeWidth={2}
                        label={{
                          value: `Classical ${selectedConfidence} VaR`,
                          position: 'insideTopRight',
                          fill: '#fbbf24',
                          fontSize: 11,
                        }}
                      />
                    )}
                    {/* Quantum VaR line */}
                    {phase === 'var_done' && selectedVarLine && selectedVarLine.quantum != null && (
                      <ReferenceLine
                        x={selectedVarLine.quantum}
                        stroke="#a78bfa"
                        strokeDasharray="4 3"
                        strokeWidth={2}
                        label={{
                          value: `Quantum ${selectedConfidence} VaR`,
                          position: 'insideTopLeft',
                          fill: '#a78bfa',
                          fontSize: 11,
                        }}
                      />
                    )}
                    <Area type="monotone" dataKey="density" stroke="#7ce7ff" fill="url(#lossFill)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {(!isMobile || carouselIndex === 3) && (
              <ChartCard title="Classical vs Quantum VaR" subtitle={snapshot.varComparison.length > 0 ? `${horizonLabel(varHorizon)} horizon — click a bar to show VaR on distribution` : 'Side-by-side at confidence levels'}>
                {snapshot.varComparison.length > 0 ? (
                  <ResponsiveContainer>
                    <BarChart data={snapshot.varComparison} barCategoryGap="20%" style={{ cursor: 'pointer' }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2b3c" />
                      <XAxis dataKey="confidence" stroke="#97a7c0" tickLine={false} />
                      <YAxis stroke="#97a7c0" tickFormatter={(v) => fmtDollar(v)} />
                      <Tooltip formatter={(v: number) => fmtDollar(v)} contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #2a2a4a' }} itemStyle={{ color: '#e0e0e0' }} labelStyle={{ color: '#c0c0c0' }} />
                      <Bar dataKey="classical" name="Classical MC" fill="#7ce7ff" radius={[8, 8, 4, 4]} onClick={(data) => toggleConfidence(data.confidence)}>
                        {snapshot.varComparison.map((entry) => {
                          const isActive = selectedConfidence === entry.confidence;
                          return <Cell key={`cell-c-${entry.confidence}`} fillOpacity={isActive ? 1 : selectedConfidence ? 0.25 : 0.8} stroke={isActive ? '#fff' : 'none'} strokeWidth={isActive ? 2 : 0} />;
                        })}
                      </Bar>
                      <Bar dataKey="quantum" name="Quantum (IQAE)" fill="#a78bfa" radius={[8, 8, 4, 4]} onClick={(data) => toggleConfidence(data.confidence)}>
                        {snapshot.varComparison.map((entry) => {
                          const isActive = selectedConfidence === entry.confidence;
                          return <Cell key={`cell-q-${entry.confidence}`} fillOpacity={isActive ? 1 : selectedConfidence ? 0.25 : 0.8} stroke={isActive ? '#fff' : 'none'} strokeWidth={isActive ? 2 : 0} />;
                        })}
                      </Bar>
                      <Legend />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart-placeholder">Calculate VaR to see comparison</div>
                )}
              </ChartCard>
            )}

            {isMobile && (
              <div className="carousel-dots">
                {[0, 1, 2, 3].map((i) => (
                  <button key={i} className={clsx('dot', { active: carouselIndex === i })} onClick={() => setCarouselIndex(i)} />
                ))}
              </div>
            )}
          </div>
        </section>
      </main>

      <section className="analysis">
        {/* IQAE Console */}
        {iqaeLog.length > 0 && (
          <div className="iqae-console">
            <div className="console-header">
              <span>IQAE Live Output</span>
              {iqaePhase && <span className="console-phase">{iqaePhase}</span>}
            </div>
            <div className="console-body">
              {iqaeLog.map((line, i) => (
                <div key={i} className="console-line">{line}</div>
              ))}
              <div ref={consoleEndRef} />
            </div>
          </div>
        )}

        <div className="panel analysis-card">
          <div className="analysis-head">
            <div>
              <p className="section-title">Detailed Analysis</p>
              <p className="section-note">
                {snapshot.analysis
                  ? 'Results from backend optimization and VaR calculation.'
                  : 'Run optimization to see solver metadata and VaR details.'}
              </p>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {snapshot.analysis?.solver && (
                <div className="badge">{snapshot.analysis.solver.toUpperCase()}</div>
              )}
              <div className="badge">{(snapshot.analysis?.var_method ?? '--').toUpperCase()}</div>
            </div>
          </div>
          <div className="analysis-grid">
            <div>
              <p className="mini-label">Solver</p>
              <p className="mini-value">{snapshot.analysis?.solver ?? '--'}</p>
            </div>
            <div>
              <p className="mini-label">Device</p>
              <p className="mini-value">{(snapshot.analysis?.device ?? '--').toUpperCase()}</p>
            </div>
            <div>
              <p className="mini-label">QIHD Time</p>
              <p className="mini-value">{snapshot.analysis?.qihd_time ? `${snapshot.analysis.qihd_time.toFixed(2)}s` : '--'}</p>
            </div>
            <div>
              <p className="mini-label">Refinement Time</p>
              <p className="mini-value">{snapshot.analysis?.refinement_time ? `${snapshot.analysis.refinement_time.toFixed(2)}s` : '--'}</p>
            </div>
            <div>
              <p className="mini-label">Total Time</p>
              <p className="mini-value">{snapshot.analysis?.total_time ? `${snapshot.analysis.total_time.toFixed(2)}s` : '--'}</p>
            </div>
            <div>
              <p className="mini-label">Active Assets</p>
              <p className="mini-value">
                {snapshot.analysis?.active_assets != null
                  ? `${snapshot.analysis.active_assets} / ${snapshot.analysis.n_assets ?? '?'}`
                  : '--'}
              </p>
            </div>
            <div>
              <p className="mini-label">QIHD Shots / Steps</p>
              <p className="mini-value">
                {snapshot.analysis?.n_shots
                  ? `${snapshot.analysis.n_shots} / ${snapshot.analysis.n_steps ?? '?'}`
                  : '--'}
              </p>
            </div>
            <div>
              <p className="mini-label">VaR Method</p>
              <p className="mini-value">{snapshot.analysis?.var_method ?? '--'}</p>
            </div>
            <div>
              <p className="mini-label">VaR CI</p>
              <p className="mini-value">{snapshot.analysis?.ci ?? '--'}</p>
            </div>
            <div>
              <p className="mini-label">Mode</p>
              <p className="mini-value">{(snapshot.analysis?.mode ?? mode).toUpperCase()}</p>
            </div>
            <div>
              <p className="mini-label">Samples</p>
              <p className="mini-value">{snapshot.analysis?.n_samples ?? '--'}</p>
            </div>
            <div>
              <p className="mini-label">Max Weight Cap</p>
              <p className="mini-value">
                {snapshot.analysis?.max_weight_cap != null
                  ? `${(snapshot.analysis.max_weight_cap * 100).toFixed(0)}%`
                  : '--'}
              </p>
            </div>
          </div>

          {/* Active Positions */}
          {snapshot.analysis?.active_positions && snapshot.analysis.active_positions.length > 0 && (
            <div className="positions-table">
              <p className="mini-label" style={{ marginBottom: 8 }}>Active Positions</p>
              <div className="positions-grid">
                {snapshot.analysis.active_positions.map((pos) => (
                  <div key={pos.asset} className="position-row">
                    <span className="position-name">{pos.asset}</span>
                    <span className="position-weight">{(pos.weight * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quantum VaR details */}
          {varPair.quantum && (
            <div className="quantum-var-panel">
              <p className="mini-label">Quantum VaR (IQAE)</p>
              <div className="quantum-var-grid">
                <div>
                  <span className="qv-label">VaR</span>
                  <span className="qv-value">{fmtDollar(varPair.quantum.var_value * sqrtH * portfolioValue)}</span>
                  <span className="qv-sub">-{(varPair.quantum.var_value * sqrtH * 100).toFixed(2)}%</span>
                </div>
                <div>
                  <span className="qv-label">CVaR</span>
                  <span className="qv-value">{fmtDollar(varPair.quantum.cvar_value * sqrtH * portfolioValue)}</span>
                  <span className="qv-sub">-{(varPair.quantum.cvar_value * sqrtH * 100).toFixed(2)}%</span>
                </div>
                {varPair.quantum.metadata?.oracle_queries != null && (
                  <div>
                    <span className="qv-label">Oracle Queries</span>
                    <span className="qv-value">{(varPair.quantum.metadata.oracle_queries as number).toLocaleString()}</span>
                  </div>
                )}
                {varPair.quantum.metadata?.bisection_steps != null && (
                  <div>
                    <span className="qv-label">Bisection Steps</span>
                    <span className="qv-value">{varPair.quantum.metadata.bisection_steps as number}</span>
                  </div>
                )}
                {varPair.classical && (
                  <div>
                    <span className="qv-label">vs Classical</span>
                    <span className="qv-value">
                      {fmtDollar((varPair.quantum.var_value - varPair.classical.var_value) * sqrtH * portfolioValue)}
                    </span>
                    <span className="qv-sub">
                      {((varPair.quantum.var_value - varPair.classical.var_value) * 10000).toFixed(1)} bps
                    </span>
                  </div>
                )}
              </div>
              {varPair.quantum.method === 'iqae_simulated' && (
                <p className="qv-note">Classiq not installed — using simulated IQAE fallback.</p>
              )}
            </div>
          )}
        </div>
      </section>

      <footer className="footer">
        <div className="footer-content">
          <p className="footer-text">
            For <strong>Classiq</strong> and <strong>State Street</strong>
          </p>
        </div>
      </footer>
    </div>
  );
} 

export default App;
