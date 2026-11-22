import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchDemoHouses,
  fetchHealth,
  fetchModelMetrics,
  predictSingleConsumer,
  type HealthResponse,
  type HousePrediction,
  type ModelMetricsResponse,
} from "./lib/api";

const TIER_STYLES: Record<string, string> = {
  CRITICAL: "bg-red-500/15 text-red-400 border-red-500/40",
  HIGH: "bg-orange-500/15 text-orange-300 border-orange-500/40",
  MEDIUM: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  LOW: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
};

const TIER_BAR_COLORS: Record<string, string> = {
  CRITICAL: "bg-red-500",
  HIGH: "bg-orange-400",
  MEDIUM: "bg-amber-400",
  LOW: "bg-emerald-400",
};

/**
 * Generates a deterministic 1,029-day (147 weeks x 7 days) smart-meter profile
 * with annual seasonality, weekly rhythm, realistic noise, optional late-stage
 * bypass drop, and contiguous multi-week missing-telemetry blocks.
 */
function generateMeterProfile(
  baseKwh: number,
  bypassDropPct: number,
  missingBlockPct: number,
  erraticSpikes = false
): Array<number | null> {
  const days = 1029;
  const dropStartDay = 780; // Last ~8 months
  const missingDaysTarget = Math.round((days * missingBlockPct) / 100);

  // Build contiguous outage blocks (more realistic than single alternating nulls)
  const missingFlags = new Array<boolean>(days).fill(false);
  if (missingDaysTarget > 0) {
    const blockLen = 21;
    let remaining = missingDaysTarget;
    for (let start = 40; start < days - 30 && remaining > 0; start += Math.max(28, Math.floor((days - 70) / Math.max(1, Math.ceil(missingDaysTarget / blockLen))))) {
      for (let offset = 0; offset < blockLen && start + offset < days && remaining > 0; offset++) {
        missingFlags[start + offset] = true;
        remaining--;
      }
    }
  }

  return Array.from({ length: days }, (_, i) => {
    if (missingFlags[i]) return null;

    const seasonal = 1 + 0.22 * Math.sin((2 * Math.PI * i) / 365);
    const weekendBoost = i % 7 >= 5 ? 1.12 : 0.96;
    // Deterministic pseudo-noise in [-0.12, +0.12]
    const noise = 1 + 0.12 * Math.sin(i * 12.9898) * Math.cos(i * 78.233);
    const spike = erraticSpikes && i % 19 === 0 ? 2.4 : 1.0;

    let value = baseKwh * seasonal * weekendBoost * noise * spike;
    if (i >= dropStartDay && bypassDropPct > 0) {
      value *= Math.max(0.03, 1 - bypassDropPct / 100);
    }
    return Number(Math.max(0.05, value).toFixed(2));
  });
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metricsData, setMetricsData] = useState<ModelMetricsResponse | null>(null);
  const [houses, setHouses] = useState<HousePrediction[]>([]);
  const [disclaimer, setDisclaimer] = useState<string>("");

  const [loading, setLoading] = useState(true);
  const [statusMessage, setStatusMessage] = useState<string>("Connecting to ONNX inference engine...");
  const [error, setError] = useState<string | null>(null);

  const [tierFilter, setTierFilter] = useState<string>("ALL");
  const [searchId, setSearchId] = useState<string>("");
  const [selectedHouse, setSelectedHouse] = useState<HousePrediction | null>(null);

  // Interactive Live ONNX Sandbox state
  const [baseKwh, setBaseKwh] = useState<number>(85);
  const [bypassDropPct, setBypassDropPct] = useState<number>(65);
  const [missingBlockPct, setMissingBlockPct] = useState<number>(35);
  const [simLoading, setSimLoading] = useState(false);
  const [simResult, setSimResult] = useState<HousePrediction | null>(null);
  const [simError, setSimError] = useState<string | null>(null);

  const loadAllWithRetry = useCallback(async (maxAttempts = 5) => {
    setLoading(true);
    setError(null);
    setStatusMessage("Connecting to ONNX inference engine...");

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        if (attempt > 1) {
          setStatusMessage(
            `Waking up Render ONNX container (attempt ${attempt}/${maxAttempts} — cold start takes ~40s)...`
          );
        }
        const [h, m, d] = await Promise.all([
          fetchHealth(),
          fetchModelMetrics(),
          fetchDemoHouses(50),
        ]);
        setHealth(h);
        setMetricsData(m);
        setHouses(d.houses);
        setDisclaimer(d.disclaimer || m.disclaimer);
        setSelectedHouse((prev) => prev ?? (d.houses.length > 0 ? d.houses[0] : null));
        setLoading(false);
        return;
      } catch (err) {
        if (attempt === maxAttempts) {
          setError(err instanceof Error ? err.message : "Failed to connect to GridGuard API.");
          setLoading(false);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, attempt * 4000));
      }
    }
  }, []);

  useEffect(() => {
    loadAllWithRetry();
  }, [loadAllWithRetry]);

  const filteredHouses = useMemo(() => {
    return houses.filter((h) => {
      const matchesTier = tierFilter === "ALL" || h.priority_tier === tierFilter;
      const matchesQuery = h.consumer_id.toLowerCase().includes(searchId.trim().toLowerCase());
      return matchesTier && matchesQuery;
    });
  }, [houses, tierFilter, searchId]);

  async function runSandboxInference(
    label: string,
    kwh: number,
    dropPct: number,
    gapPct: number,
    erratic = false
  ) {
    try {
      setSimLoading(true);
      setSimError(null);
      setBaseKwh(kwh);
      setBypassDropPct(dropPct);
      setMissingBlockPct(gapPct);

      const id = `LIVE_${label}_${Date.now().toString().slice(-4)}`;
      const series = generateMeterProfile(kwh, dropPct, gapPct, erratic);
      const res = await predictSingleConsumer(id, series);
      setSimResult(res);
    } catch (err) {
      setSimError(err instanceof Error ? err.message : "Inference request failed.");
    } finally {
      setSimLoading(false);
    }
  }

  const isOnline = health?.status === "ok";
  const isWaking = loading && !isOnline;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white">
              GridGuard AI <span className="text-cyan-400">v2</span>
            </h1>
            <p className="text-xs text-slate-400">
              2-Channel 2D-CNN (ONNX Runtime) &middot; Multi-Year SGCC Electricity Theft Triage
            </p>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 font-medium ${
                isOnline
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : isWaking
                  ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
                  : "border-red-500/40 bg-red-500/10 text-red-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  isOnline
                    ? "bg-emerald-400"
                    : isWaking
                    ? "animate-pulse bg-amber-400"
                    : "bg-red-400"
                }`}
              />
              {isOnline ? "ONNX Engine Online" : isWaking ? "Waking ONNX Engine..." : "API Offline"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-6 py-6">
        {disclaimer && (
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-200">
            <strong className="font-semibold">Operational Scope:</strong> {disclaimer}
          </div>
        )}

        {error && (
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-300">
            <div>
              <strong>Backend Connection Error:</strong> {error}. Free Render instances sleep after 15
              minutes of inactivity; click retry once the container wakes up.
            </div>
            <button
              onClick={() => loadAllWithRetry(4)}
              className="rounded-lg border border-red-400/50 bg-red-500/20 px-3 py-1.5 text-xs font-semibold text-red-100 hover:bg-red-500/30"
            >
              Retry Connection
            </button>
          </div>
        )}

        {/* Holdout Benchmark KPI Strip */}
        <section className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs text-slate-400">Holdout ROC-AUC</p>
            <p className="mt-1 text-2xl font-bold text-cyan-400">
              {metricsData ? metricsData.metrics.test_roc_auc.toFixed(4) : "--"}
            </p>
            <p className="mt-1 text-xs text-slate-500">v1 26-day baseline: 0.6455</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs text-slate-400">Holdout PR-AUC</p>
            <p className="mt-1 text-2xl font-bold text-indigo-400">
              {metricsData ? metricsData.metrics.test_pr_auc.toFixed(4) : "--"}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              ~5x lift over {metricsData ? metricsData.metrics.baseline_pr_auc : "0.0853"} base rate
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs text-slate-400">Top-10 Inspection Precision</p>
            <p className="mt-1 text-2xl font-bold text-emerald-400">
              {metricsData ? `${(metricsData.metrics.top_k_precision.top_10 * 100).toFixed(0)}%` : "--"}
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Top-50 Precision:{" "}
              {metricsData ? `${(metricsData.metrics.top_k_precision.top_50 * 100).toFixed(0)}%` : "--"}
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
            <p className="text-xs text-slate-400">Training Horizon</p>
            <p className="mt-1 text-2xl font-bold text-white">1,029 Days</p>
            <p className="mt-1 text-xs text-slate-500">2-Channel (147w x 7d + NaN mask)</p>
          </div>
        </section>

        {/* Main Workspace: Ranked Queue + Telemetry Inspector & Live Sandbox */}
        <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5 lg:col-span-2">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-white">Field Inspection Priority Queue</h2>
                <p className="text-xs text-slate-400">
                  Showing {filteredHouses.length} of {houses.length} stratified SGCC consumers scored via ONNX Runtime CPU
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <input
                  type="text"
                  value={searchId}
                  onChange={(e) => setSearchId(e.target.value)}
                  placeholder="Search Consumer ID..."
                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-100 placeholder:text-slate-500 focus:border-cyan-500 focus:outline-none"
                />
                <select
                  value={tierFilter}
                  onChange={(e) => setTierFilter(e.target.value)}
                  className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
                >
                  <option value="ALL">All Tiers</option>
                  <option value="CRITICAL">CRITICAL</option>
                  <option value="HIGH">HIGH</option>
                  <option value="MEDIUM">MEDIUM</option>
                  <option value="LOW">LOW</option>
                </select>
                {(tierFilter !== "ALL" || searchId.trim() !== "") && (
                  <button
                    onClick={() => {
                      setTierFilter("ALL");
                      setSearchId("");
                    }}
                    className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-800"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>

            {loading ? (
              <div className="flex flex-col items-center justify-center gap-2 py-14 text-center">
                <span className="h-5 w-5 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
                <p className="text-sm text-slate-300">{statusMessage}</p>
              </div>
            ) : filteredHouses.length === 0 ? (
              <p className="py-12 text-center text-sm text-slate-400">
                No consumers match the current filter.
              </p>
            ) : (
              <div className="max-h-[520px] overflow-auto rounded-lg border border-slate-800">
                <table className="w-full text-left text-xs">
                  <thead className="sticky top-0 bg-slate-950 text-slate-400">
                    <tr>
                      <th className="px-3 py-2.5">Rank</th>
                      <th className="px-3 py-2.5">Consumer ID</th>
                      <th className="px-3 py-2.5">Risk Score</th>
                      <th className="px-3 py-2.5">Priority</th>
                      <th className="px-3 py-2.5">Missing %</th>
                      <th className="px-3 py-2.5">Recent 90d</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80">
                    {filteredHouses.map((h) => {
                      const isSelected = selectedHouse?.consumer_id === h.consumer_id;
                      return (
                        <tr
                          key={h.consumer_id}
                          onClick={() => setSelectedHouse(h)}
                          className={`cursor-pointer transition-colors hover:bg-slate-800/60 ${
                            isSelected ? "bg-cyan-500/10" : ""
                          }`}
                        >
                          <td className="px-3 py-2.5 font-mono text-slate-400">#{h.inspection_rank ?? "-"}</td>
                          <td className="px-3 py-2.5 font-mono text-slate-200">{h.consumer_id.slice(0, 14)}...</td>
                          <td className="px-3 py-2.5">
                            <div className="flex items-center gap-2">
                              <span className="w-11 font-semibold text-white">{h.risk_score_100.toFixed(1)}%</span>
                              <div className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-slate-800 sm:block">
                                <div
                                  className={`h-full rounded-full ${TIER_BAR_COLORS[h.priority_tier] || "bg-cyan-400"}`}
                                  style={{ width: `${Math.min(100, Math.max(2, h.risk_score_100))}%` }}
                                />
                              </div>
                            </div>
                          </td>
                          <td className="px-3 py-2.5">
                            <span
                              className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
                                TIER_STYLES[h.priority_tier] || TIER_STYLES.LOW
                              }`}
                            >
                              {h.priority_tier}
                            </span>
                          </td>
                          <td className="px-3 py-2.5 text-slate-300">
                            {(h.signals.missing_day_ratio * 100).toFixed(1)}%
                          </td>
                          <td className="px-3 py-2.5 text-slate-300">{h.signals.recent_90d_ratio.toFixed(2)}x</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="space-y-6">
            {/* Selected Consumer Diagnostic Card */}
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="text-base font-semibold text-white">Selected Consumer Signals</h2>
              {selectedHouse ? (
                <div className="mt-4 space-y-3 text-xs">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-slate-400">Consumer ID</span>
                    <span className="truncate font-mono text-slate-200" title={selectedHouse.consumer_id}>
                      {selectedHouse.consumer_id}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Theft Probability</span>
                    <span className="text-base font-bold text-cyan-400">
                      {(selectedHouse.theft_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-2">
                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-[11px] text-slate-400">Mean Daily kWh</p>
                      <p className="mt-1 text-sm font-semibold">{selectedHouse.signals.mean_daily_kwh} kWh</p>
                    </div>
                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-[11px] text-slate-400">Observed Zero Days</p>
                      <p className="mt-1 text-sm font-semibold">
                        {(selectedHouse.signals.zero_day_ratio * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-[11px] text-slate-400">Missing Mask Ratio</p>
                      <p className="mt-1 text-sm font-semibold">
                        {(selectedHouse.signals.missing_day_ratio * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-950 p-3">
                      <p className="text-[11px] text-slate-400">Recent 90d vs Mean</p>
                      <p className="mt-1 text-sm font-semibold">{selectedHouse.signals.recent_90d_ratio}x</p>
                    </div>
                  </div>
                  {selectedHouse.ground_truth_flag !== undefined && selectedHouse.ground_truth_flag !== null && (
                    <div className="mt-2 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-[11px] text-slate-300">
                      SGCC Holdout Audit Label:{" "}
                      <strong className={selectedHouse.ground_truth_flag === 1 ? "text-red-400" : "text-emerald-400"}>
                        {selectedHouse.ground_truth_flag === 1 ? "Confirmed Theft (FLAG=1)" : "Normal (FLAG=0)"}
                      </strong>
                    </div>
                  )}
                </div>
              ) : (
                <p className="mt-4 text-xs text-slate-400">Select a consumer from the queue to inspect signals.</p>
              )}
            </div>

            {/* Live POST /api/predict Interactive Sandbox */}
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="text-base font-semibold text-white">Live ONNX Stress-Test Sandbox</h2>
              <p className="mt-1 text-xs text-slate-400">
                Score a synthetic 1,029-day sequence live via <code className="text-cyan-400">POST /api/predict</code>.
              </p>

              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  disabled={simLoading || !isOnline}
                  onClick={() => runSandboxInference("RESIDENTIAL", 5.5, 0, 2, false)}
                  className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1.5 text-xs font-medium text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-50"
                >
                  Steady Residential
                </button>
                <button
                  disabled={simLoading || !isOnline}
                  onClick={() => runSandboxInference("HIGH_LOAD_BYPASS", 180, 75, 45, true)}
                  className="rounded-lg border border-red-500/40 bg-red-500/15 px-2.5 py-1.5 text-xs font-medium text-red-300 hover:bg-red-500/25 disabled:opacity-50"
                >
                  High-Load Bypass + Gaps
                </button>
                <button
                  disabled={simLoading || !isOnline}
                  onClick={() => runSandboxInference("ERRATIC_SURGE", 120, 25, 65, true)}
                  className="rounded-lg border border-amber-500/40 bg-amber-500/15 px-2.5 py-1.5 text-xs font-medium text-amber-200 hover:bg-amber-500/25 disabled:opacity-50"
                >
                  Heavy Missing Blocks
                </button>
              </div>

              {/* Interactive Parameter Sliders */}
              <div className="mt-4 space-y-2.5 rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs">
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Base Daily Load</span>
                    <span className="font-mono text-slate-200">{baseKwh} kWh</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={300}
                    step={2}
                    value={baseKwh}
                    onChange={(e) => setBaseKwh(Number(e.target.value))}
                    className="mt-1 w-full accent-cyan-400"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Late-Stage Bypass Drop</span>
                    <span className="font-mono text-slate-200">{bypassDropPct}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={95}
                    step={5}
                    value={bypassDropPct}
                    onChange={(e) => setBypassDropPct(Number(e.target.value))}
                    className="mt-1 w-full accent-cyan-400"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Missing-Mask Outage Blocks</span>
                    <span className="font-mono text-slate-200">{missingBlockPct}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={75}
                    step={5}
                    value={missingBlockPct}
                    onChange={(e) => setMissingBlockPct(Number(e.target.value))}
                    className="mt-1 w-full accent-cyan-400"
                  />
                </div>
                <button
                  disabled={simLoading || !isOnline}
                  onClick={() => runSandboxInference("CUSTOM", baseKwh, bypassDropPct, missingBlockPct, true)}
                  className="mt-1 w-full rounded-lg bg-cyan-500/20 py-1.5 text-xs font-semibold text-cyan-300 ring-1 ring-cyan-500/40 hover:bg-cyan-500/30 disabled:opacity-50"
                >
                  {simLoading ? "Running ONNX Inference..." : "Score Custom 1,029-Day Profile"}
                </button>
              </div>

              {simError && <p className="mt-3 text-xs text-red-400">{simError}</p>}
              {simResult && (
                <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-slate-300">{simResult.consumer_id}</span>
                    <span className="font-mono text-emerald-400">{simResult.inference_ms} ms</span>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <span className="text-lg font-bold text-white">{simResult.risk_score_100}% Risk</span>
                    <span
                      className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${
                        TIER_STYLES[simResult.priority_tier] || TIER_STYLES.LOW
                      }`}
                    >
                      {simResult.priority_tier}
                    </span>
                  </div>
                  <div className="mt-2.5 grid grid-cols-2 gap-1.5 border-t border-slate-800 pt-2 text-[11px] text-slate-400">
                    <span>Mean: <strong className="text-slate-200">{simResult.signals.mean_daily_kwh} kWh</strong></span>
                    <span>Missing: <strong className="text-slate-200">{(simResult.signals.missing_day_ratio * 100).toFixed(1)}%</strong></span>
                    <span>Obs. Zero: <strong className="text-slate-200">{(simResult.signals.zero_day_ratio * 100).toFixed(1)}%</strong></span>
                    <span>90d Ratio: <strong className="text-slate-200">{simResult.signals.recent_90d_ratio}x</strong></span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}