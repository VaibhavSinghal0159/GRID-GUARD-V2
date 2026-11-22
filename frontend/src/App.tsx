import { useEffect, useMemo, useState } from "react";
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

function buildSyntheticProfile(kind: "normal" | "bypass_drop" | "missing_outages"): Array<number | null> {
  const days = 1029;
  if (kind === "normal") {
    return Array.from({ length: days }, (_, i) =>
      Number((14 + 3 * Math.sin(i / 25) + (i % 7 >= 5 ? 2 : 0)).toFixed(2))
    );
  }
  if (kind === "bypass_drop") {
    return Array.from({ length: days }, (_, i) => (i < 720 ? Number((18 + (i % 5)).toFixed(2)) : 0.4));
  }
  return Array.from({ length: days }, (_, i) => (i % 4 === 0 ? null : i > 600 ? 0.0 : 16.5));
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metricsData, setMetricsData] = useState<ModelMetricsResponse | null>(null);
  const [houses, setHouses] = useState<HousePrediction[]>([]);
  const [disclaimer, setDisclaimer] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [tierFilter, setTierFilter] = useState<string>("ALL");
  const [searchId, setSearchId] = useState<string>("");
  const [selectedHouse, setSelectedHouse] = useState<HousePrediction | null>(null);

  const [simLoading, setSimLoading] = useState(false);
  const [simResult, setSimResult] = useState<HousePrediction | null>(null);
  const [simError, setSimError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    async function loadAll() {
      try {
        setLoading(true);
        const [h, m, d] = await Promise.all([
          fetchHealth(),
          fetchModelMetrics(),
          fetchDemoHouses(50),
        ]);
        if (!active) return;
        setHealth(h);
        setMetricsData(m);
        setHouses(d.houses);
        setDisclaimer(d.disclaimer || m.disclaimer);
        if (d.houses.length > 0) setSelectedHouse(d.houses[0]);
      } catch (err) {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to connect to GridGuard API.");
      } finally {
        if (active) setLoading(false);
      }
    }
    loadAll();
    return () => {
      active = false;
    };
  }, []);

  const filteredHouses = useMemo(() => {
    return houses.filter((h) => {
      const matchesTier = tierFilter === "ALL" || h.priority_tier === tierFilter;
      const matchesQuery = h.consumer_id.toLowerCase().includes(searchId.trim().toLowerCase());
      return matchesTier && matchesQuery;
    });
  }, [houses, tierFilter, searchId]);

  async function runLiveInference(kind: "normal" | "bypass_drop" | "missing_outages") {
    try {
      setSimLoading(true);
      setSimError(null);
      const id = `LIVE_${kind.toUpperCase()}_${Date.now().toString().slice(-4)}`;
      const series = buildSyntheticProfile(kind);
      const res = await predictSingleConsumer(id, series);
      setSimResult(res);
    } catch (err) {
      setSimError(err instanceof Error ? err.message : "Inference failed.");
    } finally {
      setSimLoading(false);
    }
  }

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
                health?.status === "ok"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                  : "border-red-500/40 bg-red-500/10 text-red-300"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  health?.status === "ok" ? "bg-emerald-400" : "bg-red-400"
                }`}
              />
              {health?.status === "ok" ? "ONNX Engine Online" : "API Offline"}
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
          <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-300">
            <strong>Backend Connection Error:</strong> {error}. Make sure FastAPI is running on port 8000 or{" "}
            <code className="rounded bg-slate-900 px-1">VITE_API_URL</code> is configured.
          </div>
        )}

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

        <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5 lg:col-span-2">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-white">Field Inspection Priority Queue</h2>
                <p className="text-xs text-slate-400">
                  50 stratified SGCC consumers scored live via ONNX Runtime CPU
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
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
              </div>
            </div>

            {loading ? (
              <p className="py-12 text-center text-sm text-slate-400">Scoring consumers via ONNX Runtime...</p>
            ) : (
              <div className="max-h-[460px] overflow-auto rounded-lg border border-slate-800">
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
                          <td className="px-3 py-2.5 font-semibold text-white">{h.risk_score_100.toFixed(1)}%</td>
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
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="text-base font-semibold text-white">Selected Consumer Signals</h2>
              {selectedHouse ? (
                <div className="mt-4 space-y-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">Consumer ID</span>
                    <span className="font-mono text-slate-200">{selectedHouse.consumer_id}</span>
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
                      <p className="text-[11px] text-slate-400">Zero-Usage Days</p>
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

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="text-base font-semibold text-white">Live ONNX Inference Sandbox</h2>
              <p className="mt-1 text-xs text-slate-400">
                Send a synthetic 1,029-day smart-meter sequence directly to{" "}
                <code className="text-cyan-400">POST /api/predict</code>.
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  disabled={simLoading}
                  onClick={() => runLiveInference("normal")}
                  className="rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-700 disabled:opacity-50"
                >
                  Test Normal Usage
                </button>
                <button
                  disabled={simLoading}
                  onClick={() => runLiveInference("bypass_drop")}
                  className="rounded-lg border border-red-500/40 bg-red-500/15 px-3 py-1.5 text-xs font-medium text-red-300 hover:bg-red-500/25 disabled:opacity-50"
                >
                  Test Meter Bypass Drop
                </button>
                <button
                  disabled={simLoading}
                  onClick={() => runLiveInference("missing_outages")}
                  className="rounded-lg border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-xs font-medium text-amber-200 hover:bg-amber-500/25 disabled:opacity-50"
                >
                  Test Missing-Mask Gaps
                </button>
              </div>

              {simError && <p className="mt-3 text-xs text-red-400">{simError}</p>}
              {simResult && (
                <div className="mt-4 rounded-lg border border-slate-800 bg-slate-950 p-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-slate-300">{simResult.consumer_id}</span>
                    <span className="text-slate-400">{simResult.inference_ms} ms</span>
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
                </div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}