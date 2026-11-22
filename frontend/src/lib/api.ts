const BASE_URL = ((import.meta.env.VITE_API_URL as string | undefined) || "").replace(/\/+$/, "");

export interface SignalSummary {
  missing_day_ratio: number;
  zero_day_ratio: number;
  mean_daily_kwh: number;
  recent_90d_ratio: number;
}

export interface HousePrediction {
  consumer_id: string;
  theft_probability: number;
  risk_score_100: number;
  priority_tier: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | string;
  signals: SignalSummary;
  inference_ms: number;
  inspection_rank?: number;
  ground_truth_flag?: number | null;
  disclaimer?: string;
}

export interface DemoHousesResponse {
  total_cached: number;
  returned: number;
  houses: HousePrediction[];
  disclaimer: string;
}

export interface ModelMetricsResponse {
  metrics: {
    model: string;
    dataset: string;
    split_strategy: string;
    test_roc_auc: number;
    test_pr_auc: number;
    baseline_pr_auc: number;
    top_k_precision: {
      top_10: number;
      top_50: number;
      top_100: number;
      top_500: number;
    };
  };
  data_audit: {
    total_consumers?: number;
    total_days?: number;
    theft_rate_pct?: number;
    missing_cells_pct?: number;
    date_range?: { start: string; end: string };
  };
  disclaimer: string;
}

export interface HealthResponse {
  status: string;
  model: string;
  roc_auc: number;
  pr_auc: number;
}

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, options);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const fetchHealth = () => fetchJSON<HealthResponse>("/health");
export const fetchModelMetrics = () => fetchJSON<ModelMetricsResponse>("/api/model-metrics");
export const fetchDemoHouses = (topK = 50) =>
  fetchJSON<DemoHousesResponse>(`/api/demo/houses?top_k=${topK}`);

export const predictSingleConsumer = (consumerId: string, dailyKwh: Array<number | null>) =>
  fetchJSON<HousePrediction>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ consumer_id: consumerId, daily_kwh: dailyKwh }),
  });