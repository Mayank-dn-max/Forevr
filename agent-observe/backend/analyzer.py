"""
analyzer.py — Failure Auto-Discovery Engine
============================================

Runs 6 pattern detectors over your trace history and surfaces
actionable insights automatically. No manual querying needed.

Detectors:
  1. latency_spike      — per-tool latency regression
  2. error_cluster      — repeated failures with same error
  3. cost_anomaly       — per-query cost increase
  4. output_drift       — response length / quality regression
  5. retry_storm        — tool retrying too often
  6. failure_rate_spike — overall agent failure rate rising
"""

import time
import statistics
from collections import defaultdict
from typing import List, Dict, Any

# ── Thresholds (tune these as your dataset grows) ─────────────────────────────
LATENCY_SPIKE_RATIO   = 1.5    # recent avg > 1.5x baseline avg → spike
COST_SPIKE_RATIO      = 1.5    # recent cost > 1.5x baseline → anomaly
OUTPUT_DROP_RATIO     = 0.60   # recent output < 60% of baseline → drift
MIN_RETRY_AVG         = 1.5    # avg retries per call before flagging
MIN_CLUSTER_COUNT     = 3      # same error must appear 3+ times
FAILURE_RATE_DELTA    = 0.20   # failure rate must increase by 20pp
MIN_FAILURE_RATE      = 0.30   # and be above 30% in recent window
RECENT_FRACTION       = 0.35   # last 35% of traces = "recent" window


def _split(items: list):
    """Split a list into baseline (first 65%) and recent (last 35%)."""
    if len(items) < 4:
        return items, []
    cut = max(1, int(len(items) * (1 - RECENT_FRACTION)))
    return items[:cut], items[cut:]


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR 1 — Latency Spikes
# ─────────────────────────────────────────────────────────────────────────────
def detect_latency_spikes(traces: List[Dict]) -> List[Dict]:
    """
    Per tool/span, compare recent latency vs baseline.
    Flags if recent avg > 1.5x baseline avg.
    """
    span_series: Dict[str, List[float]] = defaultdict(list)

    for trace in traces:
        for span in trace.get("spans", []):
            dur = span.get("duration_ms")
            if dur and dur > 0:
                span_series[span["name"]].append(dur)

    insights = []
    for span_name, durations in span_series.items():
        baseline, recent = _split(durations)
        if not recent:
            continue

        baseline_avg = statistics.mean(baseline)
        recent_avg   = statistics.mean(recent)

        if baseline_avg > 0 and recent_avg > baseline_avg * LATENCY_SPIKE_RATIO:
            pct = round((recent_avg / baseline_avg - 1) * 100)
            insights.append({
                "type":     "latency_spike",
                "severity": "critical" if pct > 200 else "warning",
                "tool":     span_name,
                "title":    f"Latency spike on '{span_name}'",
                "message":  f"Latency increased {pct}% in recent runs — "
                            f"baseline {round(baseline_avg)}ms → now {round(recent_avg)}ms",
                "evidence": {
                    "baseline_avg_ms":  round(baseline_avg),
                    "recent_avg_ms":    round(recent_avg),
                    "pct_increase":     pct,
                    "samples_baseline": len(baseline),
                    "samples_recent":   len(recent),
                },
                "recommendation": f"Check if '{span_name}' upstream API changed, "
                                  "or if payload size increased recently.",
            })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR 2 — Error Clusters
# ─────────────────────────────────────────────────────────────────────────────
def detect_error_clusters(traces: List[Dict]) -> List[Dict]:
    """
    Group failed spans by (tool_name, error_fingerprint).
    Flags clusters that appear 3+ times.
    """
    groups: Dict[tuple, List[Dict]] = defaultdict(list)

    for trace in traces:
        for span in trace.get("spans", []):
            if span.get("status") == "failed":
                error_raw = span.get("metadata", {}).get("error", "unknown")
                # Fingerprint: first 60 chars (normalises minor variations)
                fingerprint = str(error_raw)[:60].strip()
                key = (span["name"], fingerprint)
                groups[key].append({
                    "trace_id":  trace.get("trace_id", ""),
                    "timestamp": trace.get("start_time", 0),
                })

    insights = []
    for (span_name, error_fp), occurrences in groups.items():
        if len(occurrences) < MIN_CLUSTER_COUNT:
            continue

        first_ts = min(o["timestamp"] for o in occurrences)
        last_ts  = max(o["timestamp"] for o in occurrences)
        span_sec = max(1, last_ts - first_ts)

        insights.append({
            "type":     "error_cluster",
            "severity": "critical" if len(occurrences) >= 5 else "warning",
            "tool":     span_name,
            "title":    f"Repeated failure on '{span_name}'",
            "message":  f"{len(occurrences)} traces show '{span_name}' "
                        f"failing with the same error",
            "evidence": {
                "count":           len(occurrences),
                "error_sample":    error_fp,
                "time_window_min": round(span_sec / 60, 1),
                "first_seen_ago":  f"{round((time.time() - first_ts) / 60)}m ago",
                "last_seen_ago":   f"{round((time.time() - last_ts) / 60)}m ago",
            },
            "recommendation": f"'{span_name}' is consistently failing. "
                               "Investigate the downstream service or add a circuit breaker.",
        })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR 3 — Cost Anomalies
# ─────────────────────────────────────────────────────────────────────────────
def detect_cost_anomalies(traces: List[Dict]) -> List[Dict]:
    """
    Compare per-query cost in recent traces vs baseline.
    Flags if cost rose > 50%.
    """
    costs = [(t.get("start_time", 0), t.get("total_cost_usd", 0))
             for t in traces if t.get("total_cost_usd", 0) > 0]
    if len(costs) < 4:
        return []

    costs.sort(key=lambda x: x[0])
    vals = [c[1] for c in costs]
    baseline, recent = _split(vals)
    if not recent:
        return []

    baseline_avg = statistics.mean(baseline)
    recent_avg   = statistics.mean(recent)

    if baseline_avg > 0 and recent_avg > baseline_avg * COST_SPIKE_RATIO:
        pct = round((recent_avg / baseline_avg - 1) * 100)
        daily_proj = recent_avg * 1000  # rough: 1000 queries/day

        return [{
            "type":     "cost_anomaly",
            "severity": "warning",
            "tool":     "LLM generation",
            "title":    "Per-query cost rising",
            "message":  f"Cost per query increased {pct}% — "
                        f"${round(baseline_avg, 5)} → ${round(recent_avg, 5)}",
            "evidence": {
                "baseline_avg_usd": round(baseline_avg, 5),
                "recent_avg_usd":   round(recent_avg, 5),
                "pct_increase":     pct,
                "projected_daily":  f"${round(daily_proj, 2)} @ 1k queries/day",
            },
            "recommendation": "Check if model switched to a more expensive tier, "
                              "or if context length is growing.",
        }]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR 4 — Output Drift
# ─────────────────────────────────────────────────────────────────────────────
def detect_output_drift(traces: List[Dict]) -> List[Dict]:
    """
    Track length of final_response preview across traces.
    A significant drop in output length = possible quality regression.
    """
    points = []
    for trace in traces:
        for span in trace.get("spans", []):
            if span.get("name") == "final_response":
                preview = span.get("metadata", {}).get("response_preview", "")
                if preview:
                    points.append((trace.get("start_time", 0), len(preview)))

    if len(points) < 4:
        return []

    points.sort(key=lambda x: x[0])
    lengths = [p[1] for p in points]
    baseline, recent = _split(lengths)
    if not recent:
        return []

    baseline_avg = statistics.mean(baseline)
    recent_avg   = statistics.mean(recent)

    if baseline_avg > 0 and recent_avg < baseline_avg * OUTPUT_DROP_RATIO:
        pct_drop = round((1 - recent_avg / baseline_avg) * 100)
        return [{
            "type":     "output_drift",
            "severity": "warning",
            "tool":     "final_response",
            "title":    "Output quality regression detected",
            "message":  f"Response length dropped {pct_drop}% — possible quality degradation",
            "evidence": {
                "baseline_avg_chars": round(baseline_avg),
                "recent_avg_chars":   round(recent_avg),
                "pct_drop":           pct_drop,
            },
            "recommendation": "Check if the prompt changed, or if web search is returning "
                              "shorter/empty context recently.",
        }]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR 5 — Retry Storms
# ─────────────────────────────────────────────────────────────────────────────
def detect_retry_storms(traces: List[Dict]) -> List[Dict]:
    """
    Per tool, track retry_count. Flags tools that are retrying excessively.
    """
    retry_data: Dict[str, List[int]] = defaultdict(list)

    for trace in traces:
        for span in trace.get("spans", []):
            rc = span.get("retry_count", 0)
            if rc > 0:
                retry_data[span["name"]].append(rc)

    insights = []
    for tool, retry_counts in retry_data.items():
        if len(retry_counts) < 2:
            continue
        avg_retries   = statistics.mean(retry_counts)
        total_retries = sum(retry_counts)

        if avg_retries >= MIN_RETRY_AVG:
            insights.append({
                "type":     "retry_storm",
                "severity": "warning",
                "tool":     tool,
                "title":    f"Retry storm on '{tool}'",
                "message":  f"'{tool}' averaging {avg_retries:.1f} retries/call "
                            f"({total_retries} total retries across {len(retry_counts)} traces)",
                "evidence": {
                    "avg_retries_per_call": round(avg_retries, 1),
                    "total_retries":        total_retries,
                    "affected_traces":      len(retry_counts),
                    "max_retries_in_run":   max(retry_counts),
                },
                "recommendation": f"'{tool}' is unstable. Add exponential backoff, "
                                   "circuit breaker, or switch to a more reliable provider.",
            })

    return insights


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR 6 — Failure Rate Spike
# ─────────────────────────────────────────────────────────────────────────────
def detect_failure_rate_spike(traces: List[Dict]) -> List[Dict]:
    """
    Compare agent-level failure rate in recent window vs baseline.
    Flags if failure rate jumped significantly.
    """
    if len(traces) < 6:
        return []

    statuses = [t.get("status", "success") for t in traces]
    baseline_statuses, recent_statuses = _split(statuses)
    if not recent_statuses:
        return []

    def fail_rate(lst):
        return sum(1 for s in lst if s == "failed") / len(lst) if lst else 0

    baseline_rate = fail_rate(baseline_statuses)
    recent_rate   = fail_rate(recent_statuses)

    if (recent_rate - baseline_rate) >= FAILURE_RATE_DELTA and recent_rate >= MIN_FAILURE_RATE:
        return [{
            "type":     "failure_rate_spike",
            "severity": "critical",
            "tool":     "agent",
            "title":    "Agent failure rate spiking",
            "message":  f"Failure rate jumped from {round(baseline_rate*100)}% → "
                        f"{round(recent_rate*100)}% in recent runs",
            "evidence": {
                "baseline_failure_pct": round(baseline_rate * 100),
                "recent_failure_pct":   round(recent_rate * 100),
                "delta_pp":             round((recent_rate - baseline_rate) * 100),
                "recent_traces":        len(recent_statuses),
            },
            "recommendation": "Check the most recently added spans. A new tool or "
                              "prompt change may have introduced a regression.",
        }]

    return []


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYZER
# ─────────────────────────────────────────────────────────────────────────────
class PatternAnalyzer:
    def analyze(self, traces: List[Dict]) -> Dict:
        """
        Run all 6 detectors and return a structured insights report.
        Called on every /insights request — lightweight enough to run on-demand.
        """
        if len(traces) < 2:
            return {
                "status":          "insufficient_data",
                "message":         f"Need at least 2 traces to detect patterns. "
                                   f"Have {len(traces)}.",
                "insights":        [],
                "traces_analyzed": len(traces),
                "analyzed_at":     time.time(),
            }

        insights = []
        insights.extend(detect_latency_spikes(traces))
        insights.extend(detect_error_clusters(traces))
        insights.extend(detect_cost_anomalies(traces))
        insights.extend(detect_output_drift(traces))
        insights.extend(detect_retry_storms(traces))
        insights.extend(detect_failure_rate_spike(traces))

        # Sort: critical first, then warning, then info
        order = {"critical": 0, "warning": 1, "info": 2}
        insights.sort(key=lambda x: order.get(x.get("severity", "info"), 3))

        # Add unique IDs for frontend keying
        for i, ins in enumerate(insights):
            ins["id"] = f"ins_{i}_{int(time.time())}"

        return {
            "status":          "ok",
            "insights":        insights,
            "insight_count":   len(insights),
            "critical_count":  sum(1 for i in insights if i["severity"] == "critical"),
            "warning_count":   sum(1 for i in insights if i["severity"] == "warning"),
            "traces_analyzed": len(traces),
            "analyzed_at":     time.time(),
        }


# Module-level singleton
analyzer = PatternAnalyzer()
