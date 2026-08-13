"""Measure serving latency — the number that decides batch vs. real-time.

Reports p50/p95/p99, not just the mean: tail latency is what users feel and
what SLOs are written against. A mean of 5ms with a p99 of 400ms is a bad
endpoint that looks fine on a dashboard.

Run (server must be running in another terminal):

    uvicorn serving.api:app --port 8000
    python -m serving.benchmark --n 500
"""

import argparse
import statistics
import time
import urllib.error
import urllib.request

import json as _json

SAMPLE = {
    "customer_id": "bench",
    "features": {
        "recency_days": 45.0, "frequency_90d": 2.0, "monetary_90d": 120.0,
        "tenure_days": 400.0, "support_tickets_90d": 1.0, "web_sessions_30d": 3.0,
    },
}


def time_request(url: str, payload: dict) -> float:
    """Round-trip time for one request, in milliseconds."""
    data = _json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp.read()
    return (time.perf_counter() - start) * 1000


def percentile(values: list, p: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    idx = min(int(len(ordered) * p / 100), len(ordered) - 1)
    return ordered[idx]


def benchmark(url: str, n: int = 500, warmup: int = 20) -> dict:
    for _ in range(warmup):  # exclude first-call model/JIT warmup from the stats
        time_request(url, SAMPLE)

    timings = [time_request(url, SAMPLE) for _ in range(n)]
    return {
        "n": n,
        "mean_ms": round(statistics.mean(timings), 2),
        "p50_ms": round(percentile(timings, 50), 2),
        "p95_ms": round(percentile(timings, 95), 2),
        "p99_ms": round(percentile(timings, 99), 2),
        "max_ms": round(max(timings), 2),
        "throughput_rps": round(1000 / statistics.mean(timings), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark /predict latency")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    try:
        results = benchmark(args.url, args.n)
    except urllib.error.URLError:
        raise SystemExit(f"Could not reach {args.url} — is the server running?")

    print(f"\nLatency over {results['n']} sequential requests (single process):")
    for key in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]:
        print(f"  {key:<10} {results[key]:>8.2f} ms")
    print(f"  {'throughput':<10} {results['throughput_rps']:>8.1f} req/s")
    print("\nSingle-process, no concurrency — a floor for what one replica does.")


if __name__ == "__main__":
    main()
