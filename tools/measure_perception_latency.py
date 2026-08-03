#!/usr/bin/env python3
"""Measure perception-PC clock offset, network delay, and stream residual delay.

The perception server must expose ``GET /latency`` and
``GET /detections/stream`` on the same host/port.  The probe uses the standard
four-timestamp NTP equations, then applies the estimated clock offset to live
detection records.  The final recommendation is the residual delay *after*
``record['elapsed_s']``; it is therefore suitable for
``GP8_PERCEPTION_LATENCY_S`` without double-counting model inference.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import statistics
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSample:
    rtt_s: float
    offset_s: float  # server clock - client clock


def calculate_probe(t0: float, t1: float, t2: float, t3: float) -> ProbeSample:
    """Return NTP-style network RTT and server-minus-client clock offset."""
    server_work = max(0.0, t2 - t1)
    rtt = max(0.0, (t3 - t0) - server_work)
    offset = ((t1 - t0) + (t2 - t3)) / 2.0
    return ProbeSample(rtt_s=rtt, offset_s=offset)


def percentile(values, pct: float) -> float:
    """Linear-interpolated percentile for a non-empty numeric sequence."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def estimate_clock_offset(samples: list[ProbeSample]) -> float:
    """Median offset from the lowest-RTT fifth, reducing queueing asymmetry."""
    if not samples:
        raise ValueError("at least one probe sample is required")
    keep = max(1, min(len(samples), max(3, len(samples) // 5)))
    best = sorted(samples, key=lambda sample: sample.rtt_s)[:keep]
    return float(statistics.median(sample.offset_s for sample in best))


def _server_urls(server: str) -> tuple[str, str]:
    parsed = urllib.parse.urlsplit(server)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("--server must look like http://<perception-ip>:8080")
    base = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )
    return f"{base}/latency", f"{base}/detections/stream"


def run_probes(url: str, count: int, timeout: float) -> list[ProbeSample]:
    parsed = urllib.parse.urlsplit(url)
    conn_type = (
        http.client.HTTPSConnection if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_type(parsed.hostname, parsed.port, timeout=timeout)
    conn.connect()
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
    samples = []
    try:
        for index in range(count + 1):  # first request warms the TCP connection
            separator = "&" if "?" in path else "?"
            request_path = f"{path}{separator}n={index}"
            t0 = time.time_ns() / 1e9
            conn.request("GET", request_path, headers={"Cache-Control": "no-cache"})
            response = conn.getresponse()
            body = response.read()
            t3 = time.time_ns() / 1e9
            if response.status != 200:
                raise RuntimeError(f"latency probe returned HTTP {response.status}")
            payload = json.loads(body)
            if payload.get("protocol") != "gp8-latency-v1":
                raise RuntimeError("server does not support gp8-latency-v1")
            if index == 0:
                continue
            t1 = float(payload["server_receive_time_ns"]) / 1e9
            t2 = float(payload["server_send_time_ns"]) / 1e9
            samples.append(calculate_probe(t0, t1, t2, t3))
    finally:
        conn.close()
    return samples


def measure_stream(
    url: str,
    clock_offset_s: float,
    count: int,
    timeout: float,
    max_record_age_s: float,
) -> tuple[list[float], list[float]]:
    """Return (inference-to-receipt, post-inference residual) samples."""
    end_to_end = []
    residual = []
    with urllib.request.urlopen(url, timeout=timeout) as response:
        while len(residual) < count:
            raw = response.readline()
            if not raw:
                raise RuntimeError("detection stream closed before enough records arrived")
            received = time.time_ns() / 1e9
            record = json.loads(raw)
            server_timestamp = float(record.get("timestamp") or 0.0)
            inference_s = float(record.get("elapsed_s") or 0.0)
            if server_timestamp <= 0.0:
                continue
            # offset = server - client, so subtract it to express the producer's
            # timestamp on this computer's wall-clock timeline.
            timestamp_client_clock = server_timestamp - clock_offset_s
            total = received - timestamp_client_clock
            # The stream sends the current cached record immediately on connect;
            # discard it if it predates this measurement session.
            if abs(total) > max_record_age_s:
                continue
            end_to_end.append(total)
            residual.append(total - inference_s)
    return end_to_end, residual


def _ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.2f} ms"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server", required=True,
        help="Perception stream base URL, e.g. http://192.168.0.10:8080",
    )
    parser.add_argument("--probes", type=int, default=40)
    parser.add_argument("--records", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument(
        "--max-record-age", type=float, default=2.0,
        help="Discard the cached first stream record if older than this many seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.probes < 3 or args.records < 1:
        raise SystemExit("--probes must be >= 3 and --records must be >= 1")
    probe_url, stream_url = _server_urls(args.server)

    print(f"probing clock/network: {probe_url}", flush=True)
    probes = run_probes(probe_url, args.probes, args.timeout)
    offset = estimate_clock_offset(probes)
    rtts = [sample.rtt_s for sample in probes]
    print(
        f"clock offset (server-client): {_ms(offset)}\n"
        f"network RTT: min {_ms(min(rtts))}, median {_ms(percentile(rtts, 50))}, "
        f"p95 {_ms(percentile(rtts, 95))}\n"
        f"estimated one-way network: ~{_ms(min(rtts) / 2.0)}",
        flush=True,
    )

    print(f"measuring {args.records} live records: {stream_url}", flush=True)
    totals, residuals = measure_stream(
        stream_url, offset, args.records, args.timeout, args.max_record_age,
    )
    recommendation = max(0.0, percentile(residuals, 50))
    residual_median = percentile(residuals, 50)
    print(
        "\nstream inference-start -> robot receipt:\n"
        f"  median {_ms(percentile(totals, 50))}, p95 {_ms(percentile(totals, 95))}\n"
        "post-inference residual (serialization + stream + network):\n"
        f"  median {_ms(residual_median)}, "
        f"p95 {_ms(percentile(residuals, 95))}\n\n"
        "Recommended camera_debug setting (median residual; elapsed_s and "
        "frame_age are added separately):\n"
        f"  export GP8_PERCEPTION_LATENCY_S={recommendation:.6f}",
        flush=True,
    )
    if residual_median < -0.005:
        print(
            "WARNING: residual delay is negative. Check that the perception "
            "producer's timestamp marks inference start and elapsed_s is accurate.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
