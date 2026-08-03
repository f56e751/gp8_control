"""Continuous perception-PC to robot-PC clock synchronization.

The perception server exposes an NTP-style ``/latency`` endpoint.  This module
keeps a low-RTT estimate of ``server clock - client clock`` available to the
camera bridge so producer timestamps can be evaluated on the robot timeline.
"""

from __future__ import annotations

import http.client
import json
import math
import socket
import statistics
import threading
import time
import urllib.parse
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSample:
    rtt_s: float
    offset_s: float  # server clock - client clock


@dataclass(frozen=True)
class ClockEstimate:
    offset_s: float
    min_rtt_s: float
    updated_monotonic_s: float


def calculate_probe(t0: float, t1: float, t2: float, t3: float) -> ProbeSample:
    """Return NTP-style RTT and server-minus-client clock offset."""
    server_work = max(0.0, t2 - t1)
    rtt = max(0.0, (t3 - t0) - server_work)
    offset = ((t1 - t0) + (t2 - t3)) / 2.0
    return ProbeSample(rtt_s=rtt, offset_s=offset)


def estimate_clock_offset(samples: list[ProbeSample]) -> tuple[float, float]:
    """Return (offset, min RTT) using the lowest-RTT fifth of a batch."""
    if not samples:
        raise ValueError("at least one probe sample is required")
    keep = max(1, min(len(samples), max(3, len(samples) // 5)))
    best = sorted(samples, key=lambda sample: sample.rtt_s)[:keep]
    return (
        float(statistics.median(sample.offset_s for sample in best)),
        float(min(sample.rtt_s for sample in samples)),
    )


def latency_url_from_stream(stream_url: str) -> str:
    parsed = urllib.parse.urlsplit(stream_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("perception URL must be an http(s) URL")
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, "/latency", "", "")
    )


def run_probe_batch(url: str, count: int, timeout_s: float) -> list[ProbeSample]:
    """Collect a batch over one warmed TCP connection."""
    parsed = urllib.parse.urlsplit(url)
    conn_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_type(parsed.hostname, parsed.port, timeout=timeout_s)
    conn.connect()
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
    samples: list[ProbeSample] = []
    try:
        for index in range(count + 1):
            separator = "&" if "?" in path else "?"
            t0 = time.time_ns() / 1e9
            conn.request(
                "GET",
                f"{path}{separator}n={index}",
                headers={"Cache-Control": "no-cache"},
            )
            response = conn.getresponse()
            body = response.read()
            t3 = time.time_ns() / 1e9
            if response.status != 200:
                raise RuntimeError(
                    f"latency probe returned HTTP {response.status}"
                )
            payload = json.loads(body)
            if payload.get("protocol") != "gp8-latency-v1":
                raise RuntimeError("server does not support gp8-latency-v1")
            if index == 0:  # warm the persistent connection
                continue
            t1 = float(payload["server_receive_time_ns"]) / 1e9
            t2 = float(payload["server_send_time_ns"]) / 1e9
            samples.append(calculate_probe(t0, t1, t2, t3))
    finally:
        conn.close()
    return samples


class ContinuousClockSync:
    """Refresh a clock-offset estimate on a background thread."""

    def __init__(
        self,
        latency_url: str,
        *,
        interval_s: float = 5.0,
        probes: int = 8,
        timeout_s: float = 2.0,
        max_age_s: float = 15.0,
    ):
        self.latency_url = latency_url
        self.interval_s = max(0.5, float(interval_s))
        self.probes = max(3, int(probes))
        self.timeout_s = max(0.1, float(timeout_s))
        self.max_age_s = max(self.interval_s, float(max_age_s))
        self._lock = threading.Lock()
        self._estimate: ClockEstimate | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="perception-clock-sync", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_s + 0.5)

    def estimate(self) -> ClockEstimate | None:
        with self._lock:
            estimate = self._estimate
        if estimate is None:
            return None
        if time.monotonic() - estimate.updated_monotonic_s > self.max_age_s:
            return None
        if not math.isfinite(estimate.offset_s):
            return None
        return estimate

    def last_error(self) -> str | None:
        with self._lock:
            return self._error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                samples = run_probe_batch(
                    self.latency_url, self.probes, self.timeout_s
                )
                offset, min_rtt = estimate_clock_offset(samples)
                estimate = ClockEstimate(
                    offset_s=offset,
                    min_rtt_s=min_rtt,
                    updated_monotonic_s=time.monotonic(),
                )
                with self._lock:
                    self._estimate = estimate
                    self._error = None
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                with self._lock:
                    self._error = str(exc)
            self._stop.wait(self.interval_s)
