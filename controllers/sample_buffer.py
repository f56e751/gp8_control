# Ported unchanged (except this header) from the teammate's HW-validated
# rge_jc/hwpkg/gp8_control/robot/sample_buffer.py (2026-11-04, controller-clock
# buffer client for motoman_controllers/SampleBuffer). Transport-agnostic: the
# ROS binding lives in controllers/trajectory_controller.py.
"""컨트롤러 클록 버퍼 스트림 (11-04 사용자: "Python time.sleep 을 없애면 안돼?") — 전송 계층과 무관한 앱 쪽 절반.

종전: 앱이 2 ms 마다 time.sleep 으로 박자를 맞춰 샘플 하나씩 보냈다 → 송신 간격 1.7~2.8 ms, 한 틱에 두 샘플(증분 2 배 → CODE 112)을
막으려 늦추고(_win_time) 느린 구간에서 따라잡아, 한 판에 130~780 ms 가 쌓였다 (11-04 실물 로그).
지금: 샘플을 **절대 틱 번호**로 컨트롤러(실물: ros2_control SampleBuffer, 시뮬: Isaac 월드) 버퍼에 미리 쓴다. 컨트롤러가 자기 2 ms 틱마다
하나씩 내보내므로 샘플 k 는 정확히 틱 k 에 나간다. 앱은 진행 번호(progress)만 본다 — 박자·따라잡기·재정렬이 없다.

쓰기 규칙 (컨트롤러와 같음): chunk(start, Q) → buffer[start .. start+n−1] = Q, 그 뒤는 버림 (꼬리 교체). start < 지금 틱 + min_lead 면 거부
(1), 이음새 포함 속도·가속 한계 밖이면 거부 (2). stop → 지금 틱 뒤를 비워 그 명령에 머묾. 버퍼가 끝나면 마지막 명령에 머문다.

이 클래스는 전송을 모른다: 백엔드가 send_chunk/send_stop(/send_events) 를 주고, 진행 메시지를 on_progress 로 넘긴다.
"""
from __future__ import annotations

import collections
import itertools
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

TS = 0.002
RES_OK, RES_LATE, RES_LIMIT, RES_MALFORMED, RES_OVERFLOW = 0, 1, 2, 3, 4
RES_TIMEOUT = -1
NOW_AHEAD_MAX = 12                        # now_index 가 마지막 진행보다 앞설 수 있는 틱 (진행 전달 지연 몫; 진행은 5 틱마다 = progress_every)
RES_NAMES = {RES_OK: "ok", RES_LATE: "too late", RES_LIMIT: "limit", RES_MALFORMED: "malformed", RES_OVERFLOW: "overflow",
             RES_TIMEOUT: "no ack"}


class BufferClient:
    """진행 번호 → 틱 시계, 쓰기·확인, 석션 이벤트(native 가 아니면 진행 번호를 보고 여기서 쏜다)."""

    def __init__(self, send_chunk: Callable[[int, int, np.ndarray], None], send_stop: Callable[[int], None],
                 fire: Callable[[bool], None], send_events: Optional[Callable[[int, List[Tuple[int, bool]]], None]] = None,
                 clock_window_s: float = 1.0, name: str = "buf"):
        self._send_chunk, self._send_stop, self._send_events, self._fire = send_chunk, send_stop, send_events, fire
        self.name = name
        self._cv = threading.Condition()
        self._seq = itertools.count(1)
        self.index = None                     # 마지막 진행 틱
        self.buffer_end = None
        self.holding = True
        self.t_last = 0.0                     # 마지막 진행을 받은 monotonic
        self.command = None
        self._acks = {}                       # seq → result (최근 것만)
        self._ack_order = collections.deque(maxlen=256)
        # 틱 시계: t(k) = base + k·TS. base = 창 안 (도착 − k·TS) 최솟값 (가장 덜 늦게 온 진행) — 정확한 시각(stamp)이 오면 그것
        self._clock_win = float(clock_window_s)
        self._mins = collections.deque()      # (도착, 후보) 단조 최소 큐
        self.base = None
        # 석션 이벤트 (native 가 아니면): 틱 번호 → on. from_index 이후를 통째로 바꾼다.
        self._events: List[Tuple[int, bool]] = []
        self.ev_log: List[Tuple[int, bool, float]] = []   # (틱, on, 쏜 monotonic)
        self.tick_hooks: List[Callable[[int, float, Optional[Sequence[float]]], None]] = []   # (틱, 틱 시각, 명령) — 기록·추종 감시
        self.n_progress = 0
        self.n_index_gap = 0                  # 진행 번호가 건너뛴 횟수 (전송이 몰려 놓친 진행 — 틱 자체는 컨트롤러가 냈다)
        if send_events is None:
            threading.Thread(target=self._ev_loop, name=f"{name}-io", daemon=True).start()

    # -- 진행 (백엔드 수신 스레드에서) -----------------------------------------------------------------------------------
    def on_progress(self, index: int, t_arr: float, buffer_end: int, holding: bool, last_seq: int, last_result: int,
                    command: Optional[Sequence[float]] = None, t_stamp: Optional[float] = None) -> None:
        index = int(index)
        t_ref = float(t_stamp) if t_stamp is not None else float(t_arr)
        c = t_ref - index * TS
        with self._cv:
            if self.index is not None and index > self.index + 1:
                self.n_index_gap += 1
            if self.index is not None and index < self.index:      # 컨트롤러 재시작 (틱 0 부터) — 시계·확인을 새로
                self._mins.clear(); self._acks.clear()
            self.index, self.buffer_end, self.holding, self.t_last = index, int(buffer_end), bool(holding), float(t_arr)
            self.command = command
            q = self._mins
            while q and q[-1][1] >= c:
                q.pop()
            q.append((t_ref, c))
            while q[0][0] < t_ref - self._clock_win:
                q.popleft()
            self.base = q[0][1]
            if last_seq and last_seq not in self._acks:
                self._acks[int(last_seq)] = int(last_result); self._ack_order.append(int(last_seq))
                while len(self._acks) > 200:
                    self._acks.pop(self._ack_order.popleft(), None)
            self.n_progress += 1
            self._cv.notify_all()
        for h in self.tick_hooks:
            try:
                h(index, self.base + index * TS, command)
            except Exception:                                      # noqa: BLE001 — 기록·감시가 수신을 죽이면 안 된다
                pass

    # -- 조회 ---------------------------------------------------------------------------------------------------------
    def ready(self, max_age: float = 0.1) -> bool:
        return self.index is not None and time.monotonic() - self.t_last < max_age

    def now_index(self) -> int:
        """지금 막 나간 틱 (진행 지연만큼 예측해 더함 — 마지막 진행 + NOW_AHEAD_MAX 틱까지만: 시계 추정이 틀려도 너무 앞서지 않게)."""
        with self._cv:
            if self.index is None:
                raise RuntimeError("버퍼 컨트롤러 진행이 아직 없음")
            return min(max(self.index, int(np.floor((time.monotonic() - self.base) / TS))), self.index + NOW_AHEAD_MAX)

    def tick_mono(self, k: float) -> float:
        return self.base + float(k) * TS

    def index_at(self, t_mono: float) -> float:
        return (float(t_mono) - self.base) / TS

    def wait_index(self, k: int, timeout: float, check: Optional[Callable[[], None]] = None) -> bool:
        """진행 번호가 k 이상이 될 때까지 (check() 는 10 ms 마다 — 예외로 빠져나갈 수 있다)."""
        t_end = time.monotonic() + float(timeout)
        with self._cv:
            while self.index is None or self.index < k:
                rem = t_end - time.monotonic()
                if rem <= 0:
                    return False
                self._cv.wait(min(rem, 0.01))
                if check is not None:
                    self._cv.release()
                    try:
                        check()
                    finally:
                        self._cv.acquire()
            return True

    # -- 쓰기 ---------------------------------------------------------------------------------------------------------
    def write(self, start: int, Q: np.ndarray, timeout: float = 0.1) -> int:
        """버퍼에 쓰고 컨트롤러의 확인을 기다린다 → 결과 코드 (RES_*)."""
        Q = np.ascontiguousarray(np.asarray(Q, float).reshape(-1, 6))
        seq = next(self._seq) & 0xFFFFFFFF or next(self._seq)
        self._send_chunk(seq, int(start), Q)
        t_end = time.monotonic() + timeout
        with self._cv:
            while seq not in self._acks:
                rem = t_end - time.monotonic()
                if rem <= 0:
                    return RES_TIMEOUT
                self._cv.wait(min(rem, 0.005))
            return self._acks[seq]

    def stop(self) -> None:
        """지금 명령에 머문다 (지금 틱 뒤를 비움)."""
        self._send_stop(next(self._seq) & 0xFFFFFFFF or next(self._seq))
        with self._cv:
            self._events = [e for e in self._events if self.index is not None and e[0] <= self.index]
        if self._send_events is not None:
            try:
                self._send_events(int(self.index or 0) + 1, [])
            except Exception:                                      # noqa: BLE001
                pass

    # -- 석션 이벤트 ----------------------------------------------------------------------------------------------------
    def set_events(self, from_index: int, events: Sequence[Tuple[int, bool]]) -> None:
        """from_index 이후 이벤트를 events 로 바꾼다 (틱, on). 지난 틱의 이벤트는 바로 쏜다 (안 쏘는 것보다 늦게라도)."""
        ev = sorted((int(k), bool(on)) for k, on in events)
        if self._send_events is not None:
            self._send_events(int(from_index), ev)
            return
        with self._cv:
            self._events = [e for e in self._events if e[0] < from_index] + ev
            self._cv.notify_all()

    def _ev_loop(self) -> None:
        # 틱 시계로 쏜다 (base + k·TS, 진행 스탬프로 맞춘 컨트롤러 틱 시각): 진행이 5 틱마다 오거나 (progress_every)
        # 앱이 바빠 늦게 받아도 석션 시각이 그만큼 밀리지 않는다. 진행이 이미 그 틱을 넘었으면 바로 쏜다.
        while True:
            with self._cv:
                while True:
                    if self._events and self.index is not None and self.base is not None:
                        k0 = self._events[0][0]
                        rem = self.base + k0 * TS - time.monotonic()
                        if rem <= 0.0 or k0 <= self.index:
                            break
                        self._cv.wait(min(rem, 0.05))
                    else:
                        self._cv.wait(0.05)
                k, on = self._events.pop(0)
            try:
                self._fire(on)
            finally:
                self.ev_log.append((k, on, time.monotonic()))
