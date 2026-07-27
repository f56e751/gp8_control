"""Camera-based conveyor speed estimator — encoder-less alternative.

``ConveyorSpeedTracker``(엔코더 구독)의 drop-in 대체: 같은 ``current`` /
``check_freshness()`` 표면을 제공하되, 엔코더 토픽을 구독하는 대신 **지나가는
물체들의 속도 피팅을 집계**해 벨트 속도를 추론한다.

fit 공급원은 ``DetectionIntake._update_velocity`` — 물체가 카메라 박스를
지나는 동안 (t, y) 앵커를 최소자승 피팅한 물체별 속도로, 품질 게이트(앵커
수/시간 span/잔차 RMS)를 통과한 값만 ``observe(v_fit, track_id)``로 들어온다
(벨트-밴드 클램프 **이전** 값 — 클램프 기준이 이 트래커의 current라서 순환을
끊기 위함).

갱신 정책: **물체 ``batch_n``개(기본 3)마다 1회**. 물체가 박스를 지나는 동안
fit이 계속 정제되므로 물체별 '최신 fit'을 보류(pending)해 두고, 서로 다른
물체가 batch_n개 모이면 그들의 중앙값으로 ``current``를 갱신하고 그 물체들은
소진(done) 처리한다 — 한 물체가 두 배치에 중복 기여하지 않는다.

추정치를 CONVEYOR_TOPIC(``/conveyor/speed``)에 **발행도 한다** — 엔코더가
하던 역할을 대신해, camera_debug의 v×delay 역보정과 belt_viz 등 하위
소비자가 수정 없이 동작한다. (이 모드에서 엔코더 노드를 같이 켜면 발행이
충돌한다 — 엔코더 없는 리그 전용.)

선택: Config.CONVEYOR_SOURCE = "camera" (env GP8_CONVEYOR_SOURCE / launch
conveyor_source:=camera). 기본은 "encoder" — 기존 동작 그대로.
"""

from __future__ import annotations

import time

import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64


class CameraSpeedTracker:
    def __init__(
        self,
        node: Node,
        topic: str,
        fallback_speed: float,
        stale_seconds: float,
        batch_n: int = 3,
    ) -> None:
        self._node = node
        self._topic = topic
        self._stale_seconds = stale_seconds
        self._speed = float(fallback_speed)
        self._batch_n = max(1, int(batch_n))
        self._pending: dict = {}             # track_id -> 최신 fit (배치 대기)
        self._done: set = set()              # 이미 배치에 소진된 track_id
        self._last_fit_time: float | None = None
        self._stale_warned = False
        self._acquired = False
        # 엔코더 대체 발행 (하위 소비자 호환용).
        self._pub = node.create_publisher(Float64, topic, 1)
        node.get_logger().info(
            f"Camera belt-speed mode: {self._batch_n}개 물체마다 갱신 -> {topic} "
            f"발행 (fallback {fallback_speed:.3f} m/s until first batch)"
        )

    @property
    def current(self) -> float:
        return self._speed

    def observe(self, v_fit: float, track_id) -> None:
        """DetectionIntake가 한 물체의 품질-통과 속도 fit을 보고할 때 호출.

        물체가 박스 안에 있는 동안 fit이 정제되며 여러 번 오므로, 배치가 차기
        전까지는 그 물체의 pending 값을 최신으로 갈아끼우기만 한다.
        """
        self._last_fit_time = time.time()
        self._stale_warned = False
        if track_id not in self._done:
            self._pending[track_id] = float(v_fit)
            if len(self._pending) >= self._batch_n:
                vals = list(self._pending.values())
                self._speed = float(np.median(vals))
                self._done.update(self._pending.keys())
                ids = sorted(self._pending.keys())
                self._pending.clear()
                if not self._acquired:
                    self._acquired = True
                self._node.get_logger().info(
                    f"Camera belt speed 갱신: {self._speed:.4f} m/s "
                    f"(물체 {self._batch_n}개 id={ids}, "
                    f"fits [" + " ".join(f"{v:.3f}" for v in vals) + "])"
                )
                # done set이 무한히 자라지 않게 적당히 잘라낸다 (id는 단조 증가).
                if len(self._done) > 200:
                    self._done = set(sorted(self._done)[-100:])
        # 배치 사이에도 토픽은 살아 있게 현재 추정치를 계속 발행.
        self._pub.publish(Float64(data=float(self._speed)))

    def check_freshness(self) -> None:
        """물체가 한동안 안 지나가면 1회 경고 (마지막 추정값 유지)."""
        if self._last_fit_time is None or self._stale_warned:
            return
        if time.time() - self._last_fit_time > self._stale_seconds:
            self._node.get_logger().warn(
                f"No passing-object speed fit for >{self._stale_seconds:.1f}s; "
                f"holding {self._speed:.4f} m/s"
            )
            self._stale_warned = True
