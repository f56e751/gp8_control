#!/usr/bin/env python3
"""정적(멈춘 벨트) 연속 pick-and-throw 테스트 — robust_throw(NLP) 전용.

멈춘 컨베이어 벨트 위 **지정한 지점들**에 물체를 놓아두고, 순서대로
  hover(안전 높이) → suction ON → press(지점 z까지 하강) → PICK_TIME 유지
  → NLP 던지기(지정 target) → 다음 지점 위로 chain
을 반복한다. 벨트/카메라 없이 돌므로 ambush 대기·intercept 선택이 없다 —
RobustThrowSkill의 공개 계획 스테이지(plan_nlp_throw / build_throw_trajectory)를
직접 호출하고, 던지기는 실기와 동일한 timed-release 스트림으로 나간다.

**모든 계산은 로봇이 움직이기 전에 끝난다**: 시작 확인(Enter) 전에 전 지점의
IK + NLP 계획 + 접근/하강/던지기(chain 포함) 궤적을 전부 계산·저장해두고
(_precompute_cycles), 실행 단계는 저장된 궤적을 순서대로 디스패치만 한다 —
사이클 사이에 계획 대기가 없다. 정적 환경이라 가능한 구조다 (사이클 i의 시작
자세 = 사이클 i-1 던지기의 chain 끝, 결정적).

실행 (adv4ncr driver 스택이 떠 있어야 하고, gp8_manager 앱은 꺼져 있어야 함 —
둘 다 JointGroupPositionController에 명령을 쓰면 충돌한다):

  PYTHONPATH=$HOME/ros2_ws/src ~/ros2_ws/src/gp8_control/.venv/bin/python \\
    -m gp8_control.tests.static_pick_throw \\
    --points "0.45,0.20,0.062;0.45,-0.10,0.10" --target "1.2,0,0;1.6,0,0"

--target 은 1개(전 지점 공통) 또는 지점 수와 같은 개수(지점별)로 준다.

  --plan-only   로봇/석션에 아무 명령도 보내지 않고 IK + NLP 계획 + 궤적
                조립(로봇 한계 게이트 포함)까지만 검증. ROS 노드를 띄우지는
                않지만 import 때문에 ROS 환경 소싱은 필요하다.
  --vel-scale   포지셔닝(비-던지기) 이동 속도 스케일 (기본 0.3)
  --no-confirm  사이클마다 Enter 확인 생략 (연속 실행)

⚠️ 던지기는 실제 풀 스윙이다 — 로봇 반경 1 m 확보. Ctrl-C/종료 시 항상
suction OFF를 전송한다. 지점은 "x,y,z" — **z가 곧 프레스 최종 TCP 높이**다
(오프셋 없이 그대로 그 높이까지 내려가 누른다; z 생략 시 Config.GRASP_Z).
물체(place)는 press z + PLACE_ABOVE_PRESS(0.05 m) 높이에 놓인 것으로 보고,
잡기·hover·던지기 lift 기준은 place 높이를 쓴다 — 즉 place−press 만큼 누른다.
hover는 max(place, GRASP_Z)+HOVER_ABOVE라 접근 중 벨트를 긁지 않는다.
유지시간은 스킬 상수 PICK_TIME.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time

import numpy as np

from gp8_control.config import Config
from gp8_control.robots.gp8 import GP8
from gp8_control.skills.context import SkillContext
from gp8_control.skills.robust_throw_skill import (
    HOVER_ABOVE,
    PICK_TIME,
    RobustThrowSkill,
)
from gp8_control.trajectory.trajectory_primitive import trajectory

# 벨트 위 물체를 위에서 잡는 툴 자세 (detection_intake._R_GRASP_DEFAULT와 동일).
_R_GRASP = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

# place(물체가 놓인 TCP 높이) = press 지점 z + 이 값 [m].
# --points 의 z는 "프레스 최종 높이"이고, 물체 자체는 그보다 이만큼 위에 놓여
# 있다고 본다 — hover/잡기 기준(T_grasp)과 던지기 lift 기준은 place 높이를 쓰고,
# 하강만 press z까지 내려가 그 차이만큼 눌러 압축한다.
PLACE_ABOVE_PRESS: float = 0.05


def _make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float).ravel()
    return T


# ---------------------------------------------------------------------------
# --plan-only 대역들 (ROS 없이 계획 경로 전체를 태우기 위한 최소 구현)
# ---------------------------------------------------------------------------
class _PrintLogger:
    def info(self, m):  print(f"[INFO] {m}")
    def warn(self, m):  print(f"[WARN] {m}")
    def error(self, m): print(f"[ERR ] {m}")


class _FakeNode:
    def get_logger(self):
        return _PrintLogger()


class _FakeConveyor:
    """정적 테스트: 벨트 정지 (belt-follow가 None으로 폴백하는 값)."""
    current = 0.0

    def check_freshness(self):
        pass


class _CollectorTrajCtrl:
    """--plan-only: 디스패치를 실행하는 대신 기록·요약한다 (로봇 무명령 보장)."""

    def __init__(self) -> None:
        self.dispatches: list[dict] = []
        self.current_joints = None

    def send_trajectory_queue_with_timed_release(self, traj, vel, ts, final_joint,
                                                 release_index, suction_on_at=None):
        self.dispatches.append(dict(
            kind="throw", n=int(np.asarray(traj).shape[1]),
            T=float(np.asarray(ts)[-1]), release_index=int(release_index),
        ))
        print(f"    [collector] throw dispatch: {self.dispatches[-1]}")
        return False

    def send_trajectory_queue(self, traj, vel, ts, final_joint=None):
        self.dispatches.append(dict(
            kind="move", n=int(np.asarray(traj).shape[1]),
            T=float(np.asarray(ts)[-1]),
        ))
        return True

    def suction_on(self):  print("    [collector] suction ON")
    def suction_off(self): print("    [collector] suction OFF")


class _CaptureThrowCtrl:
    """사전 계산 단계: build_throw_trajectory의 디스패치를 가로채 궤적을 저장.

    스킬의 조립 코드(로봇 한계 게이트 포함)를 그대로 태우되 로봇으로는 아무
    것도 보내지 않는다 — 저장된 배열을 실행 단계에서 진짜 컨트롤러로 재생한다.
    """

    def __init__(self) -> None:
        self.throw: dict | None = None

    def send_trajectory_queue_with_timed_release(self, traj, vel, ts, final_joint,
                                                 release_index, suction_on_at=None):
        self.throw = dict(
            traj=np.asarray(traj, dtype=float), vel=np.asarray(vel, dtype=float),
            ts=np.asarray(ts, dtype=float),
            final_joint=np.asarray(final_joint, dtype=float),
            release_index=int(release_index),
        )
        return False


def _precompute_cycles(ctx, skill, robot, cfg, points, targets, idle_joint,
                       pos_M1, pos_M2):
    """로봇이 움직이기 전에 전 사이클의 IK + NLP + 모든 세그먼트 궤적을 계산.

    정적 환경이라 사이클 i의 시작 자세 = 사이클 i-1 던지기의 chain 끝(결정적)
    이므로 접근 궤적까지 전부 미리 만들 수 있다. NLP 실패 지점은 경고 후
    제외하고 chain을 이어 붙인다. 리턴: cycle dict 리스트
      (no, point, target, q_grasp/q_hover/q_press, res, throw{배열+release_index},
       expected_start, approach, descent)
    """
    zero6 = np.zeros(6)

    # 1) IK + NLP 계획 (전 지점; 실패는 제외)
    viable = []
    for i, (x, y, z) in enumerate(points):
        p_target = targets[i]
        T_grasp, q_grasp, q_hover, q_press = _solve_point(skill, robot, cfg, x, y, z)
        if q_grasp is None or q_hover is None or q_press is None:
            sys.exit(f"point {i + 1} ({x},{y},{z}) IK 실패 — 지점을 조정하라")
        t0 = time.time()
        planned = skill.plan_nlp_throw(q_press, T_grasp, p_target)
        if planned is None:
            print(f"  [{i + 1}] ({x:+.3f}, {y:+.3f}, {z:+.3f}) NLP 계획 실패 — 제외")
            continue
        res, lift = planned
        print(f"  [{i + 1}] ({x:+.3f}, {y:+.3f}, {z:+.3f}) → "
              f"({p_target[0]:+.2f}, {p_target[1]:+.2f}, {p_target[2]:+.2f})  "
              f"plan {time.time() - t0:.1f}s  t_f={res['t_f']:.3f}s  J={res['J']:.3f}")
        viable.append(dict(no=i + 1, point=(x, y, z), target=p_target,
                           q_grasp=q_grasp, q_hover=q_hover, q_press=q_press,
                           res=res, lift=lift))
    if not viable:
        sys.exit("계획 가능한 지점이 없음")

    # 2) 던지기 궤적 캡처(디스패치 가로채기) + 접근/하강 궤적을 chain 순서로
    real_ctrl = ctx.traj_ctrl
    cap = _CaptureThrowCtrl()
    ctx.traj_ctrl = cap
    try:
        prev_end = np.asarray(idle_joint, dtype=float)
        kept, skipped = [], []
        for j, c in enumerate(viable):
            nxt = viable[j + 1]["q_grasp"] if j + 1 < len(viable) else None
            cap.throw = None
            skill.build_throw_trajectory(c["q_press"], c["res"], c["lift"],
                                         next_grasp=nxt)
            if cap.throw is None:
                # 안전 게이트(관절 한계 / Cartesian 엔벨로프)에 걸린 사이클은
                # **통째로 제외**한다 — 접근·하강·흡착도 만들지 않으므로 실행 단계가
                # 이 지점을 아예 건드리지 않는다. 궤적만 빼고 픽을 남기면 물체를
                # 집은 채 던지지 못하는 상태가 되므로 반드시 사이클 단위로 뺀다.
                # prev_end 도 갱신하지 않아 다음 사이클의 접근이 '직전에 실제로
                # 끝난 자세'에서 이어진다 (연속성 유지 — 안전상 필수).
                x_, y_, z_ = c["point"]
                print(f"  [SKIP] cycle {c['no']} ({x_:+.3f}, {y_:+.3f}, {z_:+.3f}) "
                      f"→ 던지기 궤적이 안전 게이트에 걸림 — 이 지점은 픽도 하지 않음")
                skipped.append(c)
                continue
            c["throw"] = cap.throw
            c["expected_start"] = prev_end
            c["approach"] = trajectory(prev_end, zero6, c["q_hover"], zero6,
                                       pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
            c["descent"] = trajectory(c["q_hover"], zero6, c["q_press"], zero6,
                                      pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
            prev_end = cap.throw["final_joint"]
            kept.append(c)
    finally:
        ctx.traj_ctrl = real_ctrl

    if skipped:
        print(f"\n  ** 안전 게이트로 제외된 지점 {len(skipped)}개 "
              f"(로봇은 나머지 {len(kept)}개만 수행) **")
        for c in skipped:
            x_, y_, z_ = c["point"]
            t_ = c["target"]
            print(f"     cycle {c['no']}: ({x_:+.3f}, {y_:+.3f}, {z_:+.3f}) → "
                  f"({t_[0]:+.2f}, {t_[1]:+.2f}, {t_[2]:+.2f})")
    if not kept:
        sys.exit("안전 게이트를 통과한 사이클이 없음 — 로봇을 움직이지 않고 종료")
    return kept


# ---------------------------------------------------------------------------
# 공통 조립
# ---------------------------------------------------------------------------
def _build_ctx_and_skill(cfg, robot, node, traj_ctrl):
    """실기/plan-only 공용: 최소 SkillContext + RobustThrowSkill.

    planner=None — 이 테스트는 고정 target만 쓰므로 legacy fallback
    (plan_throw_landing)이 절대 호출되지 않는다. queue는 빈 실물(TrackedObjectQueue)
    을 줘서 chain의 `not ctx.queue` 분기(빈 큐 → idle 복귀)가 실기처럼 동작한다.
    """
    from gp8_control.tracking import TrackedObjectQueue

    M1 = np.asarray(robot.velocity_limits, dtype=float) * cfg.JOINT_VEL_LIMIT_SCALE
    M2 = M1 * cfg.JOINT_ACCEL_LIMIT_SCALE

    idle_T = _make_transform(cfg.INITIAL_R, cfg.INITIAL_T)
    idle_joint = robot.inverse_kinematics(idle_T)
    if idle_joint is None:
        sys.exit("idle/initial pose IK 실패 (cfg.INITIAL_R/T)")
    idle_joint = np.asarray(idle_joint, dtype=float)
    idle_joint[-1] = 0.0

    noop = lambda *a, **k: None  # noqa: E731
    ctx = SkillContext(
        cfg=cfg, node=node, robot=robot, traj_ctrl=traj_ctrl,
        planner=None, conveyor=_FakeConveyor(),
        queue=TrackedObjectQueue(cfg.MAX_REACH, drop_below_y=-cfg.MAX_REACH),
        M1=M1, M2=M2,
        intake=noop, publish_state=noop, set_status=noop, set_active_target=noop,
        skill_for=lambda obj: "robust_throw",
        skill_obj_for=lambda obj: skill,
        idle_joint=idle_joint,
    )
    skill = RobustThrowSkill(ctx)
    return ctx, skill, M1, M2, idle_joint


def _solve_point(skill, robot, cfg, x: float, y: float, z: float):
    """한 지점의 (T_grasp, q_grasp, q_hover, q_press) IK. 실패 항목은 None.

    ``z``는 프레스가 내려가는 **최종 TCP 높이 그대로**다 (오프셋 없음 — 운영자가
    지점별로 직접 지정/튜닝). 물체는 place = z + PLACE_ABOVE_PRESS 에 놓여 있다고
    보고, 잡기 기준(T_grasp — 던지기 lift 기준 포함)은 place 높이로 잡는다.
    hover는 max(place, GRASP_Z) + HOVER_ABOVE 라 낮은 z를 줘도 접근 중 벨트를
    긁지 않고, 하강할 때만 press z까지 내려가 place−press 만큼 누른다.
    """
    place_z = z + PLACE_ABOVE_PRESS
    T_grasp = _make_transform(_R_GRASP, [x, y, place_z])
    q_grasp = robot.inverse_kinematics(T_grasp)
    if q_grasp is None:
        return T_grasp, None, None, None
    q_grasp = np.asarray(q_grasp, dtype=float)
    hover_z = max(place_z, cfg.GRASP_Z) + HOVER_ABOVE
    q_hover = skill._tcp_z_joint(T_grasp, hover_z, q_grasp)
    q_press = skill._tcp_z_joint(T_grasp, z, q_grasp)
    return T_grasp, q_grasp, q_hover, q_press


def _parse_targets(spec: str, n_points: int) -> list[np.ndarray]:
    """'x,y,z;x,y,z;...' → 지점별 던지기 목표. 1개만 주면 전 지점 공통."""
    tgts = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        vals = [float(v) for v in tok.split(",")]
        if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
            sys.exit(f"--target 항목 '{tok}' 은 유한한 x,y,z 여야 함")
        tgts.append(np.asarray(vals, dtype=float))
    if not tgts:
        sys.exit("--target 비어 있음")
    if len(tgts) == 1:
        return tgts * n_points
    if len(tgts) != n_points:
        sys.exit(f"--target 개수({len(tgts)})가 지점 개수({n_points})와 다름 "
                 "(1개 주면 전 지점 공통)")
    return tgts


def _parse_points(spec: str, default_z: float) -> list[tuple[float, float, float]]:
    """'x,y,z;x,y,z;...' → [(x,y,z)]. z 생략(x,y)이면 ``default_z``(GRASP_Z)."""
    pts = []
    for tok in spec.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        vals = [float(v) for v in tok.split(",")]
        if len(vals) == 2:
            vals.append(default_z)
        if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
            sys.exit(f"--points 항목 '{tok}' 은 x,y,z (또는 x,y) 여야 함")
        pts.append(tuple(vals))
    if not pts:
        sys.exit("--points 비어 있음")
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="정적(멈춘 벨트) 연속 pick-and-throw 테스트 (robust_throw)")
    ap.add_argument("--points", default="0.45,0.20,0.062;0.45,-0.10,0.062",
                    help='벨트 위 물체 지점들 "x,y,z;x,y,z;..." [m base] — '
                         "z는 프레스가 내려가는 최종 TCP 높이 (z 생략 시 GRASP_Z)")
    ap.add_argument("--target", default="1.2,0,0",
                    help='던지기 착지 목표 [m base]: "x,y,z" (전 지점 공통) 또는 '
                         '지점별 "x,y,z;x,y,z;..." (지점 수와 일치해야 함)')
    ap.add_argument("--vel-scale", type=float, default=0.03,
                    help="던지기 외 모든 이동(접근/하강/init + lift/chain)의 속도 "
                         "스케일 — NLP 스윙 아크 자체는 영향 없음")
    ap.add_argument("--shuffle-points", action="store_true",
                    help="지점 방문 순서를 무작위로 섞음 (target 순서는 고정 — "
                         "i번째 사이클 = 섞인 i번째 지점 + i번째 target)")
    ap.add_argument("--seed", type=int, default=None,
                    help="--shuffle-points 재현용 시드 (생략 시 매 실행 다름)")
    ap.add_argument("--plan-only", action="store_true",
                    help="로봇/석션 무명령 — IK+NLP+궤적 조립만 검증")
    ap.add_argument("--no-confirm", action="store_true",
                    help="사이클별 Enter 확인 생략")
    ap.add_argument("--confirm-throw", action="store_true",
                    help="grasp(흡착)+유지 후 Enter 를 눌러야 던진다 "
                         "(장치 on; 기본 off = 흡착 즉시 던짐). q 입력 시 그 사이클 "
                         "던지기 생략(물체 쥔 채 놓기)")
    args, _ros = ap.parse_known_args()

    if not 0.0 < args.vel_scale <= 1.0:
        sys.exit("--vel-scale 은 (0, 1]")

    # NLP solve(BLAS/MUMPS)가 스레드를 코어 수만큼 띄우면 ros2_control의 4ms RT
    # UDP 루프가 굶어 컨트롤러가 RUN_STALL(STATE 10) → comm-loss → hardware
    # error로 넘어간다 (2026-07-21 실기 재현). 계산은 1스레드로 제한 —
    # solve 시간 손해는 미미하고(스레드 경합 제거로 오히려 빨라지기도), RT는 보호된다.
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass

    cfg = Config()
    robot = GP8()
    points = _parse_points(args.points, default_z=cfg.GRASP_Z)
    targets = _parse_targets(args.target, len(points))
    if args.shuffle_points:
        # 지점 순서만 섞는다 — targets는 그대로 두므로 "i번째 던지기의 목표"는
        # 고정되고, 어느 물리적 지점을 i번째로 방문하는지만 무작위가 된다.
        seed = args.seed if args.seed is not None else random.randrange(1 << 31)
        random.Random(seed).shuffle(points)
        print(f"[shuffle] 지점 순서 무작위화 (seed={seed} — --seed로 재현 가능)")
        for i, (x, y, z) in enumerate(points):
            t = targets[i]
            print(f"  {i + 1}. ({x:+.3f}, {y:+.3f}, {z:+.3f}) → "
                  f"target ({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f})")

    # ---------------- plan-only: ROS 없이 전체 계획 경로 ----------------
    if args.plan_only:
        node, traj_ctrl = _FakeNode(), _CollectorTrajCtrl()
        ctx, skill, M1, M2, idle_joint = _build_ctx_and_skill(
            cfg, robot, node, traj_ctrl)
        # 실기와 동일 조건으로 검증: lift/chain도 vel_scale 감속 반영
        ctx.M1 = ctx.M1 * args.vel_scale
        ctx.M2 = ctx.M1 * cfg.JOINT_ACCEL_LIMIT_SCALE
        print(f"\n=== PLAN-ONLY: {len(points)}개 지점 (지점별 target) ===")
        ok = 0
        for i, ((x, y, z), p_target) in enumerate(zip(points, targets)):
            print(f"\n[{i + 1}/{len(points)}] point ({x:+.3f}, {y:+.3f}, {z:+.3f}) "
                  f"→ target ({p_target[0]:+.2f}, {p_target[1]:+.2f}, {p_target[2]:+.2f})")
            T_grasp, q_grasp, q_hover, q_press = _solve_point(skill, robot, cfg, x, y, z)
            if q_grasp is None or q_hover is None or q_press is None:
                print("    IK 실패 — 지점 조정 필요");  continue
            t0 = time.time()
            planned = skill.plan_nlp_throw(q_press, T_grasp, p_target)
            if planned is None:
                print(f"    NLP 계획 실패 ({time.time() - t0:.1f}s)");  continue
            res, lift = planned
            nxt = None
            if i + 1 < len(points):
                _, nq, _, _ = _solve_point(skill, robot, cfg, *points[i + 1])
                nxt = nq
            skill.build_throw_trajectory(q_press, res, lift, next_grasp=nxt)
            ok += 1
            print(f"    OK: plan {time.time() - t0:.1f}s, t_f={res['t_f']:.3f}s, "
                  f"J={res['J']:.3f}")
        n_throw = sum(1 for d in traj_ctrl.dispatches if d["kind"] == "throw")
        print(f"\n계획 성공 {ok}/{len(points)}, throw dispatch {n_throw}건 (로봇 무명령)")
        return

    # ---------------- 실기 모드 ----------------
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from gp8_control.controllers.trajectory_controller import TrajectoryController

    rclpy.init()
    node = Node("static_pick_throw_test")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    traj_ctrl = TrajectoryController(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()

    suction_used = False
    try:
        if not traj_ctrl.wait_for_servers(timeout_sec=10.0):
            sys.exit("컨트롤러 액션 서버 없음 — driver 스택(bringup)이 떠 있는지 확인")
        t_end = time.time() + 5.0
        while traj_ctrl.current_joints is None and time.time() < t_end:
            time.sleep(0.05)
        if traj_ctrl.current_joints is None:
            sys.exit("/joint_states 미수신 — driver 스택 확인")

        ctx, skill, M1, M2, idle_joint = _build_ctx_and_skill(
            cfg, robot, node, traj_ctrl)
        # 던지기 외 전부 감속: 접근/하강/init 이동(pos_M1)뿐 아니라, 던지기
        # 궤적에 포함되는 lift(프레스→스윙 시작)·chain(스윙 후 park) 세그먼트도
        # ctx.M1로 만들어지므로 함께 vel_scale로 줄인다. NLP 스윙 아크는 solver
        # 자체 한계로 계획되고 lift/chain은 정지-정지 세그먼트라 감속해도
        # 던지기(스윙 속도/release/착지)에는 영향이 없다.
        ctx.M1 = ctx.M1 * args.vel_scale
        ctx.M2 = ctx.M1 * cfg.JOINT_ACCEL_LIMIT_SCALE
        pos_M1, pos_M2 = ctx.M1, ctx.M2
        zero6 = np.zeros(6)

        # ---------- 시작: init(idle) 자세로 먼저 이동 ----------
        # 사전 계산(수 초~수십 초) 동안 팔이 작업 공간을 비우고 알려진 자세에
        # 정지해 있도록, 다른 무엇보다 먼저 init으로 이동한다. 이후의 접근
        # 체인도 idle에서 출발하는 것으로 계산된다.
        q_start = np.asarray(traj_ctrl.current_joints, dtype=float)
        move_to_idle = trajectory(q_start, zero6, idle_joint, zero6,
                                  pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
        print(f"\n→ init(idle) 자세로 이동 ({move_to_idle[2][-1]:.2f}s)")
        traj_ctrl.send_trajectory_queue(
            move_to_idle[0], move_to_idle[1], move_to_idle[2],
            final_joint=idle_joint)

        # ---------- Phase A: 사전 계산 (init에 정지한 상태, suction 이전) ----------
        print(f"\n=== 사전 계산: 전체 궤적 ({len(points)}개 지점) ===")
        t_pre0 = time.time()
        cycles = _precompute_cycles(ctx, skill, robot, cfg, points, targets,
                                    idle_joint, pos_M1, pos_M2)
        print(f"사전 계산 완료: {len(cycles)}/{len(points)} 사이클, "
              f"{time.time() - t_pre0:.1f}s — 실행 중에는 계획/계산 없음")

        print(f"\n물체를 지점에 놓고 시작하라. 던지기는 실제 스윙 (반경 1 m 확보).")
        if input("전체 시퀀스 시작 Enter (q 취소): ").strip().lower() == "q":
            return

        # ---------- Phase B: 실행 (저장된 궤적 디스패치만; 팔은 이미 init) ----------
        suction_used = True
        traj_ctrl.suction_off()

        results = []
        for j, c in enumerate(cycles):
            x, y, z = c["point"]
            t = c["target"]
            print(f"\n===== [{j + 1}/{len(cycles)}] point "
                  f"({x:+.3f}, {y:+.3f}, {z:+.3f}) → target "
                  f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) =====")
            if not args.no_confirm:
                ans = input("이 사이클 실행 Enter (s 건너뜀, q 종료): ").strip().lower()
                if ans == "q":
                    break
                if ans == "s":
                    results.append((c["no"], "SKIPPED"))
                    continue

            # 접근: 사전 계산 궤적. 직전 사이클을 skip했거나 어떤 이유로 시작
            # 자세가 예상과 어긋나면 접근만 현재 자세에서 재생성한다 (계획·던지기
            # 궤적은 시작 자세와 무관하므로 그대로 유효).
            a_traj, a_vel, a_ts = c["approach"]
            q_cur = np.asarray(traj_ctrl.current_joints, dtype=float)
            if float(np.max(np.abs(q_cur - c["expected_start"]))) > 0.05:
                print("  (시작 자세가 예상과 다름 — 접근 궤적만 재생성)")
                a_traj, a_vel, a_ts = trajectory(q_cur, zero6, c["q_hover"], zero6,
                                                 pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
            print(f"→ hover 이동 ({a_ts[-1]:.2f}s)")
            traj_ctrl.send_trajectory_queue(a_traj, a_vel, a_ts,
                                            final_joint=c["q_hover"])

            traj_ctrl.suction_on()
            d_traj, d_vel, d_ts = c["descent"]
            print(f"→ press 하강 z={z:+.3f}m ({d_ts[-1]:.2f}s) + "
                  f"{PICK_TIME:.1f}s 유지")
            traj_ctrl.send_trajectory_queue(d_traj, d_vel, d_ts,
                                            final_joint=c["q_press"])
            time.sleep(PICK_TIME)

            # --confirm-throw: 흡착·유지 후 조작자가 Enter 를 눌러야 던진다. 이
            # 구간 동안 컵은 q_press 자세를 유지(JGPC zero-order-hold)하며 물체를
            # 계속 누르고 있으므로, 물체 장착/정렬 확인 뒤 던질 수 있다. suction 은
            # 이미 ON. q 입력 시 이 사이클의 던지기를 생략하고 물체를 놓는다.
            if args.confirm_throw:
                ans = input("  물체 grasp 완료 — 던지려면 Enter (q 던지기 생략): "
                            ).strip().lower()
                if ans == "q":
                    print("  던지기 생략 — suction OFF (물체 놓음)")
                    traj_ctrl.suction_off()
                    results.append((c["no"], "THROW-SKIPPED"))
                    continue

            th = c["throw"]
            print(f"→ THROW (t_f={c['res']['t_f']:.3f}s, J={c['res']['J']:.3f}, "
                  f"release idx {th['release_index']}/{th['traj'].shape[1] - 1})")
            traj_ctrl.send_trajectory_queue_with_timed_release(
                th["traj"], th["vel"], th["ts"],
                final_joint=th["final_joint"],
                release_index=th["release_index"])
            traj_ctrl.suction_off()   # timed release 이후 안전 재확인
            results.append((c["no"], "OK"))

        # 종료: idle 복귀
        q_cur = np.asarray(traj_ctrl.current_joints, dtype=float)
        q_traj, q_vel, ts = trajectory(q_cur, zero6, idle_joint, zero6,
                                       pos_M1, pos_M2, hertz=cfg.TRAJ_HZ)
        print(f"\n→ idle 복귀 ({ts[-1]:.2f}s)")
        traj_ctrl.send_trajectory_queue(q_traj, q_vel, ts, final_joint=idle_joint)

        print("\n===== 결과 =====")
        for n, st in results:
            print(f"  cycle {n}: {st}")
    except KeyboardInterrupt:
        print("\n중단됨 (Ctrl-C)")
    finally:
        if suction_used:
            try:
                traj_ctrl.suction_off()
                print("suction OFF 전송됨")
            except Exception:                     # noqa: BLE001
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
