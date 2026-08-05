"""실기 GP8 데모 수집 — 공식 `collect_data_real.py` 의 ROS2 이식.

원본: ThrowBot(MaxorPaxor) `src/scripts/decision_transformer/real_robot/
collect_data_real.py` (저자 Maxim Monastirsky 본인 코드, RA-L 2023 / IEEE 9984828).
`/PublicSSD/ryugaeun/ThrowBot_official/` 에 커밋 43e8c1b 전체를 받아 두었다.

**로직은 원본 그대로**다. 바꾼 것은 인터페이스 두 곳뿐 (2026-08-05 사용자 지시):

  ① rospy → rclpy
     원본은 `JointTrajectory` 를 `/joint_command`(MotoROS point streaming)로
     10 Hz 발행한다. 이 rig 은 `/JointGroupPositionController/commands` 로
     4 ms 위치 스트리밍이라 waypoint 를 받아주는 컨트롤러가 없다. 그래서
     **MotoROS 가 했을 보간을 우리가 직접 돌린다** — `sim/gp8_interp.py` 는
     저자 포크 `MaxorPaxor/motoman_ps` 의 `MotionServer.c:1414-1420`
     (`Ros_MotionServer_JointTrajDataToIncQueue`) 이식이고, 가속도 계수가
     원본과 일치함을 대조 확인했다. 결과적으로 팔이 그리는 곡선은 같다.

  ② OptiTrack → 수동 입력
     원본 `collect_data_real.py:91-92` 에 수동 경로가 주석으로 남아 있다:
         # object_position_ = float(input(f'Target: {x}. Input object position '))
         object_position_ = optitrack.landing_spot
     모션캡처가 없으므로 주석 쪽을 쓴다.

그대로 가져온 것 (원본 줄번호):
  · 목표 그리드 `np.arange(0.5, 2.05, 0.05)`                        (:35)
  · 조건화 목표 리턴 `ep_return = 1.0`                               (:53)
  · `arm.first_step([0,0,0,1])` 로 에피소드 시작                     (:59)
  · 탐색 게인 `action * [amp, amp, amp, 1]` 후 clip                  (:82-83)
  · 처음 2스텝은 그리퍼 강제 닫음                                    (:84-85)
  · 보상 ±1 (`target_radius` 기준), 중간 스텝은 0                    (:97-103)
  · HER K=0 재라벨 (`obj_final_pos` = 실제 착지)                     (:114-115)
  · 에피소드마다 pkl 증분 저장                                       (:120-122)

waypoint 속도 규약은 `dt_gp8_env.WP_VEL_MODE` 를 따른다 (기본 'segment' =
공식 `control_real_gp8.py:107-115`). `DT_GP8_WP_VEL` 로 바꿔 비교할 수 있다.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# 원본 상수 (collect_data_real.py)
# ---------------------------------------------------------------------------
TARGET_LIST = np.arange(0.5, 2.05, 0.05)   # :35  거리 그리드 [m]
AMP_RANGE = [1]                            # :37  탐색 게인 (원본 기본값)
EP_RETURN = 1.0                            # :53  조건화 목표 리턴 R̂
EXPLORATION = True                         # :81


def ask_landing(prompt: str):
    """원본 :91 의 수동 입력 경로. 빈 줄이면 그 던지기를 버린다."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        if raw.lower() == "q":
            raise KeyboardInterrupt
        try:
            return float(raw)
        except ValueError:
            print("    숫자만 입력 (빈 줄 = 버림, q = 종료)")


def calc_dist_from_goal(obj_pos, target):
    """원본 `utils.calc_dist_from_goal` — 수평 거리."""
    return float(np.linalg.norm(np.asarray(obj_pos)[:2] - np.asarray(target)[:2]))


# ---------------------------------------------------------------------------
# 원본 `collect_n_real_data` 의 본체 (:40-124)
# ---------------------------------------------------------------------------
def collect(arm, model, out_path, targets=None, amp_range=None, k_her=0,
            rng=None, resume=True):
    """공식 루프. `arm` 은 아래 GP8RealArm (rclpy 구현) 을 받는다."""
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    rng = rng if rng is not None else np.random.default_rng()
    targets = TARGET_LIST if targets is None else np.asarray(targets, float)
    amp_range = AMP_RANGE if amp_range is None else amp_range

    memory = []
    if resume and os.path.exists(out_path):                       # :120 증분 저장
        with open(out_path, "rb") as f:
            memory = pickle.load(f)
        print(f"이어서 수집: 기존 {len(memory)} 궤적")

    n_episodes = 0
    for x in targets:
        for amp in amp_range:
            target = np.array([float(x), 0.0, 0.0])               # :47
            arm.update_target(target)

            states = torch.zeros((0, model.state_dim), device=device,
                                 dtype=torch.float32)
            actions = torch.zeros((0, model.act_dim), device=device,
                                  dtype=torch.float32)
            rewards = torch.zeros(0, device=device, dtype=torch.float32)
            target_return = torch.tensor(EP_RETURN, device=device,
                                         dtype=torch.float32).reshape(1, 1)
            timesteps = torch.tensor(0, device=device,
                                     dtype=torch.long).reshape(1, 1)
            episode_length = 0

            if not arm.first_step(np.array([0.0, 0.0, 0.0, 1.0])):  # :59
                print(f"  목표 {x:.2f} m: 시작 거부 — 건너뜀")
                continue

            temp_mem, done, object_position = [], False, None
            while not done:
                state_ = arm.get_state()                            # :63
                state = np.append(state_, arm.target[0])            # :64 목표 이어붙임
                state = torch.from_numpy(state).reshape(
                    1, model.state_dim).to(device=device, dtype=torch.float32)
                states = torch.cat([states, state], dim=0)
                actions = torch.cat(
                    [actions, torch.zeros((1, model.act_dim), device=device)],
                    dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])

                action = model.get_action(                          # :70
                    states.to(dtype=torch.float32),
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
                actions[-1] = action
                action_ = action.detach().cpu().numpy()

                if EXPLORATION:                                     # :81-85
                    action_ = action_ * np.array([amp, amp, amp, 1])
                    action_ = np.clip(action_, -1.0, 1.0)
                    if episode_length <= 1 and action_[-1] <= arm.gripper_thresh:
                        action_[-1] = 1.0

                done, termination_reason = arm.step(action_)        # :87

                if done:
                    time.sleep(2)                                   # :90
                    object_position_ = ask_landing(                 # :91 (원본 주석 경로)
                        f'  Target: {x:.2f}. Input object position [m] ')
                    if object_position_ is None:
                        print("    버림")
                        break
                    object_position = np.array([object_position_, 0.0, 0.0])
                    distance_from_target = calc_dist_from_goal(
                        object_position, arm.target)
                    print(f"    Landing Spot: {object_position_:.3f}, "
                          f"Error: {distance_from_target:.3f}")
                    reward_ = 1.0 if distance_from_target < arm.target_radius \
                        else -1.0                                   # :97-100
                else:
                    reward_ = 0.0                                   # :103

                rewards[-1] = reward_
                target_return = torch.cat(
                    [target_return, target_return[0, -1].reshape(1, 1)], dim=1)
                timesteps = torch.cat(
                    [timesteps, torch.ones((1, 1), device=device,
                                           dtype=torch.long) * (episode_length + 1)],
                    dim=1)
                episode_length += 1
                temp_mem.append((state_, action_, reward_, done, reward_ > 0))

            arm.reset_arm()                                          # :117
            if object_position is None:
                continue

            try:                                                      # :114 HER K=0
                from gp8_control.skills.thr.dt_model.her import \
                    generate_her_memory                               # 로봇 벤더링
            except ImportError:
                from agent.her import generate_her_memory             # Thr_DT
            memory.extend(generate_her_memory(
                arm, temp_mem, target=target, obj_final_pos=object_position,
                k=k_her, rng=rng))
            n_episodes += 1

            tmp = out_path + ".tmp"                                  # :120 증분 저장
            with open(tmp, "wb") as f:
                pickle.dump(memory, f)
            os.replace(tmp, out_path)
            print(f"    저장: {len(memory)} 궤적 (에피소드 {n_episodes})\n")

    return memory, n_episodes


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="공식 collect_data_real.py 의 ROS2 이식", allow_abbrev=False)
    ap.add_argument("--weights", required=True, help="DT 체크포인트")
    ap.add_argument("--out", default=None, help="출력 pkl")
    ap.add_argument("--targets", default=None,
                    help="쉼표 구분 거리 [m] (기본: 공식 0.5~2.0 / 0.05)")
    ap.add_argument("--amp", default="1", help="쉼표 구분 탐색 게인 (원본 amp_range)")
    ap.add_argument("--k-her", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vel-scale", type=float, default=0.2,
                    help="접근/복귀 이동 속도 배율 (던지기 자체와 무관)")
    ap.add_argument("--no-resume", action="store_true")
    return ap


def main():
    args = build_arg_parser().parse_args()
    targets = ([float(s) for s in args.targets.split(",")]
               if args.targets else None)
    amps = [float(s) for s in args.amp.split(",")]
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data_gp8",
        f"memory_real_traj-{len(targets or TARGET_LIST)}_herK-{args.k_her}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from gp8_real_arm import GP8RealArm, load_dt_model     # rclpy 구현 (실기 전용)

    model = load_dt_model(args.weights)
    arm = GP8RealArm(vel_scale=args.vel_scale)
    try:
        memory, n = collect(arm, model, out, targets=targets, amp_range=amps,
                            k_her=args.k_her,
                            rng=np.random.default_rng(args.seed),
                            resume=not args.no_resume)
        print(f"\n완료: {n} 에피소드 → {len(memory)} 궤적  ({out})")
    except KeyboardInterrupt:
        print("\n중단 — 지금까지 수집분은 저장되어 있다.")
    finally:
        arm.shutdown()


if __name__ == "__main__":
    sys.exit(main())
