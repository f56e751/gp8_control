"""RL-facing environment helpers for gp8_control."""

from gp8_control.rl.common import HighLevelAction

__all__ = [
    "GP8RecyclingEnv",
    "HighLevelAction",
    "RealRlRunner",
    "RealShadowRl",
    "SimRlRunner",
]


def __getattr__(name: str):
    if name == "GP8RecyclingEnv":
        from gp8_control.rl.gym_env import GP8RecyclingEnv

        return GP8RecyclingEnv
    if name == "RealShadowRl":
        from gp8_control.rl.real_shadow import RealShadowRl

        return RealShadowRl
    if name == "RealRlRunner":
        from gp8_control.rl.real_runner import RealRlRunner

        return RealRlRunner
    if name == "SimRlRunner":
        from gp8_control.rl.sim_runner import SimRlRunner

        return SimRlRunner
    raise AttributeError(name)
