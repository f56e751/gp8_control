"""Move late rewards back to the step that caused them before PPO's advantages."""

from __future__ import annotations

import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor


class CauseCreditCallback(BaseCallback):
    """Uses info["credit_moves"] from GP8RecyclingEnv. With gamma=1 moving a reward
    within an episode keeps the return; causes before this rollout stay where they are."""

    def _on_rollout_start(self) -> None:
        self._moves = []

    def _on_step(self) -> bool:
        pos = self.model.rollout_buffer.pos  # index this step is about to be stored at
        for env_idx, info in enumerate(self.locals["infos"]):
            for lag, reward in info.get("credit_moves", ()):
                self._moves.append((pos, env_idx, int(lag), float(reward)))
        return True

    def _on_rollout_end(self) -> None:
        buf, moved = self.model.rollout_buffer, 0
        for pos, env_idx, lag, reward in self._moves:
            if pos - lag >= 0:
                buf.rewards[pos - lag, env_idx] += reward
                buf.rewards[pos, env_idx] -= reward
                moved += 1
        if moved:
            with th.no_grad():
                last_values = self.model.policy.predict_values(
                    obs_as_tensor(self.model._last_obs, self.model.device)
                )
            buf.compute_returns_and_advantage(
                last_values=last_values, dones=self.model._last_episode_starts
            )
        self.logger.record("credit/moved", moved)
        self.logger.record("credit/kept", len(self._moves) - moved)
