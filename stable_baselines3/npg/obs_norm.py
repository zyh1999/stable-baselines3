"""
Detach-style observation normalization helper (mingfei-style obs_rms).

Goal:
- Keep the main algorithm (`npg.py`) and rollout (`rollout.py`) clean.
- Provide a minimal interface:
  - update_from_rollout_buffer(rollout_buffer)
  - normalize_np(obs_np)
  - normalize_tensor(obs_t)

Notes:
- This is NOT SB3 VecNormalize. It only normalizes observations using RunningMeanStd.
- Intended for Box observation spaces (vector observations). Dict/image spaces are not supported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.running_mean_std import RunningMeanStd


@dataclass
class DetachObsNormConfig:
    eps: float = 1e-8
    clip: Optional[float] = None
    recompute_pi_old: bool = True


class DetachObsRMS:
    """
    Maintain RunningMeanStd for observations and provide normalization for numpy/torch.

    Behavior:
    - During rollout: call normalize_*() (no stats update).
    - Before train() update: call update_from_rollout_buffer() to update stats on the current rollout batch.
    """

    def __init__(self, observation_space: spaces.Space, device: th.device | str, cfg: DetachObsNormConfig):
        if not isinstance(observation_space, spaces.Box):
            raise NotImplementedError("DetachObsRMS supports only Box observation spaces.")
        self.obs_shape = observation_space.shape
        self.cfg = cfg
        self.rms = RunningMeanStd(shape=self.obs_shape)
        self.device = th.device(device)
        self._mean_t: th.Tensor = th.as_tensor(self.rms.mean, device=self.device, dtype=th.float32)
        self._var_t: th.Tensor = th.as_tensor(self.rms.var, device=self.device, dtype=th.float32)

    def _refresh_tensors(self) -> None:
        self._mean_t = th.as_tensor(self.rms.mean, device=self.device, dtype=th.float32)
        self._var_t = th.as_tensor(self.rms.var, device=self.device, dtype=th.float32)

    def update_from_rollout_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        obs = rollout_buffer.observations
        if not isinstance(obs, np.ndarray):
            raise NotImplementedError("DetachObsRMS supports only ndarray observations in RolloutBuffer.")
        # obs shape: [n_steps, n_envs, *obs_shape]
        obs_flat = obs.swapaxes(0, 1).reshape(-1, *obs.shape[2:])
        self.rms.update(obs_flat)
        self._refresh_tensors()

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        obs_f = obs.astype(np.float32, copy=False)
        mean = self.rms.mean.astype(np.float32, copy=False)
        var = self.rms.var.astype(np.float32, copy=False)
        out = (obs_f - mean) / np.sqrt(var + float(self.cfg.eps))
        if self.cfg.clip is not None:
            out = np.clip(out, -float(self.cfg.clip), float(self.cfg.clip))
        return out

    def normalize_tensor(self, obs: th.Tensor) -> th.Tensor:
        out = (obs - self._mean_t) / th.sqrt(self._var_t + float(self.cfg.eps))
        if self.cfg.clip is not None:
            out = th.clamp(out, -float(self.cfg.clip), float(self.cfg.clip))
        return out


