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
from torch import nn
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer


@dataclass
class DetachObsNormConfig:
    # Match mingfei RunningMeanStd defaults:
    # - epsilon=1e-05
    # - clamp normalized obs to [-5, 5] by default
    eps: float = 1e-5
    clip: Optional[float] = 5.0
    recompute_pi_old: bool = True


class MingfeiRunningMeanStd(nn.Module):
    """
    Exact copy of mingfei_npg/Archive/utils/running_mean_std.py behavior (per_channel=False path).
    - running_mean/running_var/count are float64 buffers
    - update uses batch mean/var over axis=0
    - normalize does (x-mean)/sqrt(var+eps) then clamp to [-5,5] (or cfg clip)
    """

    def __init__(self, insize: tuple[int, ...], epsilon: float = 1e-5, device: th.device | str = "cpu"):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.axis = [0]

        dev = th.device(device)
        self.register_buffer("running_mean", th.zeros(insize, dtype=th.float64, device=dev))
        self.register_buffer("running_var", th.ones(insize, dtype=th.float64, device=dev))
        # mingfei initializes count as ones(()) not zeros/eps
        self.register_buffer("count", th.ones((), dtype=th.float64, device=dev))

    @staticmethod
    def _update_mean_var_count_from_moments(
        mean: th.Tensor,
        var: th.Tensor,
        count: th.Tensor,
        batch_mean: th.Tensor,
        batch_var: th.Tensor,
        batch_count: int,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    @th.no_grad()
    def update(self, x: th.Tensor) -> None:
        # Match mingfei: mean/var computed over axis=0; torch.var default unbiased=True
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
            self.running_mean, self.running_var, self.count, batch_mean.double(), batch_var.double(), int(x.size(0))
        )
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.count.copy_(new_count)

    def normalize(self, x: th.Tensor, clip: Optional[float]) -> th.Tensor:
        # per_channel=False path in mingfei: current_mean/current_var are 1D tensors
        y = (x - self.running_mean.float()) / th.sqrt(self.running_var.float() + float(self.epsilon))
        if clip is not None:
            y = th.clamp(y, min=-float(clip), max=float(clip))
        return y


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
        self.rms = MingfeiRunningMeanStd(self.obs_shape, epsilon=float(cfg.eps), device=device)
        self.device = th.device(device)

    def update_from_rollout_buffer(self, rollout_buffer: RolloutBuffer) -> None:
        obs = rollout_buffer.observations
        if not isinstance(obs, np.ndarray):
            raise NotImplementedError("DetachObsRMS supports only ndarray observations in RolloutBuffer.")
        # obs shape: [n_steps, n_envs, *obs_shape]
        obs_flat = obs.swapaxes(0, 1).reshape(-1, *obs.shape[2:])
        obs_t = th.as_tensor(obs_flat, device=self.device, dtype=th.float32)
        self.rms.train(True)
        self.rms.update(obs_t)
        self.rms.train(False)

    def normalize_np(self, obs: np.ndarray) -> np.ndarray:
        obs_t = th.as_tensor(obs, device=self.device, dtype=th.float32)
        out_t = self.normalize_tensor(obs_t)
        return out_t.detach().cpu().numpy()

    def normalize_tensor(self, obs: th.Tensor) -> th.Tensor:
        return self.rms.normalize(obs, clip=self.cfg.clip)


