"""
Rollout collection helpers for NPG.

We keep the rollout logic in a separate module to avoid bloating the main algorithm file.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


def collect_rollouts_npg(algo: Any, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``.

    This mirrors SB3's default on-policy rollout collection, with an optional detach-style action squashing
    for continuous actions: ``tanh(actions) * action_high`` (instead of ``np.clip``).
    Enabled via ``algo.action_squash=True``.
    """
    assert algo._last_obs is not None, "No previous observation was provided"
    algo.policy.set_training_mode(False)

    n_steps = 0
    rollout_buffer.reset()
    if algo.use_sde:
        algo.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
        if algo.use_sde and algo.sde_sample_freq > 0 and n_steps % algo.sde_sample_freq == 0:
            algo.policy.reset_noise(env.num_envs)

        with th.no_grad():
            obs_np = algo._last_obs  # type: ignore[assignment]
            obs_norm = getattr(algo, "_obs_norm", None)
            if obs_norm is not None:
                obs_np = obs_norm.normalize_np(obs_np)
            obs_tensor = obs_as_tensor(obs_np, algo.device)  # type: ignore[arg-type]
            actions, values, log_probs = algo.policy(obs_tensor)
        actions = actions.cpu().numpy()

        clipped_actions = actions
        if isinstance(algo.action_space, spaces.Box):
            if getattr(algo, "action_squash", False):
                high = algo.action_space.high
                low = algo.action_space.low
                # Strictly correct for symmetric bounds; fallback to np.clip otherwise.
                if np.allclose(low, -high):
                    clipped_actions = np.tanh(actions) * high
                else:
                    clipped_actions = np.clip(actions, low, high)
            else:
                clipped_actions = np.clip(actions, algo.action_space.low, algo.action_space.high)

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        algo.num_timesteps += env.num_envs
        callback.update_locals(locals())
        if not callback.on_step():
            return False

        algo._update_info_buffer(infos, dones)
        n_steps += 1

        if isinstance(algo.action_space, spaces.Discrete):
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstrapping with value function (SB3 behavior)
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs_np = infos[idx]["terminal_observation"]
                obs_norm = getattr(algo, "_obs_norm", None)
                if obs_norm is not None:
                    terminal_obs_np = obs_norm.normalize_np(terminal_obs_np)
                terminal_obs = algo.policy.obs_to_tensor(terminal_obs_np)[0]
                with th.no_grad():
                    terminal_value = algo.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                # detach 对齐：TimeLimit.truncated 的 bootstrap 系数用的是 gae_lambda（Runner 里是 self.lam），不是 gamma
                # 这会影响 returns/advantages 的尺度，从而影响 PopArt 的 running stats 和 actor/critic 的有效步长。
                rewards[idx] += algo.gae_lambda * terminal_value

        rollout_buffer.add(
            algo._last_obs,  # type: ignore[arg-type]
            actions,
            rewards,
            algo._last_episode_starts,  # type: ignore[arg-type]
            values,
            log_probs,
        )
        algo._last_obs = new_obs  # type: ignore[assignment]
        algo._last_episode_starts = dones

    with th.no_grad():
        last_obs_np = new_obs
        obs_norm = getattr(algo, "_obs_norm", None)
        if obs_norm is not None:
            last_obs_np = obs_norm.normalize_np(last_obs_np)
        values = algo.policy.predict_values(obs_as_tensor(last_obs_np, algo.device))  # type: ignore[arg-type]
    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.update_locals(locals())
    callback.on_rollout_end()
    return True


