import warnings
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo._clip_fraction_traces import ClipFractionTraceLogger
from stable_baselines3.ppo.ppo import PPO


class PPOAdvDecouple(PPO):
    """
    PPO 变体：将 advantage 解耦为“loss 用的 adv”和“clip mask 判定用的 adv”，
    并提供 mean/std 开关。其余逻辑与原始 PPO 保持一致，保留 separate_optimizers /
    disable_joint_critic_update / recompute_advantage_with_current_vf / clip_fraction_trace。

    新增开关（仅在 normalize_advantage=True 时生效）：
    - loss_use_adv_mean: loss 用的 adv 是否减均值，默认沿用 normalize_advantage_mean
    - loss_use_adv_std:  loss 用的 adv 是否做 std 归一化，默认沿用 normalize_advantage_std
    - clip_mask_use_adv_mean: clip mask 判定是否用减均值的 adv，默认跟随 loss_use_adv_mean
    """

    def __init__(
        self,
        policy: Union[str, type],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        advantage_multiplier: float = 1.0,
        normalize_advantage_mean: bool = True,
        normalize_advantage_std: bool = True,
        separate_optimizers: bool = True,
        # 新增解耦开关
        clip_mask_use_adv_mean: Optional[bool] = None,
        loss_use_adv_mean: Optional[bool] = None,
        loss_use_adv_std: Optional[bool] = None,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            advantage_multiplier=advantage_multiplier,
            normalize_advantage_mean=normalize_advantage_mean,
            normalize_advantage_std=normalize_advantage_std,
            separate_optimizers=separate_optimizers,
            use_score_fisher=True,
            use_adam_ablation=False,
            clip_fraction_trace_dir=None,
            clip_fraction_trace_plot_every_steps=100_000,
            _init_setup_model=False,
            **kwargs,
        )

        # 解耦开关，默认沿用原有 normalize_advantage_mean/std
        self.loss_use_adv_mean = self.normalize_advantage_mean if loss_use_adv_mean is None else loss_use_adv_mean
        self.loss_use_adv_std = self.normalize_advantage_std if loss_use_adv_std is None else loss_use_adv_std
        self.clip_mask_use_adv_mean = self.loss_use_adv_mean if clip_mask_use_adv_mean is None else clip_mask_use_adv_mean

        if _init_setup_model:
            self._setup_model()

    def train(self) -> None:
        self.policy.set_training_mode(True)

        if self.separate_optimizers:
            assert self.actor_optimizer is not None and self.critic_optimizer is not None
            self._update_learning_rate([self.actor_optimizer, self.critic_optimizer])
        else:
            self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        clip_fractions_adv_02, clip_fractions_adv_04, clip_fractions_adv_06, clip_fractions_adv_08, clip_fractions_adv_10 = (
            [],
            [],
            [],
            [],
            [],
        )
        frac_adv_02, frac_adv_04, frac_adv_06, frac_adv_08, frac_adv_10 = [], [], [], [], []
        clip_frac_p0_20, clip_frac_p20_40, clip_frac_p40_60, clip_frac_p60_80, clip_frac_p80_100 = [], [], [], [], []

        batch_advantages = []
        batch_norm_advantages = []
        ratios = []
        grad_norms = []
        actor_grad_norms: list[np.ndarray] = []
        critic_grad_norms: list[np.ndarray] = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages_raw = rollout_data.advantages
                advantages_for_loss = advantages_raw
                advantages_for_mask = advantages_raw
                advantages_meaned = advantages_raw

                if self.normalize_advantage and len(advantages_raw) > 1:
                    raw_mean = advantages_raw.mean()
                    raw_std = advantages_raw.std()
                    if self.loss_use_adv_mean:
                        advantages_for_loss = advantages_for_loss - raw_mean
                    if self.loss_use_adv_std:
                        advantages_for_loss = advantages_for_loss / (raw_std + 1e-8)
                    if self.clip_mask_use_adv_mean:
                        advantages_for_mask = advantages_for_mask - raw_mean
                    if self.loss_use_adv_mean or self.clip_mask_use_adv_mean:
                        advantages_meaned = advantages_raw - raw_mean

                advantages = rollout_data.advantages
                batch_advantages.append(advantages.cpu().numpy())
                if self.normalize_advantage and len(advantages) > 1:
                    if self.loss_use_adv_mean or self.loss_use_adv_std:
                        batch_norm_advantages.append(advantages_for_loss.cpu().numpy())

                advantages_for_loss = advantages_for_loss * self.advantage_multiplier
                advantages_for_mask = advantages_for_mask * self.advantage_multiplier

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                ratios.append(ratio.detach().cpu().numpy())
                ratio_clipped = th.clamp(ratio, 1 - clip_range, 1 + clip_range)

                with th.no_grad():
                    mask_loss_1 = advantages_for_mask * ratio
                    mask_loss_2 = advantages_for_mask * ratio_clipped
                    use_clipped_mask = mask_loss_2 < mask_loss_1

                policy_loss_1 = advantages_for_loss * ratio
                policy_loss_2 = advantages_for_loss * ratio_clipped
                policy_objective = th.where(use_clipped_mask, policy_loss_2, policy_loss_1)
                policy_loss = -policy_objective.mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                with th.no_grad():
                    adv_abs = th.abs(advantages_for_loss)
                    clipped = (th.abs(ratio - 1) > clip_range).float()
                    for threshold, storage, frac_storage in [
                        (0.2, clip_fractions_adv_02, frac_adv_02),
                        (0.4, clip_fractions_adv_04, frac_adv_04),
                        (0.6, clip_fractions_adv_06, frac_adv_06),
                        (0.8, clip_fractions_adv_08, frac_adv_08),
                        (1.0, clip_fractions_adv_10, frac_adv_10),
                    ]:
                        mask = adv_abs < threshold
                        frac_storage.append(mask.float().mean().item())
                        if mask.any():
                            storage.append(((clipped * mask).sum() / mask.sum()).item())

                with th.no_grad():
                    adv_abs = th.abs(advantages_for_loss)
                    sorted_indices = th.argsort(adv_abs)
                    batch_size = len(adv_abs)
                    idx_20 = int(batch_size * 0.2)
                    idx_40 = int(batch_size * 0.4)
                    idx_60 = int(batch_size * 0.6)
                    idx_80 = int(batch_size * 0.8)
                    indices_groups = [
                        (sorted_indices[:idx_20], clip_frac_p0_20),
                        (sorted_indices[idx_20:idx_40], clip_frac_p20_40),
                        (sorted_indices[idx_40:idx_60], clip_frac_p40_60),
                        (sorted_indices[idx_60:idx_80], clip_frac_p60_80),
                        (sorted_indices[idx_80:], clip_frac_p80_100),
                    ]
                    clipped_batch = (th.abs(ratio - 1) > clip_range).float()
                    for indices, storage in indices_groups:
                        if len(indices) > 0:
                            fraction = clipped_batch[indices].mean().item()
                            storage.append(fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                combined_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    actor_loss = policy_loss + self.ent_coef * entropy_loss
                    critic_loss = self.vf_coef * value_loss

                    actor_loss.backward(retain_graph=True)
                    critic_loss.backward()

                    actor_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm).cpu().numpy()
                    )
                    self.actor_optimizer.step()

                    critic_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm).cpu().numpy()
                    )
                    self.critic_optimizer.step()
                    loss = combined_loss
                else:
                    self.policy.optimizer.zero_grad()
                    combined_loss.backward()
                    grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).cpu().numpy()
                    )
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/ratios_mean", np.mean(ratios))
        if len(batch_advantages) > 0:
            self.logger.record("train/advantages_mean", np.mean(batch_advantages))
            self.logger.record("train/advantages_std", np.std(batch_advantages))
            self.logger.record("train/advantages_sum", np.sum(batch_advantages))
        if len(batch_norm_advantages) > 0:
            self.logger.record("train/advantages_norm_mean", np.mean(batch_norm_advantages))
            self.logger.record("train/advantages_norm_std", np.std(batch_norm_advantages))
            self.logger.record("train/advantages_norm_sum", np.sum(batch_norm_advantages))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))

        if len(clip_fractions_adv_02) > 0:
            self.logger.record("train/clip_fraction_adv_lt_0.2", np.mean(clip_fractions_adv_02))
            self.logger.record("train/adv_fraction_lt_0.2", np.mean(frac_adv_02))
        if len(clip_fractions_adv_04) > 0:
            self.logger.record("train/clip_fraction_adv_lt_0.4", np.mean(clip_fractions_adv_04))
            self.logger.record("train/adv_fraction_lt_0.4", np.mean(frac_adv_04))
        if len(clip_fractions_adv_06) > 0:
            self.logger.record("train/clip_fraction_adv_lt_0.6", np.mean(clip_fractions_adv_06))
            self.logger.record("train/adv_fraction_lt_0.6", np.mean(frac_adv_06))
        if len(clip_fractions_adv_08) > 0:
            self.logger.record("train/clip_fraction_adv_lt_0.8", np.mean(clip_fractions_adv_08))
            self.logger.record("train/adv_fraction_lt_0.8", np.mean(frac_adv_08))
        if len(clip_fractions_adv_10) > 0:
            self.logger.record("train/clip_fraction_adv_lt_1.0", np.mean(clip_fractions_adv_10))
            self.logger.record("train/adv_fraction_lt_1.0", np.mean(frac_adv_10))

        if len(clip_frac_p0_20) > 0:
            self.logger.record("train/clip_frac_p0_20", np.mean(clip_frac_p0_20))
        if len(clip_frac_p20_40) > 0:
            self.logger.record("train/clip_frac_p20_40", np.mean(clip_frac_p20_40))
        if len(clip_frac_p40_60) > 0:
            self.logger.record("train/clip_frac_p40_60", np.mean(clip_frac_p40_60))
        if len(clip_frac_p60_80) > 0:
            self.logger.record("train/clip_frac_p60_80", np.mean(clip_frac_p60_80))
        if len(clip_frac_p80_100) > 0:
            self.logger.record("train/clip_frac_p80_100", np.mean(clip_frac_p80_100))

        if len(batch_advantages) > 0:
             self.logger.record("train/advantages_abs_mean", np.mean(np.abs(batch_advantages)))

        self.logger.record("train/loss", combined_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if len(actor_grad_norms) > 0:
            self.logger.record("train/grad_norm_actor", np.mean(actor_grad_norms))
            self.logger.record("train/grad_norm_actor/max", np.max(actor_grad_norms))
        if len(critic_grad_norms) > 0:
            self.logger.record("train/grad_norm_critic", np.mean(critic_grad_norms))
            self.logger.record("train/grad_norm_critic/max", np.max(critic_grad_norms))
        if len(grad_norms) > 0:
            self.logger.record("train/grad_norm", np.mean(grad_norms))
            self.logger.record("train/grad_norm/max", np.max(grad_norms))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


