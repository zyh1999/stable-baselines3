from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo._score_adam import ScoreAdam
from stable_baselines3.ppo.ppo import PPO


class PPOBackpack(PPO):
    """
    使用 BackPACK 的 BatchGrad 在一次 backward 中并行计算 per-sample 梯度，
    为 `ScoreAdam` 构造二阶矩统计（E[g^2] 或 (E[g])^2）而不依赖 torch.func.vmap。

    其它超参数（如 `use_score_fisher`, `use_adam_ablation`, `separate_optimizers` 等）
    与基类 `PPO` 保持一致，只是实现 per-sample 统计的方式不同。
    """

    def _setup_model(self) -> None:
        """
        先调用基类构建策略和优化器，然后用 BackPACK 的 `extend` 包装策略网络，
        使其在 backward 时产生 `param.grad_batch`（per-sample 梯度）。
        """
        super()._setup_model()
        try:
            from backpack import extend  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - 仅在未安装 backpack 时触发
            raise ImportError(
                "PPOBackpack requires `backpack-for-pytorch` to compute per-sample gradients. "
                "Install it with `pip install backpack-for-pytorch`."
            ) from exc

        # 就地扩展 policy，使其在 backward 中支持 .grad_batch
        self.policy = extend(self.policy)

    def train(self) -> None:
        """
        基于 RolloutBuffer 更新策略。
        与基类 PPO 的 train 逻辑基本一致，只是在使用 ScoreAdam + separate_optimizers 时，
        通过 BackPACK 一次性得到 per-sample 梯度并构造二阶矩，而不是用 Python for 循环。
        """
        # 与 PPO.train 相同的上半部分 --------------------------------------------
        self.policy.set_training_mode(True)
        if self.separate_optimizers:
            assert self.actor_optimizer is not None and self.critic_optimizer is not None
            self._update_learning_rate([self.actor_optimizer, self.critic_optimizer])
        else:
            self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf: Optional[float] = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        # 额外日志：advantage 与 clip 的关系
        clip_fractions_adv_02 = []
        clip_fractions_adv_04 = []
        clip_fractions_adv_06 = []
        clip_fractions_adv_08 = []
        clip_fractions_adv_10 = []

        frac_adv_02 = []
        frac_adv_04 = []
        frac_adv_06 = []
        frac_adv_08 = []
        frac_adv_10 = []

        batch_advantages = []

        clip_frac_p0_20 = []
        clip_frac_p20_40 = []
        clip_frac_p40_60 = []
        clip_frac_p60_80 = []
        clip_frac_p80_100 = []

        batch_norm_advantages = []
        ratios = []
        grad_norms = []
        actor_grad_norms: list[np.ndarray] = []
        critic_grad_norms: list[np.ndarray] = []
        continue_training = True

        last_combined_loss: Optional[th.Tensor] = None
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                batch_advantages.append(advantages.cpu().numpy())

                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    # Apply mean and/or std normalization based on switches
                    if self.normalize_advantage_mean or self.normalize_advantage_std:
                        if self.normalize_advantage_mean:
                            advantages = advantages - advantages.mean()
                        if self.normalize_advantage_std:
                            advantages = advantages / (advantages.std() + 1e-8)
                        batch_norm_advantages.append(advantages.cpu().numpy())

                advantages = advantages * self.advantage_multiplier
                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                ratios.append(ratio.detach().cpu().numpy())
                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # For score-only loss used by ScoreAdam, we need to know which samples are
                # clipped according to the *original* PPO objective (that depends on advantage
                # sign and magnitude). We record this mask once using the true losses.
                with th.no_grad():
                    # True PPO uses the min between the unclipped and clipped objectives
                    # (for each sample). Wherever policy_loss_2 < policy_loss_1, the clipped
                    # objective is active.
                    use_clipped_mask = policy_loss_2 < policy_loss_1

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # 额外日志：按 advantage 大小/分位的 clip 情况 -------------------
                with th.no_grad():
                    adv_abs = th.abs(advantages)
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
                    adv_abs = th.abs(advantages)
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

                if clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                combined_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # 使用 BackPACK 计算 per-sample 梯度统计，供 ScoreAdam 使用 ------------
                score_grads_dict: Optional[dict[int, th.Tensor]] = None
                actor_mean_grads_dict: Optional[dict[int, th.Tensor]] = None
                if self.separate_optimizers and isinstance(self.actor_optimizer, ScoreAdam):
                    score_grads_dict, actor_mean_grads_dict = self._compute_actor_backpack_stats(
                        log_prob=log_prob,
                        entropy=entropy,
                        ratio=ratio,
                        policy_loss_1=policy_loss_1,
                        policy_loss_2=policy_loss_2,
                        use_clipped_mask=use_clipped_mask,
                        clip_range=clip_range,
                    )

                # Optimization step ----------------------------------------------------
                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None

                    # Zero-grad both optimizers
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    # Backward actor then critic
                    actor_loss = policy_loss + self.ent_coef * entropy_loss
                    critic_loss = self.vf_coef * value_loss

                    if isinstance(self.actor_optimizer, ScoreAdam) and self.use_adam_ablation:
                        # Adam 消融版本：
                        # 1) 先对 critic 做 backward，拿到 critic 及共享特征提取器的梯度；
                        critic_loss.backward()
                        # 2) 再把基于 per-sample 的 actor 平均梯度 E[g] 加到 actor 参数（含共享特征）上；
                        assert actor_mean_grads_dict is not None
                        for p in self._actor_params:
                            pid = id(p)
                            g_actor = actor_mean_grads_dict.get(pid)
                            if g_actor is None:
                                continue
                            if p.grad is None:
                                p.grad = g_actor.detach().clone()
                            else:
                                p.grad = p.grad + g_actor.detach()
                    else:
                        # 原始行为：actor + critic 一起 backward，分子来自批量 loss
                        actor_loss.backward(retain_graph=True)
                        critic_loss.backward()

                    # Clip and step actor (includes shared params)
                    actor_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm).cpu().numpy()
                    )
                    if isinstance(self.actor_optimizer, ScoreAdam):
                        # Pass score-only gradients so that the denominator in Adam
                        # only depends on score-related terms, not on the advantage.
                        self.actor_optimizer.step(score_grads=score_grads_dict)
                    else:
                        self.actor_optimizer.step()

                    # Clip and step critic (critic-only params)
                    critic_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm).cpu().numpy()
                    )
                    self.critic_optimizer.step()
                    # For compatibility with later logging
                    last_combined_loss = combined_loss
                else:
                    self.policy.optimizer.zero_grad()
                    combined_loss.backward()
                    # Clip grad norm
                    grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).cpu().numpy()
                    )
                    self.policy.optimizer.step()
                    last_combined_loss = combined_loss

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs ---------------------------------------------------------------------
        self.logger.record("train/ratios_mean", np.mean(ratios))
        if len(batch_advantages) > 0:
            self.logger.record("train/advantages_mean", np.mean(batch_advantages))
            self.logger.record("train/advantages_sum", np.sum(batch_advantages))
        if len(batch_norm_advantages) > 0:
            self.logger.record("train/advantages_norm_mean", np.mean(batch_norm_advantages))
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

        if last_combined_loss is not None:
            self.logger.record("train/loss", last_combined_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        # Grad norm logging
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

    # ------------------------------------------------------------------ helpers
    def _compute_actor_backpack_stats(
        self,
        log_prob: th.Tensor,
        entropy: Optional[th.Tensor],
        ratio: th.Tensor,
        policy_loss_1: th.Tensor,
        policy_loss_2: th.Tensor,
        use_clipped_mask: th.Tensor,
        clip_range: float,
    ) -> tuple[dict[int, th.Tensor], Optional[dict[int, th.Tensor]]]:
        """
        使用 BackPACK 的 BatchGrad 一次 backward 得到所有 actor 参数的 per-sample 梯度。

        - 如果 use_score_fisher=True 且 use_adam_ablation=False：构造 score-only 的 per-sample loss，
          再根据 per-sample 梯度 g 计算 E[g^2]。
        - 如果 use_score_fisher=False 且 use_adam_ablation=False：用真实 actor loss 的 per-sample 梯度，
          计算 E[g^2] 作为 True-Fisher。
        - 如果 use_adam_ablation=True：用真实 actor loss 的 per-sample 梯度，先算 E[g] 再做裁剪，
          再得到 (E[g])^2 作为二阶矩估计。
        """
        try:
            from backpack import backpack  # type: ignore[import]
            from backpack.extensions import BatchGrad  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PPOBackpack requires `backpack-for-pytorch` to compute per-sample gradients. "
                "Install it with `pip install backpack-for-pytorch`."
            ) from exc

        assert self._actor_params is not None

        # 1) 构造 per-sample loss
        if self.use_score_fisher and not self.use_adam_ablation:
            # Score-only：保留 ratio/clip 结构，不含 advantage/entropy
            score_unclipped = ratio
            score_clipped = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            per_sample_loss = -th.where(use_clipped_mask, score_clipped, score_unclipped)
        else:
            # 真实 actor loss（policy + entropy），与主 actor_loss 的结构一致
            policy_loss_per_sample = -th.min(policy_loss_1, policy_loss_2)
            if entropy is None:
                entropy_loss_per_sample = -(-log_prob)  # = log_prob
            else:
                entropy_loss_per_sample = -entropy
            per_sample_loss = policy_loss_per_sample + self.ent_coef * entropy_loss_per_sample  # [batch]

        # 2) 用 BackPACK 的 BatchGrad 计算 per-sample 梯度
        # 先清空已有的 .grad / .grad_batch，避免累积旧值
        for p in self._actor_params:
            p.grad = None
            if hasattr(p, "grad_batch"):
                delattr(p, "grad_batch")

        loss_sum = per_sample_loss.sum()
        with backpack(BatchGrad()):
            # 需要 retain_graph=True，因为后面主分支还要对 actor_loss/critic_loss 再 backward 一次
            loss_sum.backward(retain_graph=True)

        # 3) 基于 per-sample 梯度构造二阶矩或均值梯度
        second_moment_dict: dict[int, th.Tensor] = {}
        actor_mean_grad_dict: Optional[dict[int, th.Tensor]] = {} if self.use_adam_ablation else None

        for p in self._actor_params:
            if not hasattr(p, "grad_batch"):
                continue
            grad_batch: th.Tensor = p.grad_batch  # [batch, *p.shape]
            pid = id(p)
            if self.use_adam_ablation:
                assert actor_mean_grad_dict is not None
                # 先取 batch 平均梯度 E[g]
                mean_grad = grad_batch.mean(dim=0).to(p.dtype)
                actor_mean_grad_dict[pid] = mean_grad
            else:
                # Fisher / True-Fisher：E[g^2]
                second_moment = grad_batch.pow(2).mean(dim=0)
                second_moment_dict[pid] = second_moment.to(p.dtype)

        if self.use_adam_ablation:
            assert actor_mean_grad_dict is not None
            # 对平均梯度做一次全局 norm 裁剪（模拟标准 Adam/PPO 的行为）
            self._clip_actor_mean_grads(actor_mean_grad_dict)
            # 然后用 (E[g])^2 作为二阶矩估计
            for pid, mean_grad in actor_mean_grad_dict.items():
                second_moment_dict[pid] = (mean_grad * mean_grad).to(mean_grad.dtype)

        return second_moment_dict, actor_mean_grad_dict

    def _clip_actor_mean_grads(self, actor_mean_grads_dict: dict[int, th.Tensor]) -> None:
        """
        对所有 actor 参数的平均梯度做一次全局 L2 norm 裁剪。
        """
        if self.max_grad_norm <= 0:
            return
        grads = list(actor_mean_grads_dict.values())
        if len(grads) == 0:
            return
        device = grads[0].device
        dtype = grads[0].dtype
        max_norm_tensor = th.tensor(self.max_grad_norm, device=device, dtype=dtype)
        if not th.isfinite(max_norm_tensor):
            return

        total_norm_sq = th.zeros((), device=device, dtype=dtype)
        for grad in grads:
            total_norm_sq = total_norm_sq + grad.pow(2).sum()
        total_norm = th.sqrt(total_norm_sq + 1e-16)

        clip_coef = min(1.0, self.max_grad_norm / (total_norm.item() + 1e-6))
        if clip_coef >= 1.0:
            return

        clip_coef_tensor = th.tensor(clip_coef, device=device, dtype=dtype)
        for pid, grad in actor_mean_grads_dict.items():
            actor_mean_grads_dict[pid] = grad * clip_coef_tensor


