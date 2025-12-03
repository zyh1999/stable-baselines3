from __future__ import annotations

from typing import Any, Optional
from collections import OrderedDict

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

try:
    from torch.func import vmap, grad as func_grad, functional_call
except ImportError:
    try:
        from functorch import vmap, grad as func_grad, functional_call
    except ImportError as exc:
        raise ImportError(
            "torch.func.functional_call is not available in this PyTorch version. "
            "Please install torch>=2.1 or functorch."
        ) from exc

from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo._score_adam import ScoreAdam
from stable_baselines3.ppo.ppo import PPO


class _EvalActionsModule(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        return self.policy.evaluate_actions(obs, actions)


class PPOVmap(PPO):
    """PPO variant that uses torch.func.vmap for per-sample ScoreAdam statistics."""

    def train(self) -> None:
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
        batch_advantages = []
        batch_norm_advantages = []
        ratios = []
        grad_norms = []
        actor_grad_norms: list[np.ndarray] = []
        critic_grad_norms: list[np.ndarray] = []
        continue_training = True

        last_combined_loss: Optional[th.Tensor] = None
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                batch_info = self._compute_batch_losses(rollout_data, clip_range, clip_range_vf)

                pg_losses.append(batch_info["policy_loss"].item())
                clip_fractions.append(float(batch_info["clip_fraction"]))
                value_losses.append(batch_info["value_loss"].item())
                entropy_losses.append(batch_info["entropy_loss"].item())
                batch_advantages.append(batch_info["raw_advantages"].cpu().numpy())
                if batch_info["normalized_advantages"] is not None:
                    batch_norm_advantages.append(batch_info["normalized_advantages"].cpu().numpy())
                ratios.append(batch_info["ratio"].detach().cpu().numpy())

                with th.no_grad():
                    log_ratio_tensor = batch_info["log_ratio"]
                    approx_kl_div = th.mean((th.exp(log_ratio_tensor) - 1) - log_ratio_tensor).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                score_grads_dict, actor_mean_grads_dict = self._maybe_compute_score_grads(
                    rollout_data, batch_info, clip_range
                )

                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None

                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    actor_loss = batch_info["policy_loss"] + self.ent_coef * batch_info["entropy_loss"]
                    critic_loss = self.vf_coef * batch_info["value_loss"]

                    if isinstance(self.actor_optimizer, ScoreAdam) and self.use_adam_ablation:
                        assert actor_mean_grads_dict is not None
                        critic_loss.backward()
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
                        actor_loss.backward(retain_graph=True)
                        critic_loss.backward()

                    actor_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm).cpu().numpy()
                    )
                    if isinstance(self.actor_optimizer, ScoreAdam):
                        self.actor_optimizer.step(score_grads=score_grads_dict)
                    else:
                        self.actor_optimizer.step()

                    critic_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm).cpu().numpy()
                    )
                    self.critic_optimizer.step()
                    last_combined_loss = batch_info["combined_loss"]
                else:
                    self.policy.optimizer.zero_grad()
                    batch_info["combined_loss"].backward()
                    grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm).cpu().numpy()
                    )
                    self.policy.optimizer.step()
                    last_combined_loss = batch_info["combined_loss"]

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

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
        if last_combined_loss is not None:
            self.logger.record("train/loss", last_combined_loss.item())
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

    def _compute_batch_losses(
        self,
        rollout_data,
        clip_range: float,
        clip_range_vf: Optional[float],
    ) -> dict[str, Any]:
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()

        raw_advantages = rollout_data.advantages
        normalized_advantages: Optional[th.Tensor] = None
        advantages = raw_advantages
        if self.normalize_advantage and len(advantages) > 1:
            if self.normalize_advantage_mean:
                advantages = advantages - advantages.mean()
            if self.normalize_advantage_std:
                advantages = advantages / (advantages.std() + 1e-8)
            normalized_advantages = advantages
        advantages = advantages * self.advantage_multiplier

        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()

        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
            )
        value_loss = F.mse_loss(rollout_data.returns, values_pred)

        if entropy is None:
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        combined_loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        return {
            "actions": actions,
            "values": values,
            "log_prob": log_prob,
            "entropy": entropy,
            "ratio": ratio,
            "log_ratio": log_prob - rollout_data.old_log_prob,
            "policy_loss_1": policy_loss_1,
            "policy_loss_2": policy_loss_2,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "combined_loss": combined_loss,
            "clip_fraction": clip_fraction,
            "raw_advantages": raw_advantages,
            "normalized_advantages": normalized_advantages,
            "advantages_used": advantages,
        }

    def _maybe_compute_score_grads(
        self,
        rollout_data,
        batch_info: dict[str, Any],
        clip_range: float,
    ) -> tuple[Optional[dict[int, th.Tensor]], Optional[dict[int, th.Tensor]]]:
        if not (self.separate_optimizers and isinstance(self.actor_optimizer, ScoreAdam)):
            return None, None
        assert self._actor_params is not None
        actions = batch_info["actions"]
        assert isinstance(actions, th.Tensor)
        advantages_used = batch_info["advantages_used"]
        assert isinstance(advantages_used, th.Tensor)
        score_grads_dict, actor_mean_grads_dict = self._compute_actor_vmap_stats(
            rollout_data.observations,
            actions,
            rollout_data.old_log_prob,
            advantages_used,
            clip_range,
        )
        return score_grads_dict, actor_mean_grads_dict

    def _compute_actor_vmap_stats(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        old_log_prob: th.Tensor,
        advantages: th.Tensor,
        clip_range: float,
    ) -> tuple[dict[int, th.Tensor], Optional[dict[int, th.Tensor]]]:
        assert self._actor_params is not None
        eval_module = _EvalActionsModule(self.policy)
        params = OrderedDict(eval_module.named_parameters())
        buffers = OrderedDict(eval_module.named_buffers())

        def per_sample_loss_fn(
            params_tuple,
            buffers_tuple,
            obs_sample: th.Tensor,
            action_sample: th.Tensor,
            old_lp: th.Tensor,
            adv: th.Tensor,
        ) -> th.Tensor:
            obs_batch = obs_sample.unsqueeze(0)
            action_batch = action_sample.unsqueeze(0)
            _, log_prob, entropy = functional_call(
                eval_module, (params_tuple, buffers_tuple), (obs_batch, action_batch)
            )
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0) if entropy is not None else None

            ratio = th.exp(log_prob - old_lp)
            policy_loss_1 = adv * ratio
            policy_loss_2 = adv * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            use_clipped_mask = policy_loss_2 < policy_loss_1

            if self.use_score_fisher and not self.use_adam_ablation:
                score_unclipped = ratio
                score_clipped = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                per_sample_loss = -th.where(use_clipped_mask, score_clipped, score_unclipped)
            else:
                policy_loss_per_sample = -th.min(policy_loss_1, policy_loss_2)
                if entropy is None:
                    entropy_loss_per_sample = -(-log_prob)
                else:
                    entropy_loss_per_sample = -entropy
                per_sample_loss = policy_loss_per_sample + self.ent_coef * entropy_loss_per_sample
            return per_sample_loss

        grad_fn = func_grad(per_sample_loss_fn)
        per_param_grads = vmap(
            grad_fn,
            in_dims=(None, None, 0, 0, 0, 0),
        )(params, buffers, observations, actions, old_log_prob, advantages)

        second_moment_dict: dict[int, th.Tensor] = {}
        actor_mean_grad_dict: Optional[dict[int, th.Tensor]] = {} if self.use_adam_ablation else None
        actor_param_ids = {id(p) for p in self._actor_params}

        for name, grads_for_param in per_param_grads.items():
            param = params[name]
            pid = id(param)
            if pid not in actor_param_ids:
                continue
            if self.use_adam_ablation:
                assert actor_mean_grad_dict is not None
                actor_mean_grad = grads_for_param.mean(dim=0).to(param.dtype)
                actor_mean_grad_dict[pid] = actor_mean_grad
            else:
                second_moment = grads_for_param.pow(2).mean(dim=0)
                second_moment_dict[pid] = second_moment.to(param.dtype)

        if self.use_adam_ablation:
            assert actor_mean_grad_dict is not None
            self._clip_actor_mean_grads(actor_mean_grad_dict)
            for pid, mean_grad in actor_mean_grad_dict.items():
                second_moment_dict[pid] = mean_grad * mean_grad

        return second_moment_dict, actor_mean_grad_dict

    def _clip_actor_mean_grads(self, actor_mean_grads_dict: dict[int, th.Tensor]) -> None:
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

