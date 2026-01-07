"""
Natural Policy Gradient (NPG) implemented in the SB3 style.

This follows the TRPO structure but removes backtracking line search and
uses a closed-form step size scaled to a KL budget.
"""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from stable_baselines3.npg.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

SelfNPG = TypeVar("SelfNPG", bound="NPG")


class NPG(OnPolicyAlgorithm):
    """
    Natural Policy Gradient (NPG)

    This is similar to TRPO but uses a single-step update scaled to meet a KL budget
    instead of a backtracking line search.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate for the value function
    :param n_steps: The number of steps to run for each environment per update
    :param batch_size: Minibatch size for the value function updates
    :param gamma: Discount factor
    :param cg_max_steps: Maximum iterations for the conjugate gradient solver
    :param cg_damping: Damping term in the Hessian-vector product
    :param target_kl: KL budget used to scale the natural gradient step
    :param n_critic_updates: Number of critic updates per policy update
    :param gae_lambda: GAE lambda
    :param normalize_advantage: Whether to normalize advantage
    :param sub_sampling_factor: Sub-sample rollout batch for speed (>=1)
    :param stats_window_size: Window size for logging statistics
    :param tensorboard_log: Tensorboard log location
    :param policy_kwargs: Keyword args for the policy
    :param verbose: Verbosity
    :param seed: Random seed
    :param device: Torch device
    :param _init_setup_model: Whether to build the network at instantiation
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        target_kl: float = 0.01,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = False,
        sub_sampling_factor: int = 1,
        clamp_ratio: bool = True,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0,
        norm_obj: str = "adv",  # adv | obj | ratio
        grad_mode: str = "npg",  # pg | npg
        post_grad: str = "fisher_clip",  # fisher_clip | l2_clip | norm | none
        max_grad_norm: float = 0.5,
        ent_coef: float = 0.0,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=0.0,
            vf_coef=0.0,
            max_grad_norm=0.0,
            use_sde=False,
            sde_sample_freq=-1,
            rollout_buffer_class=None,
            rollout_buffer_kwargs=None,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self.batch_size = batch_size
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        self.target_kl = target_kl
        self.n_critic_updates = n_critic_updates
        self.normalize_advantage = normalize_advantage
        self.sub_sampling_factor = sub_sampling_factor
        self.clamp_ratio = clamp_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.norm_obj = norm_obj
        self.grad_mode = grad_mode
        self.post_grad = post_grad
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        # Basic sanity checks similar to TRPO
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            if self.normalize_advantage:
                assert buffer_size > 1, "`n_steps * n_envs` must be > 1 when normalizing advantage"
            if buffer_size % self.batch_size != 0:
                # Warn but do not stop; critic minibatches can be uneven
                self.logger.warning(
                    f"Rollout size {buffer_size} not divisible by batch_size {self.batch_size}; "
                    "the last critic minibatch may be truncated."
                )
        if self.norm_obj not in {"adv", "obj", "ratio"}:
            raise ValueError("norm_obj must be one of {'adv', 'obj', 'ratio'}")
        if self.grad_mode not in {"pg", "npg"}:
            raise ValueError("grad_mode must be 'pg' or 'npg'")
        if self.post_grad not in {"fisher_clip", "l2_clip", "norm", "none"}:
            raise ValueError("post_grad must be one of {'fisher_clip', 'l2_clip', 'norm', 'none'}")

    def _compute_actor_grad(
        self, kl_div: th.Tensor, policy_objective: th.Tensor
    ) -> tuple[list[nn.Parameter], th.Tensor, th.Tensor, list[tuple[int, ...]]]:
        """Compute gradients for policy parameters only."""
        policy_objective_gradients_list = []
        grad_kl_list = []
        grad_shape: list[tuple[int, ...]] = []
        actor_params: list[nn.Parameter] = []

        for name, param in self.policy.named_parameters():
            if "value" in name:
                continue

            kl_param_grad, *_ = th.autograd.grad(
                kl_div,
                param,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
                only_inputs=True,
            )
            if kl_param_grad is not None:
                policy_objective_grad, *_ = th.autograd.grad(policy_objective, param, retain_graph=True, only_inputs=True)
                grad_shape.append(kl_param_grad.shape)
                grad_kl_list.append(kl_param_grad.reshape(-1))
                policy_objective_gradients_list.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)

        policy_objective_gradients = th.cat(policy_objective_gradients_list)
        grad_kl = th.cat(grad_kl_list)
        return actor_params, policy_objective_gradients, grad_kl, grad_shape

    def hessian_vector_product(
        self, params: list[nn.Parameter], grad_kl: th.Tensor, vector: th.Tensor, retain_graph: bool = True
    ) -> th.Tensor:
        """Fisher-vector product with optional damping."""
        jacobian_vector_product = (grad_kl * vector).sum()
        return flat_grad(jacobian_vector_product, params, retain_graph=retain_graph) + self.cg_damping * vector

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        policy_objective_values = []
        kl_divergences = []
        value_losses = []
        step_scales = []

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    None,
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = rollout_data.actions.long().flatten()

            with th.no_grad():
                old_distribution = self.policy.get_distribution(rollout_data.observations)

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # detach风格：零均值优势
            advantages = advantages - advantages.mean()

            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            if self.clamp_ratio:
                ratio = th.clamp(ratio, self.min_ratio, self.max_ratio)

            # 归一化目标
            if self.norm_obj == "adv":
                rms_sqrt = th.sqrt(th.mean(advantages.pow(2))).detach()
            elif self.norm_obj == "obj":
                rms_sqrt = th.sqrt(th.mean((ratio * advantages).pow(2))).detach()
            else:  # ratio
                rms_sqrt = ratio.mean().detach() * th.sqrt(th.mean(advantages.pow(2))).detach()

            surrogate = ratio * advantages / (rms_sqrt + 1e-8)
            policy_objective = surrogate.mean()
            # 注意：我们最大化 policy_objective；loss 在后面通过负号体现
            kl_div = kl_divergence(distribution, old_distribution).mean()

            self.policy.optimizer.zero_grad()
            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(kl_div, policy_objective)

            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)

            if self.grad_mode == "pg":
                search_direction = policy_objective_gradients
            else:
                search_direction = conjugate_gradient_solver(
                    hessian_vector_product_fn,
                    policy_objective_gradients,
                    max_iter=self.cg_max_steps,
                )

            # 后处理：按 Fisher/L2 规范化或裁剪
            if self.post_grad == "fisher_clip":
                fisher_norm = th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=True))
                fisher_norm = th.clamp(fisher_norm, min=1e-12)
                search_direction = search_direction * th.clamp(self.max_grad_norm / fisher_norm, max=1.0)
            elif self.post_grad == "l2_clip":
                l2_norm = th.matmul(search_direction, search_direction)
                l2_norm = th.clamp(l2_norm, min=1e-12)
                search_direction = search_direction * th.clamp(self.max_grad_norm / l2_norm, max=1.0)
            elif self.post_grad == "norm":
                fisher_norm = th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=True))
                fisher_norm = th.clamp(fisher_norm, min=1e-12)
                search_direction = search_direction / fisher_norm.sqrt()

            denom = th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
            denom = denom + 1e-8
            step_scale = th.sqrt(2 * self.target_kl / denom)

            start_idx = 0
            with th.no_grad():
                for param, shape in zip(actor_params, grad_shape):
                    n_params = param.numel()
                    param.data = param.data + step_scale * search_direction[start_idx : (start_idx + n_params)].view(shape)
                    start_idx += n_params

            with th.no_grad():
                new_distribution = self.policy.get_distribution(rollout_data.observations)
                new_log_prob = new_distribution.log_prob(actions)
                new_ratio = th.exp(new_log_prob - rollout_data.old_log_prob)
                if self.clamp_ratio:
                    new_ratio = th.clamp(new_ratio, self.min_ratio, self.max_ratio)
                new_policy_objective = (new_ratio * advantages).mean()
                new_kl = kl_divergence(new_distribution, old_distribution).mean()

            policy_objective_values.append(new_policy_objective.item())
            kl_divergences.append(new_kl.item())
            step_scales.append(step_scale.item())

        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # 清除 actor 共享参数上的梯度，保持 KL 约束
                for name, param in self.policy.named_parameters():
                    if "value" not in name:
                        param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/step_scale", np.mean(step_scales))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def learn(
        self: SelfNPG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "NPG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfNPG:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

