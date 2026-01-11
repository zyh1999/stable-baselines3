"""
Natural Policy Gradient (NPG) implemented in the SB3 style.

This implementation follows the "detach/true" style update:
- Compute natural gradient direction using a Fisher-vector product (local KL)
- Apply optional post-processing (fisher_clip/l2_clip/norm)
- Write the direction into ``.grad`` and update the actor using an optimizer step
- Optionally adapt the actor learning rate based on observed KL
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.distributions import kl_divergence, sum_independent_dims
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F

from sb3_contrib.common.utils import flat_grad
from stable_baselines3.npg.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.npg.rollout import collect_rollouts_npg
from stable_baselines3.npg.obs_norm import DetachObsNormConfig, DetachObsRMS


def conjugate_gradient(fn_fvp, g, nsteps=10, residual_tol=1e-10):
    x = th.zeros_like(g)
    r = g.clone()
    p = r.clone()
    rdotr = th.dot(r, r)

    for i in range(nsteps):
        z = fn_fvp(p)
        # Detach reference implementation uses no epsilon here.
        alpha = rdotr / th.dot(p, z)
        x += alpha * p
        r -= alpha * z
        new_rdotr = th.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
    return x

SelfNPG = TypeVar("SelfNPG", bound="NPG")


class NPG(OnPolicyAlgorithm):
    """
    Natural Policy Gradient (NPG)

    This is similar to TRPO but uses a single-step update scaled to meet a KL budget
    instead of a backtracking line search.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param lr_v: The learning rate for the value function (critic)
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
        lr_v: float | Schedule = 1e-3,
        n_steps: int = 2048,
        gamma: float = 0.99,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        target_kl: float = 0.01,
        lr_pi: float = 5e-2,
        use_kl_adaptive_lr: bool = True,
        lr_pi_min: float = 1e-4,
        lr_pi_max: float = 5e-2,
        lr_pi_adapt_factor: float = 1.5,
        pi_epochs: int = 1,
        pi_batch_size: int | None = None,
        v_epochs: int = 10,
        v_batch_size: int = 128,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = False,
        sub_sampling_factor: int = 1,
        action_squash: bool = False,
        clamp_ratio: bool = True,
        min_ratio: float = 0.1,
        max_ratio: float = 10.0,
        norm_obj: str = "adv",  # adv | obj | ratio
        grad_mode: str = "npg",  # pg | npg
        post_grad: str = "fisher_clip",  # fisher_clip | l2_clip | norm | none
        max_grad_norm: float = 0.5,
        max_grad_norm_v: float = 5.0,
        use_popart: bool = False,
        ent_coef: float = 0.0,
        debug_nan: bool = False,
        kl_ref: str = "current",  # current | old
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        use_detach_obs_rms: bool = False,
        detach_obs_rms_eps: float = 1e-8,
        detach_obs_rms_clip: float | None = None,
        detach_recompute_pi_old: bool = True,
        _init_setup_model: bool = True,
    ):
        # PopArt requires a custom value head (see `stable_baselines3.npg.custom_policies.NPGActorCriticPolicy`).
        # To make it easier to use from RL-ZOO configs, we auto-switch the default policies when use_popart=True.
        if use_popart:
            # Lazy import to avoid any potential circular imports at module import time.
            from stable_baselines3.npg.custom_policies import NPGActorCriticPolicy

            # RL-ZOO config commonly passes policy="MlpPolicy" (string). In that case, we override to the custom policy.
            # We also override if a default SB3 ActorCriticPolicy class/alias was passed directly.
            is_default_policy = False
            if isinstance(policy, str):
                is_default_policy = policy in self.policy_aliases
            else:
                is_default_policy = policy in {ActorCriticPolicy, MlpPolicy, CnnPolicy, MultiInputPolicy}

            if is_default_policy:
                if verbose > 0:
                    print(
                        "[NPG] use_popart=True: overriding `policy` to "
                        "`stable_baselines3.npg.custom_policies.NPGActorCriticPolicy`"
                    )
                policy = NPGActorCriticPolicy

            # Ensure the custom policy receives the flag; safe because we only inject when use_popart=True.
            policy_kwargs = {} if policy_kwargs is None else dict(policy_kwargs)
            if "use_popart" in policy_kwargs and bool(policy_kwargs["use_popart"]) is not True:
                raise ValueError("Got use_popart=True but policy_kwargs['use_popart'] is False.")
            policy_kwargs["use_popart"] = True

        super().__init__(
            policy,
            env,
            learning_rate=lr_v,
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

        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping
        self.target_kl = target_kl
        self.lr_pi = lr_pi
        self.use_kl_adaptive_lr = use_kl_adaptive_lr
        self.lr_pi_min = lr_pi_min
        self.lr_pi_max = lr_pi_max
        self.lr_pi_adapt_factor = lr_pi_adapt_factor
        self.pi_epochs = pi_epochs
        self.v_epochs = v_epochs
        self.v_batch_size = v_batch_size
        # If not specified, default actor minibatch size to critic minibatch size (clean default).
        self.pi_batch_size = v_batch_size if pi_batch_size is None else pi_batch_size
        self.normalize_advantage = normalize_advantage
        self.sub_sampling_factor = sub_sampling_factor
        self.action_squash = action_squash
        self.clamp_ratio = clamp_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.norm_obj = norm_obj
        self.grad_mode = grad_mode
        self.post_grad = post_grad
        self.max_grad_norm = max_grad_norm
        self.max_grad_norm_v = max_grad_norm_v
        self.use_popart = use_popart
        self.ent_coef = ent_coef
        self.debug_nan = debug_nan
        self.kl_ref = kl_ref

        # Detach-style obs normalization (mingfei):
        self.use_detach_obs_rms = use_detach_obs_rms
        self._detach_obs_norm_cfg = DetachObsNormConfig(
            eps=detach_obs_rms_eps, clip=detach_obs_rms_clip, recompute_pi_old=detach_recompute_pi_old
        )
        self._obs_norm: DetachObsRMS | None = None

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        # Use the custom rollout logic when we need detach-style action squashing and/or PopArt alignment.
        # Otherwise, fall back to SB3 built-in rollout collection to keep behavior standard.
        if getattr(self, "use_popart", False) or getattr(self, "action_squash", False):
            return collect_rollouts_npg(self, env, callback, rollout_buffer, n_rollout_steps)
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Override BaseAlgorithm.predict() to optionally apply detach-style action squashing at inference time.

        Important:
        - SB3's default ``policy.predict()`` clips actions to the action space when ``squash_output=False``.
          That is NOT equivalent to detach's behavior (tanh on the raw action).
        - Our training rollouts apply ``tanh`` to the raw (unclipped) Gaussian action when ``action_squash=True``.
        - To strictly match detach + our rollout collection, we must apply tanh to the raw action too.
        """
        # Replicate the relevant parts of BasePolicy.predict() but without the Box-action pre-clip,
        # so we can apply tanh squashing on the raw action (detach style).
        self.policy.set_training_mode(False)
        if self._obs_norm is not None and isinstance(observation, np.ndarray):
            observation = self._obs_norm.normalize_np(observation)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            raw_actions = self.policy._predict(obs_tensor, deterministic=deterministic)
        actions = raw_actions.cpu().numpy().reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if getattr(self, "action_squash", False) and isinstance(self.action_space, spaces.Box):
            high = self.action_space.high
            low = self.action_space.low
            # Strictly correct for symmetric bounds; fallback to clip otherwise.
            if np.allclose(low, -high):
                actions = np.tanh(actions) * high
            else:
                actions = np.clip(actions, low, high)
        else:
            # Default SB3 behavior for Box actions: clip to avoid out-of-bound errors
            if isinstance(self.action_space, spaces.Box):
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

    def _setup_model(self) -> None:
        super()._setup_model()
        if self.use_detach_obs_rms:
            self._obs_norm = DetachObsRMS(self.observation_space, self.device, self._detach_obs_norm_cfg)
        # Basic sanity checks similar to TRPO
        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            if self.normalize_advantage:
                assert buffer_size > 1, "`n_steps * n_envs` must be > 1 when normalizing advantage"
            if buffer_size % self.v_batch_size != 0:
                # Warn but do not stop; critic minibatches can be uneven
                self.logger.warning(
                    f"Rollout size {buffer_size} not divisible by v_batch_size {self.v_batch_size}; "
                    "the last critic minibatch may be truncated."
                )
        if self.pi_epochs < 1:
            raise ValueError("pi_epochs must be >= 1")
        if self.pi_batch_size < 1:
            raise ValueError("pi_batch_size must be >= 1")
        if self.v_epochs < 1:
            raise ValueError("v_epochs must be >= 1")
        if self.v_batch_size < 1:
            raise ValueError("v_batch_size must be >= 1")
        if self.norm_obj not in {"adv", "obj", "ratio"}:
            raise ValueError("norm_obj must be one of {'adv', 'obj', 'ratio'}")
        if self.grad_mode not in {"pg", "npg"}:
            raise ValueError("grad_mode must be 'pg' or 'npg'")
        if self.post_grad not in {"fisher_clip", "l2_clip", "norm", "none"}:
            raise ValueError("post_grad must be one of {'fisher_clip', 'l2_clip', 'norm', 'none'}")
        if self.kl_ref not in {"current", "old"}:
            raise ValueError("kl_ref must be 'current' or 'old'")
        if self.lr_pi <= 0:
            raise ValueError("lr_pi must be > 0")
        if self.lr_pi_min <= 0 or self.lr_pi_max <= 0:
            raise ValueError("lr_pi_min/lr_pi_max must be > 0")
        if self.lr_pi_min > self.lr_pi_max:
            raise ValueError("lr_pi_min must be <= lr_pi_max")
        if self.lr_pi_adapt_factor <= 1.0:
            raise ValueError("lr_pi_adapt_factor must be > 1.0")

        # Build an explicit critic optimizer (detach 对齐：critic 固定用 Adam)。
        # Avoid relying on parameter name strings (e.g., "value" in name).
        if not hasattr(self.policy, "mlp_extractor") or not hasattr(self.policy.mlp_extractor, "value_net"):
            raise RuntimeError("Policy does not expose mlp_extractor.value_net, cannot build critic optimizer explicitly.")
        if not hasattr(self.policy, "value_net"):
            raise RuntimeError("Policy does not expose value_net, cannot build critic optimizer explicitly.")

        # Explicitly collect critic parameters
        critic_params = list(self.policy.mlp_extractor.value_net.parameters()) + list(self.policy.value_net.parameters())

        if len(critic_params) == 0:
            raise RuntimeError("Could not find critic parameters to build critic optimizer.")
        self.critic_params = critic_params

        # 显式固定使用 Adam（不从 policy_kwargs 透传 optimizer_class）
        # 初始化时获取 schedule 的初始值（如果是 linear，这里就是 initial_lr）
        initial_lr = self.lr_schedule(1.0)
        self.critic_optimizer = th.optim.Adam(critic_params, lr=initial_lr)

        # Build an explicit actor optimizer (detach 对齐：actor 用 SGD，步长由 lr_pi 控制).
        critic_param_ids = {id(p) for p in critic_params}
        actor_params = [p for p in self.policy.parameters() if p.requires_grad and id(p) not in critic_param_ids]
        if len(actor_params) == 0:
            raise RuntimeError("Could not find actor parameters to build actor optimizer.")
        self.actor_params = actor_params
        self.actor_optimizer = th.optim.SGD(actor_params, lr=self.lr_pi, momentum=0.0, weight_decay=0.0)

    def _flatten_grads(self, grads: list[th.Tensor | None], params: list[nn.Parameter]) -> th.Tensor:
        """Flatten gradients, inserting zeros for None, preserving parameter order."""
        flat = []
        for g, p in zip(grads, params):
            if g is None:
                flat.append(th.zeros_like(p).view(-1))
            else:
                flat.append(g.contiguous().view(-1))
        return th.cat(flat)

    def _set_grads_from_flat(self, params: list[nn.Parameter], flat: th.Tensor) -> None:
        """Write a flat vector into ``.grad`` fields of parameters (detach style)."""
        start = 0
        for p in params:
            n = p.numel()
            g = flat[start : start + n].view_as(p)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)
            start += n

    def hessian_vector_product(
        self,
        params: list[nn.Parameter],
        grad_kl_flat: th.Tensor,
        vector: th.Tensor,
        retain_graph: bool = True,
    ) -> th.Tensor:
        """Fisher-vector product with optional damping."""
        jacobian_vector_product = (grad_kl_flat * vector).sum()
        hvp_list = th.autograd.grad(
            jacobian_vector_product,
            params,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        hvp_flat = self._flatten_grads(list(hvp_list), params)
        return hvp_flat + self.cg_damping * vector

    def train(self) -> None:
        self.policy.set_training_mode(True)

        policy_objective_values = []
        kl_divergences = []
        value_losses = []

        # Detach-style: update obs RMS stats with current rollout observations BEFORE training.
        if self._obs_norm is not None:
            self._obs_norm.update_from_rollout_buffer(self.rollout_buffer)

        # PopArt (detach 对齐): update running stats with current rollout returns BEFORE actor/critic updates.
        # Important: use shape [N, 1] to match PopArt(norm_axes=1) broadcasting semantics.
        if self.use_popart:
            with th.no_grad():
                returns_tensor = th.as_tensor(self.rollout_buffer.returns, device=self.device).view(-1, 1)
                self.policy.value_net.update(returns_tensor)

        # 复制旧策略参数，用于多 epoch/mini-batch 时稳定计算 KL(old || new) 与 old_distribution
        old_policy = copy.deepcopy(self.policy)
        old_policy.set_training_mode(False)

        # Actor: multi-epoch minibatch updates (detach 对齐: pi_epochs + pi_batch_size)
        total_samples = int(self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs)
        actor_batch_size = self.pi_batch_size
        if total_samples % actor_batch_size != 0:
            self.logger.warning(
                f"Rollout size {total_samples} not divisible by pi_batch_size {actor_batch_size}; "
                "the last actor minibatch may be truncated."
            )

        for _ in range(self.pi_epochs):
            for rollout_data in self.rollout_buffer.get(batch_size=actor_batch_size):
                if self.sub_sampling_factor > 1:
                    # RolloutBufferSamples fields: observations, actions, old_values, old_log_prob, advantages, returns
                    rollout_data = RolloutBufferSamples(
                        rollout_data.observations[:: self.sub_sampling_factor],
                        rollout_data.actions[:: self.sub_sampling_factor],
                        rollout_data.old_values[:: self.sub_sampling_factor],
                        rollout_data.old_log_prob[:: self.sub_sampling_factor],
                        rollout_data.advantages[:: self.sub_sampling_factor],
                        rollout_data.returns[:: self.sub_sampling_factor],
                    )

                obs_t = rollout_data.observations
                if self._obs_norm is not None:
                    obs_t = self._obs_norm.normalize_tensor(obs_t)

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # KL reference distribution:
                # - detach 对齐 (algo='true'): use current policy (detached) as reference => local KL, Fisher should be PSD.
                # - optional: use old_policy as reference (non-local KL), may lead to indefinite curvature.
                with th.no_grad():
                    if self.kl_ref == "current":
                        ref_distribution = self.policy.get_distribution(obs_t)
                    else:
                        ref_distribution = old_policy.get_distribution(obs_t)

                distribution = self.policy.get_distribution(obs_t)
                log_prob = distribution.log_prob(actions)

                advantages = rollout_data.advantages
                # PopArt (detach 对齐): normalize advantages using PopArt stats (mean/std of returns).
                # Detach then subtracts adv.mean() inside the update, cancelling the mean shift;
                # net effect is scaling by 1/std.
                if self.use_popart:
                    advantages = self.policy.value_net.normalize(advantages.unsqueeze(-1)).squeeze(-1)
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # detach风格：零均值优势
                advantages = advantages - advantages.mean()

                # Detach-style: after updating obs RMS stats, recompute pi_old (old log-prob) under updated normalization.
                if self._obs_norm is not None and self._obs_norm.cfg.recompute_pi_old:
                    with th.no_grad():
                        old_log_prob = old_policy.get_distribution(obs_t).log_prob(actions)
                else:
                    old_log_prob = rollout_data.old_log_prob

                ratio = th.exp(log_prob - old_log_prob)
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
                # detach 对齐：Fisher 用的是 KL(ref || new)
                # Important: SB3's kl_divergence() returns per-dimension KL for Normal (shape [B, act_dim]).
                # Detach uses sum over action dims then mean over batch: sum(-1).mean().
                kl_div = sum_independent_dims(kl_divergence(ref_distribution, distribution)).mean()
                # detach 对齐：这里只优化 surrogate objective（当前配置 ent_coef=0，对齐时不引入 entropy 项）
                loss = -policy_objective

                # Fisher-vector product is based on local KL (kl_div) like detach's _ent_kl
                kl_grads = th.autograd.grad(kl_div, self.actor_params, create_graph=True, retain_graph=True, allow_unused=True)
                grad_kl_flat = self._flatten_grads(list(kl_grads), self.actor_params)
                hessian_vector_product_fn = partial(self.hessian_vector_product, self.actor_params, grad_kl_flat)

                # Compute gradients of loss for actor params (do not rely on .grad)
                loss_grads = th.autograd.grad(loss, self.actor_params, retain_graph=True, allow_unused=True)
                loss_grad_flat = self._flatten_grads(list(loss_grads), self.actor_params).detach()

                if self.grad_mode == "pg":
                    step_dir = loss_grad_flat
                else:
                    step_dir = conjugate_gradient(
                        hessian_vector_product_fn,
                        loss_grad_flat,
                        nsteps=self.cg_max_steps,
                    )

                # 后处理：按 Fisher/L2 规范化或裁剪
                if self.post_grad == "fisher_clip":
                    fisher_norm = th.matmul(step_dir, hessian_vector_product_fn(step_dir, retain_graph=True))
                    if fisher_norm.item() < 0.0:
                        raise AssertionError(f"fisher_norm is negative: {fisher_norm.item()}")
                    step_dir = step_dir * th.clamp(self.max_grad_norm / fisher_norm, max=1.0)
                elif self.post_grad == "l2_clip":
                    l2_norm = th.matmul(step_dir, step_dir)
                    if l2_norm.item() < 0.0:
                        raise AssertionError(f"l2_norm is negative: {l2_norm.item()}")
                    step_dir = step_dir * th.clamp(self.max_grad_norm / l2_norm, max=1.0)
                elif self.post_grad == "norm":
                    fisher_norm = th.matmul(step_dir, hessian_vector_product_fn(step_dir, retain_graph=True))
                    if fisher_norm.item() < 0.0:
                        raise AssertionError(f"fisher_norm is negative: {fisher_norm.item()}")
                    step_dir = step_dir / fisher_norm.sqrt()

                # detach 对齐：把自然梯度方向写入 .grad，然后用 SGD.step() 更新（步长由 lr_pi 控制）
                self.actor_optimizer.zero_grad()
                self._set_grads_from_flat(self.actor_params, step_dir)
                self.actor_optimizer.step()

                with th.no_grad():
                    # For logging: compute "real" KL against old_policy (like detach's real_kl using outputs_old)
                    old_distribution_for_log = old_policy.get_distribution(obs_t)
                    # This is where NaNs often surface (distribution mean/log_std)
                    new_distribution = self.policy.get_distribution(obs_t)
                    new_log_prob = new_distribution.log_prob(actions)
                    new_ratio = th.exp(new_log_prob - old_log_prob)
                    if self.clamp_ratio:
                        new_ratio = th.clamp(new_ratio, self.min_ratio, self.max_ratio)
                    new_policy_objective = (new_ratio * advantages).mean()
                    # detach 对齐：日志/自适应 lr 用的是 KL(old || new)，并且同样是 sum(-1).mean()
                    new_kl = sum_independent_dims(kl_divergence(old_distribution_for_log, new_distribution)).mean()

                policy_objective_values.append(new_policy_objective.item())
                kl_divergences.append(new_kl.item())

        # detach 对齐：KL 自适应调 actor lr（默认用 target_kl 作为阈值）
        if self.use_kl_adaptive_lr and len(kl_divergences) > 0:
            curr_kl = float(np.mean(kl_divergences))
            lr = float(self.actor_optimizer.param_groups[0]["lr"])
            if curr_kl > self.target_kl * 2.0:
                lr = max(lr / self.lr_pi_adapt_factor, self.lr_pi_min)
            elif curr_kl < self.target_kl / 2.0:
                lr = min(lr * self.lr_pi_adapt_factor, self.lr_pi_max)
            self.actor_optimizer.param_groups[0]["lr"] = lr
            self.logger.record("train/lr_pi", lr)

        for _ in range(self.v_epochs):
            for rollout_data in self.rollout_buffer.get(self.v_batch_size):
                obs_t = rollout_data.observations
                if self._obs_norm is not None:
                    obs_t = self._obs_norm.normalize_tensor(obs_t)
                target_values = rollout_data.returns.view(-1, 1)
                if self.use_popart:
                    # PopArt 路径：保留“层层显式”计算（用于对齐/调试）
                    with th.no_grad():
                        target_values = self.policy.value_net.normalize(target_values)

                    features = self.policy.extract_features(obs_t)
                    if self.policy.share_features_extractor:
                        latent_vf = self.policy.mlp_extractor.forward_critic(features)
                    else:
                        pi_features, vf_features = features
                        latent_vf = self.policy.mlp_extractor.forward_critic(vf_features)

                    values_pred = self.policy.value_net(latent_vf)  # normalized value
                    value_loss = F.mse_loss(values_pred, target_values)
                else:
                    # 非 PopArt：走 SB3 原生路径，不再手动拆网络结构
                    values_pred = self.policy.predict_values(obs_t)
                    value_loss = F.mse_loss(values_pred, target_values)
                value_losses.append(value_loss.item())

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                # detach 对齐：critic 梯度裁剪（默认 5.0）
                if self.max_grad_norm_v > 0:
                    th.nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm_v)
                self.critic_optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        # Continuous actions: log action std in a way that supports both
        # - SB3-style global log_std parameter (Tensor)
        # - state_dependent_std=True where policy.log_std is None and std is produced by the net
        std_mean: float | None = None
        if hasattr(self.policy, "log_std") and th.is_tensor(self.policy.log_std):
            std_mean = th.exp(self.policy.log_std).mean().item()
        else:
            try:
                with th.no_grad():
                    sample = next(self.rollout_buffer.get(1))
                    dist = self.policy.get_distribution(sample.observations)
                    if hasattr(dist, "distribution") and hasattr(dist.distribution, "scale"):
                        std_mean = dist.distribution.scale.mean().item()
            except StopIteration:
                pass
        if std_mean is not None:
            self.logger.record("train/std", std_mean)
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

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        # Save explicit critic optimizer state when present
        return ["policy", "policy.optimizer", "critic_optimizer", "actor_optimizer"], []
