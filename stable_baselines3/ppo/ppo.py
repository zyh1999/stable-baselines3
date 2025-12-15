import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3.ppo._score_adam import ScoreAdam

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param separate_optimizers: If True, use two optimizers to update actor and critic separately
        (hyperparameters identical). Shared feature extractor (if any) is updated once using
        the combined gradients from both losses.
    :param use_score_fisher: When using ``ScoreAdam`` for the actor with separate optimizers,
        controls how the second-moment estimate is constructed:
        - If True (default), use score-only per-sample losses (without advantage) to build a
          Fisher-like diagonal E[g^2].
        - If False, use the true per-sample actor loss (policy + entropy terms) to build E[g^2].
    :param use_adam_ablation: When using ``ScoreAdam`` for the actor, enable an Adam-style ablation:
        - The numerator uses per-sample actor gradients with max_grad_norm clipping, then averages them
          to build E[g].
        - The denominator uses (E[g])^2 as the second-moment estimate, mimicking standard Adam while
          still exposing per-sample control.
    :param disable_joint_critic_update: 若为 True，则在联合阶段（actor 更新阶段）不更新 critic，
        只保留 critic loss 计算用于日志。需要 separate_optimizers=True 才能真正冻结 critic。
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
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
        use_score_fisher: bool = True,
        use_adam_ablation: bool = False,
        disable_joint_critic_update: bool = False,
        critic_rollout_multiplier: int = 4,
        critic_warmup_epochs: Optional[int] = None,
        critic_warmup_batch_size: Optional[int] = None,
        enable_critic_warmup: bool = True,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
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

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.advantage_multiplier = advantage_multiplier
        self.normalize_advantage_mean = normalize_advantage_mean
        self.normalize_advantage_std = normalize_advantage_std
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.separate_optimizers = separate_optimizers
        # 控制 ScoreAdam 的二阶矩是基于 score 还是基于真实 actor loss 的梯度
        self.use_score_fisher = use_score_fisher
        # 消融开关：让 ScoreAdam 按“普通 Adam”风格工作（分子/分母都基于 batch-mean 梯度）
        self.use_adam_ablation = use_adam_ablation
        self.disable_joint_critic_update = disable_joint_critic_update

        # Split-optimizer related attributes
        self.actor_optimizer: Optional[th.optim.Optimizer] = None
        self.critic_optimizer: Optional[th.optim.Optimizer] = None
        self._actor_params: Optional[list[th.nn.Parameter]] = None
        self._critic_params: Optional[list[th.nn.Parameter]] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

        # When requested, build two optimizers with separated parameter groups
        if self.separate_optimizers:
            # Helpers to collect unique parameters
            def _extend_unique(dst: list[th.nn.Parameter], params_iter) -> None:
                seen = {id(p) for p in dst}
                for p in params_iter:
                    if id(p) not in seen:
                        dst.append(p)
                        seen.add(id(p))

            actor_params: list[th.nn.Parameter] = []
            critic_params: list[th.nn.Parameter] = []

            # Actor-specific modules
            _extend_unique(actor_params, self.policy.mlp_extractor.policy_net.parameters())
            _extend_unique(actor_params, self.policy.action_net.parameters())
            if hasattr(self.policy, "log_std") and isinstance(self.policy.log_std, th.nn.Parameter):
                actor_params.append(self.policy.log_std)

            # Critic-specific modules
            _extend_unique(critic_params, self.policy.mlp_extractor.value_net.parameters())
            _extend_unique(critic_params, self.policy.value_net.parameters())

            # Feature extractors: shared or separate
            if getattr(self.policy, "share_features_extractor", True):
                _extend_unique(actor_params, self.policy.features_extractor.parameters())
            else:
                _extend_unique(actor_params, self.policy.pi_features_extractor.parameters())
                _extend_unique(critic_params, self.policy.vf_features_extractor.parameters())

            # Save param lists for clipping/logging
            self._actor_params = actor_params
            self._critic_params = critic_params

            # Create optimizers mirroring policy optimizer hyperparameters
            initial_lr = self.lr_schedule(1)
            self.actor_optimizer = self.policy.optimizer_class(self._actor_params, lr=initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]
            self.critic_optimizer = self.policy.optimizer_class(self._critic_params, lr=initial_lr, **self.policy.optimizer_kwargs)  # type: ignore[arg-type]

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        if self.separate_optimizers:
            assert self.actor_optimizer is not None and self.critic_optimizer is not None
            if self.disable_joint_critic_update:
                self._update_learning_rate([self.actor_optimizer])
            else:
                self._update_learning_rate([self.actor_optimizer, self.critic_optimizer])
        else:
            self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        # New metrics: clip fraction for advantages < threshold
        clip_fractions_adv_02 = []
        clip_fractions_adv_04 = []
        clip_fractions_adv_06 = []
        clip_fractions_adv_08 = []
        clip_fractions_adv_10 = []
        
        # New: Record the fraction of samples falling into each advantage range
        frac_adv_02 = []
        frac_adv_04 = []
        frac_adv_06 = []
        frac_adv_08 = []
        frac_adv_10 = []
        
        batch_advantages = []
        
        # New: Percentile-based clip fractions (0-20%, 20-40%, etc.)
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
        # train for n_epochs epochs
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
                if self.normalize_advantage and len(advantages) > 1:                    # Apply mean and/or std normalization based on switches
                    # If both switches are False, keep advantages as-is
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

                # Calculate clip fractions for different advantage magnitudes
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

                # New: Calculate clip fractions for advantage percentiles (0-20%, 20-40%, etc.)
                # This separates samples by their RELATIVE rank in the batch, not absolute value.
                with th.no_grad():
                    adv_abs = th.abs(advantages)
                    # Sort advantages to identify indices for each percentile bucket
                    # argsort gives the indices that would sort the array
                    sorted_indices = th.argsort(adv_abs)
                    batch_size = len(adv_abs)
                    
                    # Define bucket boundaries
                    idx_20 = int(batch_size * 0.2)
                    idx_40 = int(batch_size * 0.4)
                    idx_60 = int(batch_size * 0.6)
                    idx_80 = int(batch_size * 0.8)
                    
                    # Create masks for each bucket based on sorted order
                    # We create a boolean mask of the same shape as adv_abs
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
                            # Calculate clip fraction for samples in this percentile range
                            fraction = clipped_batch[indices].mean().item()
                            storage.append(fraction)

                if self.clip_range_vf is None:
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

                combined_loss = policy_loss + self.ent_coef * entropy_loss + (0.0 if (self.disable_joint_critic_update and self.separate_optimizers) else self.vf_coef * value_loss)

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optional: compute per-sample gradients for ScoreAdam.
                #
                # 对 ScoreAdam 有三种模式（仅在 separate_optimizers=True 时启用）：
                # - use_score_fisher=True 且 use_adam_ablation=False:
                #     分母用基于 score-only 的 E[g^2]（Fisher 对角近似），分子来自标准 batch loss backward。
                # - use_score_fisher=False 且 use_adam_ablation=False:
                #     分母用基于真实 actor loss（含 advantage + 熵）的 E[g^2]。
                # - use_adam_ablation=True:
                #     做“普通 Adam”的消融版本：
                #       * 分子：对 per-sample 的 actor loss 梯度先做 max_grad_norm 裁剪，再取平均 E[g]；
                #       * 分母：对这个平均梯度按元素平方得到 (E[g])^2，传给 ScoreAdam 作为自适应二阶矩。
                #     这样就模拟了 Adam 中基于 batch-mean 梯度的二阶矩估计，但仍然保留 per-sample 级别的裁剪控制。
                score_grads_dict: Optional[dict[int, th.Tensor]] = None
                actor_mean_grads_dict: Optional[dict[int, th.Tensor]] = None
                if self.separate_optimizers and isinstance(self.actor_optimizer, ScoreAdam):
                    assert self._actor_params is not None

                    # 选择 per-sample loss 的模式
                    if self.use_adam_ablation:
                        # Adam 消融 & true-Fisher 都使用真实 actor loss（policy + entropy）
                        # 这里只是先构造逐样本的 actor loss，后续根据 use_adam_ablation 决定如何聚合梯度
                        policy_loss_per_sample = -th.min(policy_loss_1, policy_loss_2)  # [batch]
                        if entropy is None:
                            # 与上面 entropy_loss = -mean(-log_prob) 对应：逐样本为 log_prob
                            entropy_loss_per_sample = -(-log_prob)  # = log_prob
                        else:
                            # 与上面 entropy_loss = -mean(entropy) 对应：逐样本为 -entropy
                            entropy_loss_per_sample = -entropy
                        per_sample_loss = policy_loss_per_sample + self.ent_coef * entropy_loss_per_sample  # [batch]
                    elif self.use_score_fisher:
                        # 1) 基于 "score-only" 的逐样本 loss：
                        #    - 保留 ratio / clip 结构
                        #    - 去掉 advantage 和 entropy 的缩放（分母只看 score）
                        score_unclipped = ratio
                        score_clipped = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                        # PPO 原始目标在每个样本上选择 unclipped 或 clipped，这里用同一个 mask
                        score_objective = th.where(use_clipped_mask, score_clipped, score_unclipped)  # [batch]
                        # 仅用 policy 的 score 部分作为逐样本 loss
                        per_sample_loss = -score_objective
                    else:
                        # True-Fisher：基于真实 actor loss（policy + entropy）的逐样本 loss，用于构造 E[g^2]
                        policy_loss_per_sample = -th.min(policy_loss_1, policy_loss_2)  # [batch]
                        if entropy is None:
                            entropy_loss_per_sample = -(-log_prob)  # = log_prob
                        else:
                            entropy_loss_per_sample = -entropy
                        per_sample_loss = policy_loss_per_sample + self.ent_coef * entropy_loss_per_sample  # [batch]

                    # 2) 逐样本梯度计算
                    #
                    # - Fisher / True-Fisher 模式：累积 g^2
                    # - Adam 消融模式：累积原始梯度 g，用于后续计算 batch 平均梯度
                    second_moment_acc: dict[int, th.Tensor] = {
                        id(p): th.zeros_like(p) for p in self._actor_params
                    }
                    if self.use_adam_ablation:
                        actor_grad_sum: dict[int, th.Tensor] = {
                            id(p): th.zeros_like(p) for p in self._actor_params
                        }

                    batch_size = per_sample_loss.shape[0]
                    for j in range(batch_size):
                        loss_j = per_sample_loss[j]
                        per_grads = th.autograd.grad(
                            loss_j, self._actor_params, retain_graph=True, allow_unused=True
                        )
                        
                        for p, g in zip(self._actor_params, per_grads):
                            if g is None:
                                continue
                            pid = id(p)
                            if self.use_adam_ablation:
                                # Adam ablation: Accumulate RAW gradients
                                actor_grad_sum[pid] += g
                            else:
                                # Fisher / True-Fisher: accumulate g^2
                                second_moment_acc[pid] += g * g

                    if self.use_adam_ablation:
                        # Adam ablation:
                        # 1. Calculate E[g] using accumulated RAW gradients
                        actor_mean_grads_dict = {
                            pid: (g_sum / batch_size) for pid, g_sum in actor_grad_sum.items()
                        }
                        
                        # 2. Clip the AVERAGED gradients E[g] (Standard Adam/PPO behavior)
                        mean_grads_list = [actor_mean_grads_dict[id(p)] for p in self._actor_params]
                        if th.isfinite(th.as_tensor(self.max_grad_norm)) and self.max_grad_norm > 0:
                            th.nn.utils.clip_grad_norm_(mean_grads_list, self.max_grad_norm)
                            
                        # 3. Square the CLIPPED averaged gradients to get (E[g])^2
                        score_grads_dict = {
                            id(p): (g_mean * g_mean) for p, g_mean in zip(self._actor_params, mean_grads_list)
                        }
                    else:
                        # Fisher / True-Fisher: 对 score-only 或真实 actor loss 的梯度做 E[g^2]
                        score_grads_dict = {
                            pid: (acc / batch_size) for pid, acc in second_moment_acc.items()
                        }

                # Optimization step
                if self.separate_optimizers:
                    assert self.actor_optimizer is not None and self.critic_optimizer is not None
                    assert self._actor_params is not None and self._critic_params is not None

                    # Zero-grad both optimizers
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    actor_loss = policy_loss + self.ent_coef * entropy_loss
                    critic_loss = self.vf_coef * value_loss

                    if isinstance(self.actor_optimizer, ScoreAdam) and self.use_adam_ablation:
                        critic_loss.backward()
                        assert actor_mean_grads_dict is not None
                        for p in self._actor_params:
                            pid = id(p)
                            g_actor = actor_mean_grads_dict[pid]
                            if p.grad is None:
                                p.grad = g_actor.detach().clone()
                            else:
                                p.grad = p.grad + g_actor.detach()
                    else:
                        if self.disable_joint_critic_update:
                            # 只反向 actor，不反向 critic
                            actor_loss.backward()
                        else:
                            actor_loss.backward(retain_graph=True)
                            critic_loss.backward()

                    # Clip and step actor
                    actor_grad_norms.append(
                        th.nn.utils.clip_grad_norm_(self._actor_params, self.max_grad_norm).cpu().numpy()
                    )
                    if isinstance(self.actor_optimizer, ScoreAdam):
                        self.actor_optimizer.step(score_grads=score_grads_dict)
                    else:
                        self.actor_optimizer.step()

                    if not self.disable_joint_critic_update:
                        critic_grad_norms.append(
                            th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm).cpu().numpy()
                        )
                        self.critic_optimizer.step()
                    # For compatibility with later logging
                    loss = combined_loss
                else:
                    if self.disable_joint_critic_update and self.verbose >= 1:
                        warnings.warn("disable_joint_critic_update 在 separate_optimizers=False 时无效，将更新 actor+critic。")
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

        # Logs
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

        # Record last combined loss value
        self.logger.record("train/loss", combined_loss.item())
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

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Include separate optimizers in state dicts when enabled.
        """
        if self.separate_optimizers:
            return ["policy", "policy.optimizer", "actor_optimizer", "critic_optimizer"], []
        # Default behavior from parent
        return super()._get_torch_save_params()

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
