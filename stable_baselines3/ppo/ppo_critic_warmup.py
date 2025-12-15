import warnings
from typing import Any, Optional, Union

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo.ppo import PPO


class PPOCriticWarmup(PPO):
    """
    PPO 变体：默认分离 actor/critic 优化器，先用更大的 rollout 只训练 critic，
    再用常规 rollout 训练 actor+critic。优化器仅用常规 Adam，不涉及 ScoreAdam。

    新增参数：
    - critic_rollout_multiplier: critic 预训练阶段 rollout 步数放大倍数，>1 表示先用
      ``critic_rollout_multiplier * n_steps`` 的数据只训练 critic。默认 4。
    - critic_warmup_epochs: critic 预训练阶段 epoch 数（每次 critic rollout 收集完后，对这批数据重复训练几轮）。
      默认 5，和 actor 的 n_epochs 完全分开。
    - critic_warmup_batch_size: critic 预训练阶段 batch 大小，None 时与 batch_size 相同。
    - enable_critic_warmup: 是否启用两阶段流程。默认启用。
    - disable_joint_critic_update: 是否在联合阶段冻结 critic（只更新 actor）。默认 False。
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
        critic_rollout_multiplier: int = 4,
        critic_warmup_epochs: int = 5,
        critic_warmup_batch_size: Optional[int] = None,
        enable_critic_warmup: bool = True,
        disable_joint_critic_update: bool = True,
        _init_setup_model: bool = True,
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
            use_score_fisher=False,  # 不使用 ScoreAdam
            use_adam_ablation=False,
            disable_joint_critic_update=disable_joint_critic_update,
            _init_setup_model=False,  # 我们稍后手动 setup 以便先记录自定义字段
        )

        # critic 预训练配置
        assert critic_rollout_multiplier >= 1, "`critic_rollout_multiplier` 必须 ≥ 1"
        self.critic_rollout_multiplier = critic_rollout_multiplier
        self.critic_rollout_steps = self.n_steps * self.critic_rollout_multiplier
        assert critic_warmup_epochs >= 1, "`critic_warmup_epochs` 必须 ≥ 1"
        self.critic_warmup_epochs = critic_warmup_epochs
        self.critic_warmup_batch_size = critic_warmup_batch_size if critic_warmup_batch_size is not None else self.batch_size
        self.enable_critic_warmup = enable_critic_warmup
        self.disable_joint_critic_update = disable_joint_critic_update
        if self.enable_critic_warmup and not self.separate_optimizers:
            warnings.warn("critic 预训练需要 `separate_optimizers=True`，已自动关闭该功能。")
            self.enable_critic_warmup = False
        self._n_critic_updates = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # 先走父类逻辑（会创建 actor/critic 优化器）
        super()._setup_model()
        # 准备额外的 critic rollout buffer
        self.critic_rollout_buffer: Optional[RolloutBuffer] = None
        if self.enable_critic_warmup:
            self.critic_rollout_buffer = self.rollout_buffer_class(
                self.critic_rollout_steps,
                self.observation_space,  # type: ignore[arg-type]
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
                **self.rollout_buffer_kwargs,
            )

    def _train_critic_only(self, rollout_buffer: RolloutBuffer) -> None:
        """
        只更新 critic（包括可能的 critic 专用特征提取器），保持 actor 不变。
        """
        if not self.enable_critic_warmup or not self.separate_optimizers:
            return
        assert self.critic_optimizer is not None and self._critic_params is not None

        # 单独调度 critic lr
        self._update_learning_rate([self.critic_optimizer])
        self.policy.set_training_mode(True)

        value_losses = []
        critic_grad_norms = []
        max_batch = rollout_buffer.buffer_size * rollout_buffer.n_envs
        critic_batch_size = min(self.critic_warmup_batch_size, max_batch)

        for _ in range(self.critic_warmup_epochs):
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
            else:
                clip_range_vf = None

            for rollout_data in rollout_buffer.get(critic_batch_size):
                values = self.policy.predict_values(rollout_data.observations).flatten()
                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                critic_grad_norms.append(
                    th.nn.utils.clip_grad_norm_(self._critic_params, self.max_grad_norm).cpu().numpy()
                )
                self.critic_optimizer.step()

                value_losses.append(value_loss.item())
                self._n_critic_updates += 1

        if len(value_losses) > 0:
            self.logger.record("train/critic_only_value_loss", np.mean(value_losses))
        if len(critic_grad_norms) > 0:
            self.logger.record("train/critic_only_grad_norm", np.mean(critic_grad_norms))
        self.logger.record("train/n_critic_updates", self._n_critic_updates, exclude="tensorboard")

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPOCriticWarmup",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            # 阶段1：可选的 critic 预训练，大 rollout，actor 不更新
            if self.enable_critic_warmup:
                assert self.critic_rollout_buffer is not None
                # 记录 warmup 前的 global step，用完再还原，确保 critic rollout 不计入全局步数/进度
                timesteps_before_critic = self.num_timesteps
                cont = self.collect_rollouts(
                    self.env, callback, self.critic_rollout_buffer, n_rollout_steps=self.critic_rollout_steps
                )
                if not cont:
                    break
                # 计入进度并更新 critic
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
                self._train_critic_only(self.critic_rollout_buffer)
                # 从全局计数中移除 critic rollout 的步数，使其不影响 progress/日志
                critic_timesteps = self.num_timesteps - timesteps_before_critic
                self.num_timesteps = timesteps_before_critic
                # 也同步回 callback 内部的计时（用于 early stop 等）
                callback.num_timesteps = self.num_timesteps

            # 阶段2：标准 actor+critic 更新，rollout 步数不变
            cont = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
            if not cont:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train()

        callback.on_training_end()
        return self

