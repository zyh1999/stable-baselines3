import copy
from typing import Any, ClassVar, Optional, TypeVar, Union, Tuple, Dict

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal, Independent
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import polyak_update, get_parameters_by_name
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor, MlpPolicy, CnnPolicy, MultiInputPolicy

SelfMPO = TypeVar("SelfMPO", bound="MPO")

# MPO Constants
_MPO_FLOAT_EPSILON = 1e-8
_MIN_LOG_TEMPERATURE = -18.0
_MIN_LOG_ALPHA = -18.0


class MPOActor(Actor):
    """
    Actor network (policy) for MPO.
    Overrides SAC Actor to use DiagGaussianDistribution (unbounded) instead of SquashedDiagGaussianDistribution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override the distribution to be non-squashed (Unbounded Gaussian)
        # We rely on MPO's action penalization to keep actions within bounds soft-ly.
        if not self.use_sde:
            action_dim = get_action_dim(self.action_space)
            self.action_dist = DiagGaussianDistribution(action_dim)


class MPOPolicy(SACPolicy):
    """
    Policy class (with both actor and critic) for MPO.
    Overrides SAC Policy to use MPOActor.
    """
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MPOActor(**actor_kwargs).to(self.device)


class MPO(OffPolicyAlgorithm):
    """
    Maximum a Posteriori Policy Optimization (MPO) - Off-Policy Version.

    Paper: https://arxiv.org/abs/1806.06920
    Based on the implementation from DeepMind Acme:
    https://github.com/google-deepmind/acme/blob/master/acme/jax/losses/mpo.py

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps.
    :param gradient_steps: How many gradient steps to do after each rollout
    :param action_noise: the action noise type (None by default)
    :param replay_buffer_class: Replay buffer class to use
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
    :param epsilon: KL constraint on the non-parametric auxiliary policy (temperature dual).
    :param epsilon_mean: KL constraint on the mean of the Gaussian policy (alpha_mean dual).
    :param epsilon_stddev: KL constraint on the stddev of the Gaussian policy (alpha_stddev dual).
    :param init_log_temperature: Initial value for log temperature.
    :param init_log_alpha_mean: Initial value for log alpha mean.
    :param init_log_alpha_stddev: Initial value for log alpha stddev.
    :param per_dim_constraining: Whether to enforce KL constraint on each dimension independently.
    :param action_penalization: Whether to penalize out-of-bound actions (MO-MPO).
    :param epsilon_penalty: KL constraint on the probability of violating the action constraint.
    :param num_sample_actions: Number of actions to sample for MPO expectation (N in paper).
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling during warm up
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MPOPolicy,
        "CnnPolicy": CnnPolicy, # Note: CnnPolicy would need a similar override if used with images
        "MultiInputPolicy": MultiInputPolicy, # Same here
    }
    
    policy: MPOPolicy
    actor: MPOActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic
    actor_target: MPOActor

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        epsilon: float = 1e-1,
        epsilon_mean: float = 1e-3,
        epsilon_stddev: float = 1e-5,
        init_log_temperature: float = 1.0,
        init_log_alpha_mean: float = 1.0,
        init_log_alpha_stddev: float = 1.0,
        per_dim_constraining: bool = True,
        action_penalization: bool = True,
        epsilon_penalty: float = 1e-3,
        num_sample_actions: int = 20,
        # Dual learning rate (typically higher than actor/critic LR)
        dual_learning_rate: float = 1e-2,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        # MPO Hyperparameters
        self.epsilon = float(epsilon)
        self.epsilon_mean = float(epsilon_mean)
        self.epsilon_stddev = float(epsilon_stddev)
        self.init_log_temperature = float(init_log_temperature)
        self.init_log_alpha_mean = float(init_log_alpha_mean)
        self.init_log_alpha_stddev = float(init_log_alpha_stddev)
        self.per_dim_constraining = per_dim_constraining
        self.action_penalization = action_penalization
        self.epsilon_penalty = float(epsilon_penalty)
        self.num_sample_actions = num_sample_actions
        self.dual_learning_rate = dual_learning_rate

        # Dual variables containers
        self.log_temperature: Optional[th.Tensor] = None
        self.log_alpha_mean: Optional[th.Tensor] = None
        self.log_alpha_stddev: Optional[th.Tensor] = None
        self.log_penalty_temperature: Optional[th.Tensor] = None
        
        # Dual optimizers
        self.dual_optimizer: Optional[th.optim.Optimizer] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        
        # Setup Target Actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_target.to(self.device)
        self.actor_target.set_training_mode(False)

        # Running mean and running var for critic (if any)
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        
        # Initialize MPO Dual Variables
        action_dim = int(np.prod(self.action_space.shape))
        if self.per_dim_constraining:
            dual_shape = (action_dim,)
            # Scale epsilon by action dimension for per-dimension constraint
            # This ensures epsilon represents the target KL per dimension
            self.epsilon_mean = self.epsilon_mean / action_dim
            self.epsilon_stddev = self.epsilon_stddev / action_dim
        else:
            dual_shape = (1,)
        
        def create_param(value, shape):
            return nn.Parameter(th.full(shape, value, device=self.device, dtype=th.float32))

        self.log_temperature = create_param(self.init_log_temperature, (1,))
        self.log_alpha_mean = create_param(self.init_log_alpha_mean, dual_shape)
        self.log_alpha_stddev = create_param(self.init_log_alpha_stddev, dual_shape)
        
        dual_params = [self.log_temperature, self.log_alpha_mean, self.log_alpha_stddev]

        if self.action_penalization:
            self.log_penalty_temperature = create_param(self.init_log_temperature, (1,))
            dual_params.append(self.log_penalty_temperature)
        else:
            self.log_penalty_temperature = None
        
        # Use separate learning rate for dual variables
        self.dual_optimizer = th.optim.Adam(dual_params, lr=self.dual_learning_rate)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        
        optimizers = [self.actor.optimizer, self.critic.optimizer, self.dual_optimizer]
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        temperature_losses, alpha_losses = [], []
        
        for _ in range(gradient_steps):
            # 1. Sample Replay Buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # --- Critic Update ---
            with th.no_grad():
                next_actions, next_log_prob = self.actor_target.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor Update ---
            N = self.num_sample_actions
            # [Batch, ObsDim] -> [Batch * N, ObsDim]
            obs_expanded = replay_data.observations.repeat_interleave(N, dim=0)
            
            with th.no_grad():
                # Sample actions from Target Policy for Expectation
                sampled_actions, _ = self.actor_target.action_log_prob(obs_expanded)
                # Get Target Distribution parameters
                mean_actions, log_std, kwargs = self.actor_target.get_action_dist_params(replay_data.observations)
                target_dist = self.actor_target.action_dist.proba_distribution(mean_actions, log_std, **kwargs).distribution

            # Evaluate Q-values for sampled actions
            q_values_expanded = th.cat(self.critic(obs_expanded, sampled_actions), dim=1)
            q_values_expanded, _ = th.min(q_values_expanded, dim=1, keepdim=True)
            
            # Reshape [Batch*N, 1] -> [Batch, N]
            q_values = q_values_expanded.reshape(batch_size, N)
            sampled_actions = sampled_actions.reshape(batch_size, N, -1)
            
            # Get Online Distribution
            mean_actions, log_std, kwargs = self.actor.get_action_dist_params(replay_data.observations)
            online_dist = self.actor.action_dist.proba_distribution(mean_actions, log_std, **kwargs).distribution

            # Compute MPO Loss
            mpo_loss, mpo_stats = self._compute_mpo_loss(
                online_dist=online_dist,
                target_dist=target_dist,
                sampled_actions=sampled_actions, # [B, N, D]
                q_values=q_values,               # [B, N]
            )

            actor_losses.append(mpo_stats['loss_policy'])
            temperature_losses.append(mpo_stats['loss_temperature'])
            alpha_losses.append(mpo_stats['loss_alpha'])

            self.actor.optimizer.zero_grad()
            self.dual_optimizer.zero_grad()
            mpo_loss.backward()
            self.actor.optimizer.step()
            self.dual_optimizer.step()
            
            self._clip_dual_params()

            # --- Update Target Networks ---
            if self._n_updates % self.train_freq.frequency == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/temperature_loss", np.mean(temperature_losses))
        self.logger.record("train/alpha_loss", np.mean(alpha_losses))
        self.logger.record("train/kl_mean_rel", mpo_stats['kl_mean_rel'])

    def _clip_dual_params(self):
        with th.no_grad():
            self.log_temperature.clamp_(min=_MIN_LOG_TEMPERATURE)
            self.log_alpha_mean.clamp_(min=_MIN_LOG_ALPHA)
            self.log_alpha_stddev.clamp_(min=_MIN_LOG_ALPHA)
            if self.log_penalty_temperature is not None:
                self.log_penalty_temperature.clamp_(min=_MIN_LOG_TEMPERATURE)

    def _compute_weights_and_temperature_loss(self, values: th.Tensor, epsilon: float, temperature: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Computes normalized importance weights and temperature loss.
        values: [B, N]
        temperature: Scalar
        """
        N = values.shape[1]
        
        # Temper values
        tempered_values = values.detach() / temperature
        
        # Normalized weights: softmax over N samples
        normalized_weights = F.softmax(tempered_values, dim=1).detach() # [B, N]

        # Temperature Loss
        # logsumexp over N samples
        q_logsumexp = th.logsumexp(tempered_values, dim=1) # [B]
        log_num_actions = np.log(N)
        
        # Loss per batch: epsilon + logsumexp - log(N)
        loss_temperature = epsilon + q_logsumexp - log_num_actions
        loss_temperature = (temperature * loss_temperature).mean() # Mean over batch
        
        return normalized_weights, loss_temperature

    def _compute_mpo_loss(
        self,
        online_dist: Normal,
        target_dist: Normal,
        sampled_actions: th.Tensor, # [B, N, D]
        q_values: th.Tensor         # [B, N]
    ) -> Tuple[th.Tensor, dict]:
        
        B, N, D = sampled_actions.shape
        
        # Dual Vars
        temperature = F.softplus(self.log_temperature) + _MPO_FLOAT_EPSILON
        alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = F.softplus(self.log_alpha_stddev) + _MPO_FLOAT_EPSILON

        # --- E-Step: Weights calculation ---
        normalized_weights, loss_temperature = self._compute_weights_and_temperature_loss(
            q_values, self.epsilon, temperature
        )

        # --- Action Penalization (MO-MPO) ---
        if self.action_penalization and self.log_penalty_temperature is not None:
            penalty_temperature = F.softplus(self.log_penalty_temperature) + _MPO_FLOAT_EPSILON
            
            # Compute cost: 0 inside [-1, 1], quadratic outside
            # sampled_actions: [B, N, D]
            diff_out_of_bound = sampled_actions - th.clamp(sampled_actions, -1.0, 1.0)
            cost_out_of_bound = -th.norm(diff_out_of_bound, p=2, dim=-1) # [B, N] (Negative norm because we want to maximize it towards 0)
            
            penalty_weights, loss_penalty_temperature = self._compute_weights_and_temperature_loss(
                cost_out_of_bound, self.epsilon_penalty, penalty_temperature
            )
            
            # Combine weights: sum of weights as per MO-MPO
            normalized_weights = normalized_weights + penalty_weights
            loss_temperature = loss_temperature + loss_penalty_temperature

        # --- M-Step: Policy Loss ---
        # We want to maximize: sum_{b} sum_{n} w_{bn} * log_prob(a_{bn} | s_b)
        
        # Decompose Online Policy (Fixed Std & Fixed Mean)
        # Online Dist has shape [B, D]. We need to broadcast to [B, N, D] to evaluate log_prob of sampled actions.
        online_mean = online_dist.loc.unsqueeze(1).expand(B, N, D)
        online_scale = online_dist.scale.unsqueeze(1).expand(B, N, D)
        target_scale = target_dist.scale.unsqueeze(1).expand(B, N, D)
        target_mean = target_dist.loc.unsqueeze(1).expand(B, N, D)
        
        fixed_std_dist = Normal(loc=online_mean, scale=target_scale)
        fixed_mean_dist = Normal(loc=target_mean, scale=online_scale)
        
        # Weighted Log Prob
        # log_prob shape: [B, N] (summed over D)
        log_prob_mean = fixed_std_dist.log_prob(sampled_actions).sum(dim=-1)
        loss_policy_mean = -th.sum(normalized_weights * log_prob_mean) / B
        
        log_prob_std = fixed_mean_dist.log_prob(sampled_actions).sum(dim=-1)
        loss_policy_stddev = -th.sum(normalized_weights * log_prob_std) / B

        # --- KL Constraint ---
        # KL(Target || Online)
        # Analytical KL between two Gaussians.
        # Shapes: [B, D]
        kl_mean = th.distributions.kl_divergence(target_dist, Normal(online_dist.loc, target_dist.scale))
        kl_stddev = th.distributions.kl_divergence(target_dist, Normal(target_dist.loc, online_dist.scale))
        
        if not self.per_dim_constraining:
             kl_mean = kl_mean.sum(dim=-1, keepdim=True)
             kl_stddev = kl_stddev.sum(dim=-1, keepdim=True)
        
        # Dual Losses
        def compute_dual_loss(kl, alpha, epsilon):
            mean_kl = kl.mean(dim=0) # Average over batch -> [D] or [1]
            loss_kl = th.sum(alpha.detach() * mean_kl)
            loss_alpha = th.sum(alpha * (epsilon - mean_kl.detach()))
            return loss_kl, loss_alpha, mean_kl
            
        loss_kl_mean, loss_alpha_mean, mean_kl_val = compute_dual_loss(kl_mean, alpha_mean, self.epsilon_mean)
        loss_kl_stddev, loss_alpha_stddev, std_kl_val = compute_dual_loss(kl_stddev, alpha_stddev, self.epsilon_stddev)

        # Total Loss
        loss_policy = loss_policy_mean + loss_policy_stddev
        loss_kl_penalty = loss_kl_mean + loss_kl_stddev
        loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature
        
        total_loss = loss_policy + loss_kl_penalty + loss_dual

        stats = {
            "loss_policy": loss_policy.item(),
            "loss_temperature": loss_temperature.item(),
            "loss_alpha": (loss_alpha_mean + loss_alpha_stddev).item(),
            "kl_mean_rel": mean_kl_val.mean().item() / self.epsilon_mean,
        }
        
        return total_loss, stats
