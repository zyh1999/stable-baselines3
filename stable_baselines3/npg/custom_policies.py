"""
Custom policies for NPG.

This module keeps the previously-added custom policy implementation (PopArt value head,
optional state-dependent std for DiagGaussian), but it is NOT used by default.

If you want to use it, pass it explicitly as the `policy` argument:

    from stable_baselines3.npg.custom_policies import NPGActorCriticPolicy
    model = NPG(NPGActorCriticPolicy, env, use_popart=True, ...)
"""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.npg.popart import PopArt


class NPGActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy with:
    - optional state-dependent log_std parameterization for Box actions (detach-style split into mu/log_std)
    - optional PopArt value head
    """

    def __init__(self, *args, state_dependent_std: bool = False, use_popart: bool = False, **kwargs):
        self.state_dependent_std = state_dependent_std
        self.use_popart = use_popart
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:  # type: ignore[override]
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.state_dependent_std:
                assert isinstance(self.action_space, spaces.Box)
                action_dim = int(self.action_space.shape[0])
                self.action_net = nn.Linear(latent_dim_pi, 2 * action_dim)
                self.log_std = None  # type: ignore[assignment]
            else:
                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        else:
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)  # type: ignore[assignment]

        if self.use_popart:
            self.value_net = PopArt(self.mlp_extractor.latent_dim_vf, 1)
        else:
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            import numpy as np
            from functools import partial

            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                if isinstance(module, PopArt):
                    module.reset_parameters()
                else:
                    module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:  # type: ignore[override]
        if self.state_dependent_std and isinstance(self.action_dist, DiagGaussianDistribution):
            out = self.action_net(latent_pi)  # type: ignore[arg-type]
            mean_actions, log_std = out.chunk(2, dim=-1)
            return self.action_dist.proba_distribution(mean_actions, log_std)
        return super()._get_action_dist_from_latent(latent_pi)

    def forward(self, obs: th.Tensor, deterministic: bool = False):  # type: ignore[override]
        """
        Important (PopArt):
        - SB3's ActorCriticPolicy.forward() returns the *raw* value head output.
        - When value_net is PopArt, that raw output is the *normalized* value prediction.
        - Detach-style runners/GAE expect unnormalized values in env-reward scale.
        So we unnormalize here to make rollout buffer values consistent with rewards/returns.
        """
        actions, values, log_prob = super().forward(obs, deterministic=deterministic)
        if self.use_popart and isinstance(self.value_net, PopArt):
            values = self.value_net.unnormalize(values)
        return actions, values, log_prob

    def predict_values(self, obs: th.Tensor) -> th.Tensor:  # type: ignore[override]
        values = super().predict_values(obs)
        if self.use_popart and isinstance(self.value_net, PopArt):
            return self.value_net.unnormalize(values)
        return values


