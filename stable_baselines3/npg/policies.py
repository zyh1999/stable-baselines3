"""
Policy aliases for NPG.

We provide a custom ActorCriticPolicy variant for continuous actions that can optionally
parameterize a state-dependent log standard deviation (detach-style: output is split into mu/log_std).
This behavior is enabled via ``policy_kwargs=dict(state_dependent_std=True)``.
"""

from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import DiagGaussianDistribution, Distribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.npg.popart import PopArt


class NPGActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy with an optional state-dependent log_std parameterization for Box actions.

    When ``state_dependent_std=True`` and action space is continuous (Box),
    we create a single action head that outputs ``2 * action_dim`` and split it into
    ``mu`` and ``log_std`` (detach style).
    """

    def __init__(self, *args, state_dependent_std: bool = False, use_popart: bool = False, **kwargs):
        self.state_dependent_std = state_dependent_std
        self.use_popart = use_popart
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:  # type: ignore[override]
        # Build feature extractor + mlp_extractor first (same as SB3)
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Continuous actions (Gaussian)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.state_dependent_std:
                assert isinstance(self.action_space, spaces.Box)
                action_dim = int(self.action_space.shape[0])
                self.action_net = nn.Linear(latent_dim_pi, 2 * action_dim)
                # no global log_std parameter in this mode
                self.log_std = None  # type: ignore[assignment]
            else:
                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            # gSDE path unchanged
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        else:
            # Discrete (or other) distributions path unchanged
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)  # type: ignore[assignment]

        # Value head
        if self.use_popart:
            self.value_net = PopArt(self.mlp_extractor.latent_dim_vf, 1)
        else:
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Init weights and optimizer (reuse SB3 implementation)
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
                # keep SB3 behavior consistent
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                if isinstance(module, PopArt):
                    # detach-style PopArt init
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

    def predict_values(self, obs: th.Tensor) -> th.Tensor:  # type: ignore[override]
        values = super().predict_values(obs)
        if self.use_popart and isinstance(self.value_net, PopArt):
            return self.value_net.unnormalize(values)
        return values


# Aliases
MlpPolicy = NPGActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy

