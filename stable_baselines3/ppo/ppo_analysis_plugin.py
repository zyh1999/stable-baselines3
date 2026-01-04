import torch as th

class PPOAnalysisPlugin:
    """
    A plugin for analyzing PPO gradient dynamics, specifically focusing on
    the impact of advantage mean normalization on gradient signs and clipping behavior.
    """

    @staticmethod
    def compute_grad_norm_for_mask(
        mask: th.Tensor,
        target_tensor: th.Tensor,
        parameters,
        *,
        denom: str = "batch",
    ) -> float:
        """
        Compute the gradient norm for a subset of samples defined by a mask.
        
        Args:
            mask: Boolean mask [batch_size]
            target_tensor: The loss tensor (unreduced) [batch_size]
            parameters: Iterable of model parameters to differentiate with respect to
            denom:
                - "batch": use (1/N) * sum_{i in S} loss_i  (sum-style contribution wrt the full-batch mean loss)
                - "mask":  use (1/|S|) * sum_{i in S} loss_i (per-sample mean within the subset)
            
        Returns:
            Gradient L2 norm (scalar)
        """
        # Avoid tensor->bool ambiguity across torch versions
        if not mask.any().item():
            return 0.0
        
        batch_n = int(target_tensor.numel())
        if denom == "batch":
            div = float(batch_n)
        elif denom == "mask":
            div = float(int(mask.sum().item()))
        else:
            raise ValueError(f"Unknown denom={denom}, expected 'batch' or 'mask'")

        # Sum of loss for masked samples with requested normalization
        masked_loss = (target_tensor * mask.float()).sum() / max(div, 1.0)
        
        # Compute gradients
        # Note: target_tensor must be part of the computation graph
        grads = th.autograd.grad(
            masked_loss, 
            parameters, 
            retain_graph=True, 
            allow_unused=True
        )
        
        # Compute L2 norm
        total_norm = 0.0
        for g in grads:
            if g is not None:
                total_norm += g.detach().data.norm(2).item() ** 2
        
        return total_norm ** 0.5

    @staticmethod
    def _get_actor_params(ppo_instance):
        """
        Return a stable list of actor parameters for analysis.

        - If separate_optimizers=True and _actor_params exists, use it (fast path).
        - Otherwise, derive an actor-param subset that mirrors PPO._setup_model() logic:
          policy_net + action_net (+ log_std) + (shared feature extractor if applicable).
        """
        # Fast path (most common for your use-case)
        if getattr(ppo_instance, "separate_optimizers", False) and getattr(ppo_instance, "_actor_params", None) is not None:
            return list(ppo_instance._actor_params)

        policy = ppo_instance.policy
        actor_params = []

        def _extend_unique(dst, params_iter):
            seen = {id(p) for p in dst}
            for p in params_iter:
                if id(p) not in seen:
                    dst.append(p)
                    seen.add(id(p))

        # Actor-specific modules
        _extend_unique(actor_params, policy.mlp_extractor.policy_net.parameters())
        _extend_unique(actor_params, policy.action_net.parameters())
        if hasattr(policy, "log_std") and isinstance(policy.log_std, th.nn.Parameter):
            actor_params.append(policy.log_std)

        # Feature extractor: shared or separate
        if getattr(policy, "share_features_extractor", True):
            _extend_unique(actor_params, policy.features_extractor.parameters())
        else:
            _extend_unique(actor_params, policy.pi_features_extractor.parameters())

        return actor_params

    @staticmethod
    def is_ppo_clipped(adv: th.Tensor, r: th.Tensor, eps: float) -> th.Tensor:
        """
        Check if PPO objective would be clipped for given advantage and ratio.
        
        PPO Clip condition: min(adv*r, adv*clip(r)) selects the clipped term.
        This happens when the clipped objective is SMALLER than the raw objective.
        (Since PPO maximizes the objective).
        
        When this returns True, it means the gradient w.r.t theta is effectively masked (zeroed)
        or saturated at the clip boundary.
        """
        r_clipped = th.clamp(r, 1 - eps, 1 + eps)
        obj_unclipped = adv * r
        obj_clipped = adv * r_clipped
        
        # PPO objective is min(obj_unclipped, obj_clipped).
        # If obj_clipped < obj_unclipped, then the min operator selects the clipped term.
        # This implies the constraint is active (Clip Effective).
        return obj_clipped < obj_unclipped

    @staticmethod
    def run_analysis(
        ppo_instance,
        advantages_raw: th.Tensor,
        advantages_meaned: th.Tensor,
        ratio: th.Tensor,
        clip_range: float,
        ppo_surrogate_for_actor: th.Tensor
    ) -> None:
        """
        Main entry point to run the analysis and log to wandb.

        NOTE:
        - We analyze ONLY the policy-gradient part (pg loss) via `ppo_surrogate_for_actor`.
        - We ALWAYS ignore entropy loss, regardless of ent_coef.
        """
        if not ppo_instance.normalize_advantage_mean:
            return

        # 1. Identify Sign Flips
        sign_flip_mask = (th.sign(advantages_raw) != th.sign(advantages_meaned)) & (advantages_raw != 0)
        
        # 2. Identify Clip Status Shifts
        # mask_raw: Would it be clipped if we used raw advantage?
        # mask_meaned: Would it be clipped if we used meaned advantage?
        mask_clipped_by_raw = PPOAnalysisPlugin.is_ppo_clipped(advantages_raw, ratio.detach(), clip_range)
        mask_clipped_by_meaned = PPOAnalysisPlugin.is_ppo_clipped(advantages_meaned, ratio.detach(), clip_range)
        
        # Case A: Unmask -> Mask (Clip Added)
        mask_clip_added = (~mask_clipped_by_raw) & mask_clipped_by_meaned
        
        # Case B: Mask -> Unmask (Clip Released)
        mask_clip_released = mask_clipped_by_raw & (~mask_clipped_by_meaned)
        
        # 3. Compute Gradient Norms (PG loss only; entropy ignored)
        # We minimize loss, so target is -objective. Norm is sign-invariant though.
        loss_tensor = -ppo_surrogate_for_actor
        
        # Always use a materialized list, to avoid generator exhaustion
        params = PPOAnalysisPlugin._get_actor_params(ppo_instance)
        
        all_mask = th.ones_like(loss_tensor, dtype=th.bool)

        # (1) Total Gradient Norm (sum-style contribution wrt full-batch mean loss)
        norm_total_sum = PPOAnalysisPlugin.compute_grad_norm_for_mask(
            all_mask,
            loss_tensor,
            params,
            denom="batch",
        )
        # (2) Total Gradient Norm (per-sample mean within "all", equivalent to denom="batch")
        norm_total_mean = norm_total_sum
        
        if norm_total_sum > 1e-8:
            # -------- Sign Flip --------
            norm_flip_sum = PPOAnalysisPlugin.compute_grad_norm_for_mask(sign_flip_mask, loss_tensor, params, denom="batch")
            ppo_instance.logger.record("analysis/grad_norm_ratio_sign_flip_sum", norm_flip_sum / norm_total_sum)

            norm_flip_mean = PPOAnalysisPlugin.compute_grad_norm_for_mask(sign_flip_mask, loss_tensor, params, denom="mask")
            ppo_instance.logger.record("analysis/grad_norm_ratio_sign_flip_mean", norm_flip_mean / max(norm_total_mean, 1e-12))

            # -------- Clip Added (Unmask -> Mask) --------
            norm_clip_added_sum = PPOAnalysisPlugin.compute_grad_norm_for_mask(mask_clip_added, loss_tensor, params, denom="batch")
            ppo_instance.logger.record("analysis/grad_norm_ratio_clip_added_sum", norm_clip_added_sum / norm_total_sum)

            norm_clip_added_mean = PPOAnalysisPlugin.compute_grad_norm_for_mask(mask_clip_added, loss_tensor, params, denom="mask")
            ppo_instance.logger.record("analysis/grad_norm_ratio_clip_added_mean", norm_clip_added_mean / max(norm_total_mean, 1e-12))

            # -------- Clip Released (Mask -> Unmask) --------
            norm_clip_released_sum = PPOAnalysisPlugin.compute_grad_norm_for_mask(mask_clip_released, loss_tensor, params, denom="batch")
            ppo_instance.logger.record("analysis/grad_norm_ratio_clip_released_sum", norm_clip_released_sum / norm_total_sum)

            norm_clip_released_mean = PPOAnalysisPlugin.compute_grad_norm_for_mask(mask_clip_released, loss_tensor, params, denom="mask")
            ppo_instance.logger.record("analysis/grad_norm_ratio_clip_released_mean", norm_clip_released_mean / max(norm_total_mean, 1e-12))
        
        # 4. Record Sample Counts (ratios)
        # batch_n = loss_tensor.numel()
        ppo_instance.logger.record("analysis/sample_ratio_sign_flip", sign_flip_mask.float().mean().item())
        ppo_instance.logger.record("analysis/sample_ratio_clip_added", mask_clip_added.float().mean().item())
        ppo_instance.logger.record("analysis/sample_ratio_clip_released", mask_clip_released.float().mean().item())

