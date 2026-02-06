"""
Pi-Flow model for distillation training with policy-based ODE integration.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from einops import rearrange
from typing import Optional, Tuple

from .base import PiFlowSelfForcingModel
from utils.loss import get_denoising_loss


class Piflow(PiFlowSelfForcingModel):
    """
    Pi-Flow model for training with policy-based ODE integration.
    
    Supports:
    - GMM policy (Gaussian Mixture Model)
    - DX policy (Direct x0 prediction with grid interpolation)
    - L2 loss (velocity matching)
    - Optional DMD loss for distribution matching
    """
    def __init__(self, args, device):
        super().__init__(args, device)
        self.num_training_frames = args.num_training_frames
        self.num_frame_per_block = args.num_frame_per_block
        self.independent_first_frame = args.independent_first_frame
        
        # Pi-flow specific settings
        self.policy_type = getattr(args, 'policy_type', 'gmm')
        self.integration_nfe = getattr(args, 'integration_nfe', 128)
        
        # Loss scales (0.0 = disabled)
        self.piflow_loss_scale = getattr(args, 'piflow_loss_scale', 1.0)
        self.dmd_loss_scale = getattr(args, 'dmd_loss_scale', 0.0)
        
        # Initialize policy heads
        self._init_policy_heads(args)
        
    def _init_policy_heads(self, args):
        """Initialize policy heads based on policy type."""
        policy_head_cfg = getattr(args, 'policy_head_cfg', {})
        
        if self.policy_type == 'gmm':
            self.generator.init_gmm_heads(
                num_gaussians=policy_head_cfg.get('K', 8),
                latent_c=policy_head_cfg.get('latent_c', 16),
                patch_size=policy_head_cfg.get('patch_size', 2),
                gm_num_logstd_layers=policy_head_cfg.get('gm_num_logstd_layers', 2),
                logstd_inner_dim=policy_head_cfg.get('logstd_inner_dim', 256),
                init_logstd=policy_head_cfg.get('init_logstd', np.log(0.05)),
            )
        elif self.policy_type == 'dx':
            self.generator.init_dx_heads(
                num_grid_points=policy_head_cfg.get('K', 8),
                latent_c=policy_head_cfg.get('latent_c', 16),
                patch_size=policy_head_cfg.get('patch_size', 2),
            )
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def initialize_policy_head_weights(self, verbose=True):
        """Initialize policy head weights from pretrained generator weights."""
        if self.policy_type == 'dx':
            self.generator.initialize_dx_head_weights()
            if verbose:
                print(f"[Piflow] Initialized DX policy head weights")
        elif self.policy_type == 'gmm':
            self.generator.initialize_gmm_head_weights()
            if verbose:
                print(f"[Piflow] Initialized GMM policy head weights")
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
    
    def generator_loss(
        self,
        real_images_or_videos: torch.Tensor,
        conditional_dict: dict,
        piflow_loss_scale: float = None,
        dmd_loss_scale: float = None,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator loss for pi-flow training.
        
        Args:
            real_images_or_videos: Real video tensor [B, T, C, H, W]
            conditional_dict: Dict with text embeddings and other conditions
            piflow_loss_scale: Scale for pi-flow loss (None = use self.piflow_loss_scale)
            dmd_loss_scale: Scale for DMD loss (None = use self.dmd_loss_scale)
            
        Returns:
            loss: Total loss tensor
            loss_dict: Dict with individual loss values for logging
        """
        # Use instance defaults if not provided
        if piflow_loss_scale is None:
            piflow_loss_scale = self.piflow_loss_scale
        if dmd_loss_scale is None:
            dmd_loss_scale = self.dmd_loss_scale
        
        B, T, C, H, W = real_images_or_videos.shape
        image_or_video_shape = [B, T, C, H, W]
        
        # Run generator with backward simulation
        pred_video, gradient_mask, timestep_from, timestep_to, piflow_data = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
        )
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute pi-flow loss if enabled and we have piflow_data
        if piflow_loss_scale > 0 and piflow_data and len(piflow_data) > 0:
            piflow_loss, piflow_loss_dict = self._compute_piflow_loss(
                piflow_data=piflow_data,
                conditional_dict=conditional_dict,
            )
            total_loss = total_loss + piflow_loss_scale * piflow_loss
            loss_dict.update(piflow_loss_dict)
        
        # Compute DMD loss if enabled
        if dmd_loss_scale > 0:
            dmd_loss, dmd_loss_dict = self._compute_dmd_loss(
                pred_video=pred_video,
                gradient_mask=gradient_mask,
                timestep_from=timestep_from,
                timestep_to=timestep_to,
                conditional_dict=conditional_dict,
            )
            total_loss = total_loss + dmd_loss_scale * dmd_loss
            loss_dict.update(dmd_loss_dict)
        
        return total_loss, loss_dict
    
    def _compute_piflow_loss(
        self,
        piflow_data: list,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute pi-flow loss from collected trajectory data.
        
        For each sampled (x_t, timestep_t, policy_params), we:
        1. Use ODE integration with policy to get x_s at timestep_s (no grad)
        2. Compute policy velocity at x_s (with grad)
        3. Compute teacher velocity at x_s (no grad)
        4. L2 loss: ||v_policy - v_teacher||^2
        """
        if len(piflow_data) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), {}
        
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        num_samples = 0
        
        for data in piflow_data:
            x_t = data['x_t']  # [B, T, C, H, W]
            policy_params = data['policy_params']
            timestep_t = data['timestep_t']
            timestep_s = data['timestep_s']
            
            # ODE integration from t to s using policy (no gradient through integration)
            with torch.no_grad():
                x_s = self._ode_integrate(
                    x_t=x_t,
                    timestep_t=timestep_t,
                    timestep_s=timestep_s,
                    policy_params=policy_params,
                )
            
            # Compute policy velocity at x_s (with gradient through policy)
            sigma_s = self._timestep_to_sigma(timestep_s)
            x_s_reordered = rearrange(x_s, 'b t c h w -> b c t h w')
            policy_velocity = self._compute_policy_velocity(
                x_t=x_s_reordered,
                sigma_t=sigma_s,
                policy_params=policy_params,
            )
            
            # Get teacher velocity at x_s (no gradient)
            with torch.no_grad():
                teacher_velocity = self._get_teacher_velocity(
                    x_s, 
                    timestep_s, 
                    conditional_dict
                )
            
            # L2 velocity matching loss
            loss = ((policy_velocity - teacher_velocity) ** 2).mean()
            
            total_loss = total_loss + loss
            num_samples += 1
        
        if num_samples > 0:
            total_loss = total_loss / num_samples
            
        return total_loss, {'piflow_loss': total_loss.detach().item()}
    
    def _timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert discrete timestep (0-1000) to sigma."""
        sigmas = self.scheduler.sigmas.to(timestep.device)
        timesteps = self.scheduler.timesteps.to(timestep.device)
        
        if timestep.dim() == 2:
            timestep_flat = timestep[:, 0]  # Use first frame's timestep
        else:
            timestep_flat = timestep
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id]
        return sigma
    
    def _t_to_sigma(self, raw_t: torch.Tensor) -> torch.Tensor:
        """Convert raw timestep t in [0, 1] to sigma using flow matching shift schedule."""
        shift = self.scheduler.shift
        sigma = shift * raw_t / (1 + (shift - 1) * raw_t)
        return sigma
    
    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to raw timestep t in [0, 1]."""
        shift = self.scheduler.shift
        raw_t = sigma / (shift - (shift - 1) * sigma)
        return raw_t
    
    def _compute_policy_velocity(
        self,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
        policy_params: dict,
    ) -> torch.Tensor:
        """
        Compute velocity using policy with gradient.
        
        Args:
            x_t: (B, C, T, H, W) - current state
            sigma_t: (B,) - current sigma
            policy_params: dict with policy parameters
        
        Returns:
            velocity: (B, C, T, H, W) - policy velocity
        """
        from functools import partial
        
        if self.policy_type == 'dx':
            pi = partial(
                self.generator.pi,
                x_0_grid=policy_params['x_0_grid'],
            )
        else:  # gmm
            pi = partial(
                self.generator.pi,
                a_s=policy_params['a_s'],
                mu_s=policy_params['mu_s'],
                std_s=policy_params['std_s'],
                x_s=policy_params.get('x_s', x_t),
                sigma_s=policy_params.get('sigma_s', sigma_t),
            )
        
        velocity = pi(x_t=x_t, sigma_t=sigma_t)
        return velocity
    
    def _ode_integrate(
        self,
        x_t: torch.Tensor,
        timestep_t: torch.Tensor,
        timestep_s: torch.Tensor,
        policy_params: dict,
    ) -> torch.Tensor:
        """
        ODE integration from timestep_t to timestep_s using policy (Euler method).
        
        Args:
            x_t: (B, T, C, H, W) - state at timestep t
            timestep_t: (B,) or (B, T) - starting timestep (higher noise)
            timestep_s: (B,) or (B, T) - target timestep (lower noise)
            policy_params: dict with policy parameters
        
        Returns:
            x_s: (B, T, C, H, W) - integrated state at timestep_s
        """
        from functools import partial
        eps = 1e-4
        
        # Get sigma values
        sigma_t = self._timestep_to_sigma(timestep_t)
        sigma_s = self._timestep_to_sigma(timestep_s)
        
        # Convert to raw t values
        raw_t = self._sigma_to_t(sigma_t)
        raw_s = self._sigma_to_t(sigma_s)
        
        if (raw_t == raw_s).all():
            return x_t
        
        # Permute to (B, C, T, H, W) for policy
        x_current = rearrange(x_t, 'b t c h w -> b c t h w')
        
        # Reference sigma for GMM
        sigma_ref = sigma_t
        
        # Create partial function with bound policy params
        if self.policy_type == 'dx':
            pi = partial(
                self.generator.pi,
                x_0_grid=policy_params['x_0_grid'],
            )
        else:  # gmm
            pi = partial(
                self.generator.pi,
                a_s=policy_params['a_s'],
                mu_s=policy_params['mu_s'],
                std_s=policy_params['std_s'],
                x_s=x_current.clone(),  # Reference x at starting point
                sigma_s=sigma_ref,
            )
        
        # Compute number of substeps based on time delta
        delta_t = raw_t - raw_s
        num_substeps = (delta_t * self.integration_nfe).round().long().clamp(min=1)
        max_substeps = num_substeps.max().item()
        substep_size = delta_t / num_substeps.float()
        
        # ODE integration loop (Euler method)
        raw_current = raw_t.clone()
        
        for substep_id in range(max_substeps):
            sigma_current = self._t_to_sigma(raw_current)
            
            # Compute velocity using policy
            v_t = pi(x_t=x_current, sigma_t=sigma_current)
            
            # Euler step: x_{next} = x_current - v_t * (sigma_current - sigma_next)
            raw_next = (raw_current - substep_size).clamp(min=eps)
            sigma_next = self._t_to_sigma(raw_next)
            sigma_diff = (sigma_current - sigma_next).view(-1, 1, 1, 1, 1)
            x_next = x_current - v_t * sigma_diff
            
            # Only update samples that haven't reached their target
            active_mask = (num_substeps > substep_id).view(-1, 1, 1, 1, 1)
            x_current = torch.where(active_mask, x_next, x_current)
            raw_current = torch.where(num_substeps > substep_id, raw_next, raw_current)
        
        # Permute back to (B, T, C, H, W)
        return rearrange(x_current, 'b c t h w -> b t c h w')
    
    def _get_teacher_velocity(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        """Get velocity prediction from teacher (real_score) model."""
        # Prepare inputs for teacher model
        B, T, C, H, W = x_t.shape
        
        # Get text embeddings
        encoder_hidden_states = conditional_dict.get('encoder_hidden_states')
        encoder_hidden_states_t5 = conditional_dict.get('encoder_hidden_states_t5')
        
        # Reshape for model input
        x_t_input = rearrange(x_t, 'b t c h w -> b c t h w')
        
        # Get model output
        with torch.no_grad():
            model_output = self.real_score(
                x=x_t_input,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
            )
        
        return model_output
    
    def _compute_dmd_loss(
        self,
        pred_video: torch.Tensor,
        gradient_mask: Optional[torch.Tensor],
        timestep_from: int,
        timestep_to: int,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute DMD loss for distribution matching with segment-aware timestep sampling."""
        # Get text embeddings
        encoder_hidden_states = conditional_dict.get('encoder_hidden_states')
        encoder_hidden_states_t5 = conditional_dict.get('encoder_hidden_states_t5')
        
        B, T, C, H, W = pred_video.shape
        
        # Segment-aware timestep sampling with shift schedule
        # Use segment bounds if available, otherwise fall back to global range
        min_t = max(int(timestep_to), 20) if timestep_to is not None else 20
        max_t = min(int(timestep_from), 980) if timestep_from is not None else 980
        if min_t >= max_t:
            min_t, max_t = 20, 980  # fallback if segment is degenerate
        
        # Convert segment bounds to raw_t (unwarped continuous space)
        raw_t_min = self._sigma_to_t(torch.tensor(min_t / 1000.0, device=self.device))
        raw_t_max = self._sigma_to_t(torch.tensor(max_t / 1000.0, device=self.device))
        
        # Sample uniform raw_t in segment range, then warp through shift schedule
        raw_t = torch.rand(B, device=self.device) * (raw_t_max - raw_t_min) + raw_t_min
        sigma_t = self._t_to_sigma(raw_t)
        timestep = (sigma_t * 1000).clamp(20, 980).long()
        timestep = timestep.unsqueeze(1).expand(B, T)
        
        # Add noise to predicted video
        noise = torch.randn_like(pred_video)
        noisy_video = self.scheduler.add_noise(pred_video, noise, timestep)
        
        # Get real and fake score predictions
        noisy_video_input = rearrange(noisy_video, 'b t c h w -> b c t h w')
        
        with torch.no_grad():
            real_score_pred = self.real_score(
                x=noisy_video_input,
                timestep=timestep[:, 0],
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_t5=encoder_hidden_states_t5,
            )
        
        fake_score_pred = self.fake_score(
            x=noisy_video_input,
            timestep=timestep[:, 0],
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
        )
        
        # Score difference
        score_diff = fake_score_pred - real_score_pred
        
        # Compute loss
        loss = (score_diff ** 2).mean()
        
        return loss, {'dmd_loss': loss.detach().item()}
    
    def critic_loss(
        self,
        real_images_or_videos: torch.Tensor,
        conditional_dict: dict,
        **kwargs
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute critic (fake_score) loss for DMD training.
        
        Only used when dmd_loss_scale > 0.
        """
        B, T, C, H, W = real_images_or_videos.shape
        
        # Get text embeddings
        encoder_hidden_states = conditional_dict.get('encoder_hidden_states')
        encoder_hidden_states_t5 = conditional_dict.get('encoder_hidden_states_t5')
        
        # Sample timestep
        timestep = self._get_timestep(
            min_timestep=20,
            max_timestep=980,
            batch_size=B,
            num_frame=T,
            num_frame_per_block=self.num_frame_per_block,
            uniform_timestep=False,
        )
        
        # Add noise to real video
        noise = torch.randn_like(real_images_or_videos)
        noisy_video = self.scheduler.add_noise(real_images_or_videos, noise, timestep)
        noisy_video_input = rearrange(noisy_video, 'b t c h w -> b c t h w')
        
        # Get fake score prediction
        fake_score_pred = self.fake_score(
            x=noisy_video_input,
            timestep=timestep[:, 0],
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
        )
        
        # Target is the noise
        noise_input = rearrange(noise, 'b t c h w -> b c t h w')
        
        # Compute denoising loss
        loss = self.denoising_loss_func(fake_score_pred, noise_input)
        
        return loss, {'critic_loss': loss.detach().item()}
