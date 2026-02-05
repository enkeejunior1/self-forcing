"""
Pi-Flow Training Pipeline for autoregressive video generation with policy-based ODE integration.
"""
from utils.scheduler import SchedulerInterface
from utils.wan_piflow_wrapper import WanPiflowWrapper
from typing import List, Optional, Dict, Any
from functools import partial
import random
import torch
import torch.distributed as dist


class PiFlowTrainingPipeline:
    """
    Training pipeline for Pi-Flow with policy-based ODE integration.
    
    Supports:
    - GMM (Gaussian Mixture Model) policy
    - DX (Direct x0) policy
    - DMD-style (predict + resample) or ODE integration sampling
    """
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: WanPiflowWrapper,
        num_frame_per_block: int = 3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        policy_type: str = '',
        sampling_type: str = 'dmd',  # 'dmd' (predict-and-resample) or 'piflow' (ODE integration)
        integration_nfe: int = 128,  # Number of substeps for ODE integration
        x0_pred_type: str = 'dmd',   # 'dmd' (direct prediction) or 'piflow' (ODE integration to σ≈0)
        **kwargs
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        # Wan-specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        
        # Policy settings
        self.policy_type = policy_type
        self.use_policy = policy_type in ['gmm', 'dmm', 'dx']
        
        # Sampling settings
        self.sampling_type = sampling_type
        self.integration_nfe = integration_nfe
        self.x0_pred_type = x0_pred_type
        self.eps = 1e-4

    # ==================== Sigma/Timestep Conversion ====================
    
    def t_to_sigma(self, raw_t: torch.Tensor) -> torch.Tensor:
        """Convert raw timestep t in [0, 1] to sigma using flow matching shift schedule."""
        shift = self.scheduler.shift
        sigma = shift * raw_t / (1 + (shift - 1) * raw_t)
        return sigma
    
    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to raw timestep t in [0, 1] (inverse of t_to_sigma)."""
        shift = self.scheduler.shift
        raw_t = sigma / (shift - (shift - 1) * sigma)
        return raw_t
    
    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """Convert discrete timestep (0-1000) to sigma."""
        sigmas = self.scheduler.sigmas.to(timestep.device)
        timesteps = self.scheduler.timesteps.to(timestep.device)
        
        if timestep.dim() == 2:
            timestep_flat = timestep.flatten(0, 1)
        else:
            timestep_flat = timestep
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id]
        
        if timestep.dim() == 2:
            return sigma.reshape(timestep.shape)
        return sigma

    # ==================== ODE Integration ====================
    
    def from_s_to_t(
        self,
        x_s: torch.Tensor,
        raw_s: torch.Tensor,
        raw_t: torch.Tensor,
        params: dict,
        policy_type: str
    ) -> torch.Tensor:
        """
        ODE integration from timestep s to t using Euler method.
        
        Args:
            x_s: (B, F, C, H, W) - noisy sample at timestep s
            raw_s: (B,) - raw timestep s (in [0, 1])
            raw_t: (B,) - raw timestep t (in [0, 1], t < s)
            params: dict with policy parameters
            policy_type: 'gmm', 'dmm', or 'dx'
        
        Returns:
            x_t: (B, F, C, H, W) - sample at timestep t
        """
        if (raw_s == raw_t).all():
            return x_s
        
        # Permute to (B, C, F, H, W) for policy
        x_s_reordered = x_s.permute(0, 2, 1, 3, 4)
        
        # Reference sigma for GMM/DMM
        sigma_s_ref = self.t_to_sigma(raw_s)
        
        # Compute number of substeps based on time delta
        delta_t = raw_s - raw_t
        num_substeps = (delta_t * self.integration_nfe).round().long().clamp(min=1)
        max_substeps = num_substeps.max().item()
        substep_size = delta_t / num_substeps.float()
        
        # Create partial function with bound params
        if policy_type == 'dx':
            pi = partial(
                self.generator.pi,
                x_0_grid=params['x_0_grid'],
            )
        else:  # gmm or dmm
            pi = partial(
                self.generator.pi,
                a_s=params['a_s'],
                mu_s=params['mu_s'],
                std_s=params['std_s'],
                x_s=x_s_reordered,
                sigma_s=sigma_s_ref,
            )
        
        # ODE integration loop
        x_current = x_s_reordered
        raw_current = raw_s.clone()
        
        for substep_id in range(max_substeps):
            sigma_current = self.t_to_sigma(raw_current)
            
            # Compute velocity using policy
            v_t = pi(x_t=x_current, sigma_t=sigma_current)
            
            # Euler step: x_{t-} = x_t - v_t * (sigma_t - sigma_{t-})
            raw_next = (raw_current - substep_size).clamp(min=self.eps)
            sigma_next = self.t_to_sigma(raw_next)
            sigma_diff = (sigma_current - sigma_next).view(-1, 1, 1, 1, 1)
            x_next = x_current - v_t * sigma_diff
            
            # Only update samples that haven't reached their target
            active_mask = (num_substeps > substep_id).view(-1, 1, 1, 1, 1)
            x_current = torch.where(active_mask, x_next, x_current)
            raw_current = torch.where(num_substeps > substep_id, raw_next, raw_current)
        
        # Permute back to (B, F, C, H, W)
        return x_current.permute(0, 2, 1, 3, 4)

    # ==================== x0 Prediction ====================
    
    def get_x0(
        self,
        params: dict,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
        policy_type: str,
        x0_pred_type: str = None
    ) -> torch.Tensor:
        """
        Get x0 prediction using either direct prediction or ODE integration.
        
        Args:
            params: dict with policy parameters
            x_t: (B, F, C, H, W) - noisy sample at timestep t
            sigma_t: (B,) or (B, F) - noise level at timestep t
            policy_type: 'gmm', 'dmm', or 'dx'
            x0_pred_type: 'dmd' (direct) or 'piflow' (ODE to σ≈0)
        
        Returns:
            x_0: (B, F, C, H, W) - predicted clean sample
        """
        if x0_pred_type is None:
            x0_pred_type = self.x0_pred_type
        
        if x0_pred_type == 'piflow':
            return self._get_x0_piflow(params, x_t, sigma_t, policy_type)
        else:
            return self._get_x0_direct(params, x_t, sigma_t, policy_type)
    
    def _get_x0_direct(
        self,
        params: dict,
        x_s: torch.Tensor,
        sigma_s: torch.Tensor,
        policy_type: str
    ) -> torch.Tensor:
        """Direct x0 prediction without ODE integration (DMD style)."""
        # Permute to (B, C, F, H, W) for generator
        x_s_reordered = x_s.permute(0, 2, 1, 3, 4)
        
        # Use generator's get_direct_x0 method
        x_0 = self.generator.get_direct_x0(
            params=params, 
            x_s=x_s_reordered, 
            sigma_s=sigma_s if sigma_s.dim() == 1 else sigma_s[:, 0],
            policy_type=policy_type
        )
        
        # Permute back to (B, F, C, H, W)
        return x_0.permute(0, 2, 1, 3, 4)
    
    def _get_x0_piflow(
        self,
        params: dict,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
        policy_type: str
    ) -> torch.Tensor:
        """Get x0 via ODE integration from current timestep t to σ≈0."""
        sigma_t_scalar = sigma_t if sigma_t.dim() == 1 else sigma_t[:, 0]
        raw_t = self.sigma_to_t(sigma_t_scalar)
        raw_0 = torch.full_like(raw_t, self.eps)
        x_0 = self.from_s_to_t(x_t, raw_t, raw_0, params, policy_type)
        return x_0

    # ==================== Random Index Generation ====================
    
    def generate_and_sync_list(
        self,3
        num_blocks: int,
        num_denoising_steps: int,
        device: torch.device
    ) -> List[int]:
        """Generate synchronized random exit indices across distributed ranks."""
        is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            if self.last_step_only:
                indices_list = [num_denoising_steps - 1] * num_blocks
            else:
                indices_list = [random.randint(0, num_denoising_steps - 1) for _ in range(num_blocks)]
            indices = torch.tensor(indices_list, dtype=torch.long, device=device)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if is_distributed:
            dist.broadcast(indices, src=0)
        
        return indices.tolist()

    # ==================== Main Inference Loop ====================
    
    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        return_reproduce_state: bool = False,
        **conditional_dict
    ) -> torch.Tensor:
        """
        Run autoregressive inference with trajectory collection for pi-flow training.
        
        Args:
            noise: (B, F, C, H, W) - input noise tensor
            initial_latent: Optional initial latent for i2v
            return_sim_step: Whether to return simulation step info
            return_reproduce_state: Whether to return reproducibility state
            **conditional_dict: Text embeddings and other conditions
        
        Returns:
            Tuple of (output, timestep_from, timestep_to, piflow_data, [reproduce_state])
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # Calculate number of blocks
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Initialize caches
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)

        # Cache context feature for i2v
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # For reproducibility: record all randomness
        ar_noise_bank = []
        ar_noise_meta = []

        # Setup denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # Storage for piflow data
        piflow_x_t_list = []
        piflow_params_list = []
        piflow_timestep_list = []

        # Block loop
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames
            ]

            # Denoising steps within block
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64
                ) * current_timestep

                if not exit_flag:
                    # Non-exit step: denoise without gradient
                    with torch.no_grad():
                        if self.use_policy:
                            params, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                policy_mode=self.policy_type
                            )
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        
                        next_timestep = self.denoising_step_list[index + 1]
                        
                        if self.sampling_type == 'piflow' and self.use_policy:
                            # ODE integration
                            sigma_t = self.timestep_to_sigma(timestep)
                            sigma_s = self.timestep_to_sigma(
                                next_timestep * torch.ones_like(timestep)
                            )
                            raw_t = self.sigma_to_t(sigma_t[:, 0])
                            raw_s = self.sigma_to_t(sigma_s[:, 0])
                            noisy_input = self.from_s_to_t(
                                noisy_input, raw_t, raw_s, params, self.policy_type
                            )
                        else:
                            # DMD-style: predict x0, add noise
                            den_flat = denoised_pred.flatten(0, 1)
                            noise_flat = torch.randn_like(den_flat)
                            if return_reproduce_state:
                                ar_noise_bank.append(noise_flat.detach().cpu())
                                ar_noise_meta.append(dict(
                                    kind="resample",
                                    block_index=int(block_index),
                                    step_index=int(index),
                                    next_timestep=float(next_timestep),
                                    frames=int(current_num_frames),
                                ))
                            noisy_input = self.scheduler.add_noise(
                                den_flat,
                                noise_flat,
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Exit step: collect piflow data
                    should_store = current_start_frame >= start_gradient_frame_index
                    
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            if self.use_policy:
                                params, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length,
                                    policy_mode=self.policy_type
                                )
                            else:
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=conditional_dict,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                    else:
                        # With gradients for last 21 frames
                        if self.use_policy:
                            params, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                policy_mode=self.policy_type
                            )
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    
                    # Store piflow data for gradient frames
                    if should_store:
                        piflow_x_t_list.append(noisy_input.clone())
                        piflow_timestep_list.append(timestep.clone())
                        if self.use_policy:
                            if self.policy_type == 'dx':
                                piflow_params_list.append({'x_0_grid': params['x_0_grid']})
                            else:
                                piflow_params_list.append({
                                    'mu_s': params['mu_s'],
                                    'a_s': params['a_s'],
                                    'std_s': params['std_s'],
                                })
                    
                    # Get x0 using configured method
                    if self.use_policy:
                        sigma_t = self.timestep_to_sigma(timestep)
                        denoised_pred = self.get_x0(
                            params=params,
                            x_t=noisy_input,
                            sigma_t=sigma_t,
                            policy_type=self.policy_type,
                            x0_pred_type=self.x0_pred_type
                        )
                    break

            # Record output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Update cache with context noise
            context_timestep = torch.ones_like(timestep) * self.context_noise
            den_flat = denoised_pred.flatten(0, 1)
            noise_flat = torch.randn_like(den_flat)
            if return_reproduce_state:
                ar_noise_bank.append(noise_flat.detach().cpu())
                ar_noise_meta.append(dict(
                    kind="context",
                    block_index=int(block_index),
                    context_noise=int(self.context_noise),
                    frames=int(current_num_frames),
                ))
            denoised_pred = self.scheduler.add_noise(
                den_flat,
                noise_flat,
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            current_start_frame += current_num_frames

        # Calculate denoised timesteps
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        # Build piflow_data dict
        piflow_data = None
        if len(piflow_x_t_list) > 0:
            piflow_x_t = torch.cat(piflow_x_t_list, dim=1)
            piflow_timestep = torch.cat(piflow_timestep_list, dim=1)
            
            piflow_data = {
                'x_t': piflow_x_t,
                'timestep': piflow_timestep,
                'exit_step': exit_flags[0],
                'denoised_timestep_from': denoised_timestep_from,
                'denoised_timestep_to': denoised_timestep_to,
            }
            
            if self.use_policy and len(piflow_params_list) > 0:
                if self.policy_type == 'dx':
                    piflow_params = {
                        'x_0_grid': torch.cat([p['x_0_grid'] for p in piflow_params_list], dim=3),
                    }
                else:
                    piflow_params = {
                        'mu_s': torch.cat([p['mu_s'] for p in piflow_params_list], dim=3),
                        'a_s': torch.cat([p['a_s'] for p in piflow_params_list], dim=3),
                        'std_s': piflow_params_list[0]['std_s'],
                    }
                piflow_data['params'] = piflow_params
                piflow_data['policy_type'] = self.policy_type

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1, piflow_data

        if return_reproduce_state:
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                piflow_data,
                dict(
                    ar_exit_flags=exit_flags,
                    ar_noise_bank=ar_noise_bank,
                    ar_noise_meta=ar_noise_meta,
                    ar_denoising_step_list=[int(x) for x in self.denoising_step_list],
                    ar_same_step_across_blocks=bool(self.same_step_across_blocks),
                    ar_last_step_only=bool(self.last_step_only),
                    ar_num_frame_per_block=int(self.num_frame_per_block),
                    ar_context_noise=int(self.context_noise),
                ),
            )

        return output, denoised_timestep_from, denoised_timestep_to, piflow_data

    # ==================== Cache Initialization ====================
    
    def _initialize_kv_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize per-GPU KV cache for the Wan model."""
        kv_cache1 = []
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        """Initialize per-GPU cross-attention cache for the Wan model."""
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
