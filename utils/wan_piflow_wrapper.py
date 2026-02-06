"""
WanPiflowWrapper and Policy classes for Pi-Flow training.

This module provides:
- GMPolicy: Gaussian Mixture Model policy for velocity computation
- DXPolicy: Direct x0 grid policy for velocity computation  
- WanPiflowWrapper: Wrapper around CausalWanModel with policy heads
"""
import types
import math
import numpy as np
import warnings
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.causal_model import CausalWanModel


# ============================================================
# Policy Classes
# ============================================================

class DXPolicy(nn.Module):
    """
    DX-based policy for computing velocity v_t from K grid point predictions.
    Grid points are uniformly spaced in sigma space from 1 (noisy) to 0 (clean).
    Interpolates between grid points based on current sigma_t.
    """
    def __init__(
        self, 
        eps: float = 1e-4, 
        use_gradient_checkpointing: bool = False,
        use_patch_K: bool = False,
        patch_size: tuple = (1, 2, 2),
    ):
        super().__init__()
        self.eps = eps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_patch_K = use_patch_K
        self.patch_size = patch_size
    
    def forward(
        self, 
        x_t: torch.Tensor, 
        sigma_t: torch.Tensor,
        x_0_grid: torch.Tensor,
        x_s: torch.Tensor = None,
        sigma_s: torch.Tensor = None,
        use_gradient_checkpointing: bool = None,
    ) -> torch.Tensor:
        """
        Compute velocity v_t by interpolating x_0 from grid points.
        
        Args:
            x_t: (B, C, F, H, W) - current noisy sample
            sigma_t: (B,) - current noise level
            x_0_grid: (B, K, C, F, H, W) - x_0 predictions at K grid points
            use_gradient_checkpointing: Optional override for gradient checkpointing
            
        Returns:
            v_t: (B, C, F, H, W) - velocity prediction
        """
        # Allow per-call override of gradient checkpointing
        if use_gradient_checkpointing is None:
            use_gradient_checkpointing = self.use_gradient_checkpointing
        
        if use_gradient_checkpointing and self.training:
            v_t, _ = torch.utils.checkpoint.checkpoint(
                dx_policy_forward,
                x_t, sigma_t, x_0_grid, self.eps,
                use_reentrant=False,
            )
        else:
            v_t, _ = dx_policy_forward_jit(x_t, sigma_t, x_0_grid, self.eps)
        return v_t


@torch.jit.script
def dx_policy_forward_jit(
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    x_0_grid: torch.Tensor,
    eps: float = 1e-4,
):
    """JIT-compiled DX policy forward pass."""
    B, K, C, F, H, W = x_0_grid.shape
    sigma_t_exp = sigma_t.view(B, 1, 1, 1, 1)
    t_normalized = 1.0 - sigma_t
    
    if K == 1:
        x_0 = x_0_grid.squeeze(1)
        v_t = (x_t - x_0) / sigma_t_exp.clamp(min=eps)
        return v_t, x_0
    
    t_scaled = t_normalized.clamp(min=0, max=1) * (K - 1)
    t0 = t_scaled.floor().long().clamp(min=0, max=K - 2)
    t1 = t0 + 1
    
    w1 = (t_scaled - t0.to(x_t.dtype)).view(B, 1, 1, 1, 1)
    w0 = 1 - w1
    
    t0_expanded = t0.view(B, 1, 1, 1, 1, 1).expand(-1, -1, C, F, H, W)
    t1_expanded = t1.view(B, 1, 1, 1, 1, 1).expand(-1, -1, C, F, H, W)
    
    x_0_t0 = torch.gather(x_0_grid, dim=1, index=t0_expanded).squeeze(1)
    x_0_t1 = torch.gather(x_0_grid, dim=1, index=t1_expanded).squeeze(1)
    
    x_0 = w0 * x_0_t0 + w1 * x_0_t1
    v_t = (x_t - x_0) / sigma_t_exp.clamp(min=eps)
    return v_t, x_0


def dx_policy_forward(
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    x_0_grid: torch.Tensor,
    eps: float = 1e-4,
):
    """Non-JIT DX policy forward pass for gradient checkpointing."""
    B, K, C, F, H, W = x_0_grid.shape
    sigma_t_exp = sigma_t.view(B, 1, 1, 1, 1)
    t_normalized = 1.0 - sigma_t
    
    if K == 1:
        x_0 = x_0_grid.squeeze(1)
        v_t = (x_t - x_0) / sigma_t_exp.clamp(min=eps)
        return v_t, x_0
    
    t_scaled = t_normalized.clamp(min=0, max=1) * (K - 1)
    t0 = t_scaled.floor().long().clamp(min=0, max=K - 2)
    t1 = t0 + 1
    
    w1 = (t_scaled - t0.to(x_t.dtype)).view(B, 1, 1, 1, 1)
    w0 = 1 - w1
    
    t0_expanded = t0.view(B, 1, 1, 1, 1, 1).expand(-1, -1, C, F, H, W)
    t1_expanded = t1.view(B, 1, 1, 1, 1, 1).expand(-1, -1, C, F, H, W)
    
    x_0_t0 = torch.gather(x_0_grid, dim=1, index=t0_expanded).squeeze(1)
    x_0_t1 = torch.gather(x_0_grid, dim=1, index=t1_expanded).squeeze(1)
    
    x_0 = w0 * x_0_t0 + w1 * x_0_t1
    v_t = (x_t - x_0) / sigma_t_exp.clamp(min=eps)
    return v_t, x_0


class GMPolicy(nn.Module):
    """
    GMM-based policy for computing velocity v_t from GMM parameters.
    Supports both GMM (full Bayesian update) and DMM (hard assignment) modes.
    """
    def __init__(
        self, 
        eps: float = 1e-4, 
        use_gradient_checkpointing: bool = False, 
        policy_type: str = 'gmm',
        patch_size: tuple = (1, 2, 2),
        use_patch_distance: bool = False,
        use_patch_K: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.policy_type = policy_type
        self.patch_size = patch_size
        self.use_patch_distance = use_patch_distance
        self.use_patch_K = use_patch_K
        assert policy_type in ['gmm', 'dmm'], f"Invalid policy_type: {policy_type}"
    
    def forward(
        self, 
        x_t: torch.Tensor, 
        sigma_t: torch.Tensor,
        a_s: torch.Tensor, 
        mu_s: torch.Tensor, 
        std_s: torch.Tensor,
        x_s: torch.Tensor, 
        sigma_s: torch.Tensor,
        use_gradient_checkpointing: bool = None,
    ) -> torch.Tensor:
        """
        Compute velocity v_t based on GMM parameters from x_s.
        
        Args:
            x_t: (B, C, F, H, W) - noisy sample at timestep t
            sigma_t: (B,) - noise level at timestep t
            a_s: (B, K, 1, F, H, W) - log-weights
            mu_s: (B, K, C, F, H, W) - velocity means
            std_s: (B, 1, 1, 1, 1, 1) - standard deviations
            x_s: (B, C, F, H, W) - noisy sample at timestep s
            sigma_s: (B,) - noise level at timestep s
            use_gradient_checkpointing: Optional override for gradient checkpointing
            
        Returns:
            v_t: (B, C, F, H, W) - velocity at timestep t
        """
        use_dmm = self.policy_type == 'dmm'
        
        # Allow per-call override of gradient checkpointing
        if use_gradient_checkpointing is None:
            use_gradient_checkpointing = self.use_gradient_checkpointing
        
        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                gmm_policy_forward,
                x_t, sigma_t, a_s, mu_s, std_s, x_s, sigma_s, self.eps, use_dmm,
                use_reentrant=False,
            )
        else:
            return gmm_policy_forward_jit(
                x_t, sigma_t, a_s, mu_s, std_s, x_s, sigma_s, self.eps, use_dmm
            )


@torch.jit.script
def gmm_policy_forward_jit(
    x_t: torch.Tensor, 
    sigma_t: torch.Tensor,
    a_s: torch.Tensor,
    mu_s: torch.Tensor, 
    std_s: torch.Tensor,
    x_s: torch.Tensor,
    sigma_s: torch.Tensor,
    eps: float = 1e-4,
    use_dmm: bool = False,
):
    """JIT-compiled GMM policy forward pass."""
    x_t = x_t.unsqueeze(1)
    x_s = x_s.unsqueeze(1)
    sigma_t = sigma_t.view(-1, 1, 1, 1, 1, 1)
    sigma_s = sigma_s.view(-1, 1, 1, 1, 1, 1)

    # Compute v_s for fallback
    A_s = a_s.softmax(dim=1)
    v_s = (A_s * mu_s).sum(dim=1, keepdim=True)
    
    # Compute x0 predictions
    x0_s = x_s - sigma_s * mu_s
    x0_var_s = (std_s * sigma_s).square()

    # Bayesian update
    nu_x = sigma_s.square() * (1 - sigma_t) * x_t - sigma_t.square() * (1 - sigma_s) * x_s
    xi_x = sigma_s.square() * (1 - sigma_t).square() - sigma_t.square() * (1 - sigma_s).square()
    xi_x = xi_x.clamp(min=eps)
    x0_t = nu_x / xi_x
    x0_var_t = sigma_s.square() * sigma_t.square() / xi_x

    denomi = (x0_var_t + x0_var_s).clamp(min=eps)
    mu_t = (x0_var_t * x0_s + x0_var_s * x0_t) / denomi
    
    dist = (x0_s - x0_t).square().sum(dim=2, keepdim=True)
    a_t = a_s - 0.5 * dist / denomi
    A_t = a_t.softmax(dim=1)

    if use_dmm:
        a_t_dmm = a_s - 0.5 * dist / x0_var_t.clamp(min=eps)
        A_t_dmm = a_t_dmm.softmax(dim=1)
        x_0 = (x0_s * A_t_dmm).sum(dim=1, keepdim=True)
    else:
        x_0 = (mu_t * A_t).sum(dim=1, keepdim=True)
    
    v_t = (x_t - x_0) / sigma_t.clamp(min=eps)

    # Numerical safety fallback
    is_numerical_safe = torch.logical_or(
        sigma_s.square() * (1 - sigma_t).square() - sigma_t.square() * (1 - sigma_s).square() > eps,
        x0_var_t + x0_var_s > eps
    )
    v_t = torch.where(is_numerical_safe, v_t, v_s)
    v_t = v_t.squeeze(1)

    all_equal = bool((sigma_s == sigma_t).all())
    if all_equal:
        return v_s.squeeze(1)
    return v_t


def gmm_policy_forward(
    x_t: torch.Tensor, 
    sigma_t: torch.Tensor,
    a_s: torch.Tensor,
    mu_s: torch.Tensor, 
    std_s: torch.Tensor,
    x_s: torch.Tensor,
    sigma_s: torch.Tensor,
    eps: float = 1e-4,
    use_dmm: bool = False,
):
    """Non-JIT GMM policy forward pass for gradient checkpointing."""
    x_t = x_t.unsqueeze(1)
    x_s = x_s.unsqueeze(1)
    sigma_t = sigma_t.view(-1, 1, 1, 1, 1, 1)
    sigma_s = sigma_s.view(-1, 1, 1, 1, 1, 1)

    A_s = a_s.softmax(dim=1)
    v_s = (A_s * mu_s).sum(dim=1, keepdim=True)
    
    x0_s = x_s - sigma_s * mu_s
    x0_var_s = (std_s * sigma_s).square()

    nu_x = sigma_s.square() * (1 - sigma_t) * x_t - sigma_t.square() * (1 - sigma_s) * x_s
    xi_x = sigma_s.square() * (1 - sigma_t).square() - sigma_t.square() * (1 - sigma_s).square()
    xi_x = xi_x.clamp(min=eps)
    x0_t = nu_x / xi_x
    x0_var_t = sigma_s.square() * sigma_t.square() / xi_x

    denomi = (x0_var_t + x0_var_s).clamp(min=eps)
    mu_t = (x0_var_t * x0_s + x0_var_s * x0_t) / denomi
    
    dist = (x0_s - x0_t).square().sum(dim=2, keepdim=True)
    a_t = a_s - 0.5 * dist / denomi
    A_t = a_t.softmax(dim=1)

    if use_dmm:
        a_t_dmm = a_s - 0.5 * dist / x0_var_t.clamp(min=eps)
        A_t_dmm = a_t_dmm.softmax(dim=1)
        x_0 = (x0_s * A_t_dmm).sum(dim=1, keepdim=True)
    else:
        x_0 = (mu_t * A_t).sum(dim=1, keepdim=True)
    
    v_t = (x_t - x_0) / sigma_t.clamp(min=eps)

    is_numerical_safe = torch.logical_or(xi_x > eps, denomi > eps)
    v_t = torch.where(is_numerical_safe, v_t, v_s)
    v_t = v_t.squeeze(1)

    if (sigma_s == sigma_t).all():
        return v_s.squeeze(1)
    return v_t


# ============================================================
# WanPiflowWrapper
# ============================================================

class WanPiflowWrapper(torch.nn.Module):
    """
    Wrapper around CausalWanModel with policy prediction heads.
    Supports GMM, DMM, and DX policy types for pi-flow training.
    """
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            from wan.modules.model import WanModel
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        self.uniform_timestep = not is_causal
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760
        self.has_gmm_heads = False
        self.has_dx_heads = False
        self.pi = None
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def init_gmm_heads(
        self, 
        num_gaussians: int = 4,
        latent_c: int = 16,
        patch_size: tuple = (1, 2, 2),
        gm_num_logstd_layers: int = 2,
        logstd_inner_dim: int = 256,
        init_logstd: float = 0.05,
        use_patch_K: bool = False,
    ) -> None:
        """Create GMM prediction head modules."""
        self.inner_dim = self.model.dim
        self.num_gaussians = num_gaussians
        self.latent_c = latent_c
        self.patch_size = patch_size
        self.patch_size_total = math.prod(patch_size)
        self._init_logstd = init_logstd
        self.use_patch_K = use_patch_K
        
        self.proj_out_means = nn.Linear(
            self.inner_dim, num_gaussians * latent_c * self.patch_size_total
        )
        self.proj_out_logweights = nn.Linear(self.inner_dim, num_gaussians)
        
        logstd_layers = []
        in_dim = self.inner_dim
        for _ in range(gm_num_logstd_layers - 1):
            logstd_layers.extend([nn.SiLU(), nn.Linear(in_dim, logstd_inner_dim)])
            in_dim = logstd_inner_dim
        self.proj_out_logstds = nn.Sequential(*logstd_layers, nn.SiLU(), nn.Linear(in_dim, 1))
        
        self.has_gmm_heads = True
        self.policy_head_type = 'gmm'
        print(f"[WanPiflowWrapper] Created GMM heads with K={num_gaussians}, latent_c={latent_c}")

    def initialize_gmm_head_weights(self) -> None:
        """Initialize GMM head weights from pretrained proj_out."""
        assert self.has_gmm_heads, "GMM heads not created"
        
        for m in self.proj_out_logstds:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.proj_out_logstds[-1].bias.data.fill_(np.log(self._init_logstd))
        
        with torch.no_grad():
            proj_out_weight = self.model.head.head.weight.data
            proj_out_bias = self.model.head.head.bias.data
            
            self.proj_out_means.weight.data = proj_out_weight[None].expand(
                self.num_gaussians, -1, -1
            ).reshape(self.num_gaussians * self.latent_c * self.patch_size_total, -1)
            self.proj_out_means.bias.data = proj_out_bias[None].expand(
                self.num_gaussians, -1
            ).reshape(self.num_gaussians * self.latent_c * self.patch_size_total)
            
            device = self.proj_out_means.bias.device
            dtype = self.proj_out_means.bias.dtype
            rand_noise = torch.randn(self.num_gaussians * self.latent_c, device=device, dtype=dtype) * 0.05
            self.proj_out_means.bias.data += rand_noise[:, None].expand(-1, self.patch_size_total).flatten()
        
        nn.init.zeros_(self.proj_out_logweights.weight)
        nn.init.zeros_(self.proj_out_logweights.bias)
        print(f"[WanPiflowWrapper] Initialized GMM head weights from pretrained")

    def init_dx_heads(
        self,
        num_grid_points: int = 4,
        latent_c: int = 16,
        patch_size: tuple = (1, 2, 2),
        use_patch_K: bool = False,
    ) -> None:
        """Create DX prediction head modules."""
        self.inner_dim = self.model.dim
        self.num_grid_points = num_grid_points
        self.latent_c = latent_c
        self.patch_size = patch_size
        self.patch_size_total = math.prod(patch_size)
        self.use_patch_K = use_patch_K
        
        self.proj_out_dx = nn.Linear(
            self.inner_dim, num_grid_points * latent_c * self.patch_size_total
        )
        
        self.has_dx_heads = True
        self.policy_head_type = 'dx'
        self.pi = DXPolicy(eps=1e-4, use_gradient_checkpointing=False, use_patch_K=use_patch_K, patch_size=patch_size)
        print(f"[WanPiflowWrapper] Created DX heads with K={num_grid_points}, latent_c={latent_c}")

    def initialize_dx_head_weights(self) -> None:
        """Initialize DX head weights from pretrained proj_out."""
        assert self.has_dx_heads, "DX heads not created"
        
        with torch.no_grad():
            proj_out_weight = self.model.head.head.weight.data
            proj_out_bias = self.model.head.head.bias.data
            
            self.proj_out_dx.weight.data = proj_out_weight[None].expand(
                self.num_grid_points, -1, -1
            ).reshape(self.num_grid_points * self.latent_c * self.patch_size_total, -1)
            self.proj_out_dx.bias.data = proj_out_bias[None].expand(
                self.num_grid_points, -1
            ).reshape(self.num_grid_points * self.latent_c * self.patch_size_total)
            
            device = self.proj_out_dx.bias.device
            dtype = self.proj_out_dx.bias.dtype
            rand_noise = torch.randn(self.num_grid_points * self.latent_c, device=device, dtype=dtype) * 0.05
            self.proj_out_dx.bias.data += rand_noise[:, None].expand(-1, self.patch_size_total).flatten()
        print(f"[WanPiflowWrapper] Initialized DX head weights from pretrained")

    def set_policy(self, policy_type: str = 'dmm', use_patch_distance: bool = False, use_patch_K: bool = None) -> None:
        """Set the policy object for ODE integration."""
        if use_patch_K is None:
            use_patch_K = getattr(self, 'use_patch_K', False)
        patch_size = getattr(self, 'patch_size', (1, 2, 2))
        
        if hasattr(self, 'has_dx_heads') and self.has_dx_heads:
            self.pi = DXPolicy(eps=1e-4, use_gradient_checkpointing=False, use_patch_K=use_patch_K, patch_size=patch_size)
            self.policy_type_for_ode = 'dx'
        elif hasattr(self, 'has_gmm_heads') and self.has_gmm_heads:
            self.pi = GMPolicy(eps=1e-4, use_gradient_checkpointing=False, policy_type=policy_type, 
                             patch_size=patch_size, use_patch_distance=use_patch_distance, use_patch_K=use_patch_K)
            self.policy_type_for_ode = policy_type
        else:
            raise ValueError("No policy heads initialized")
        print(f"[WanPiflowWrapper] Set policy to {self.policy_type_for_ode}")

    def get_direct_x0(self, params: dict, x_s: torch.Tensor, sigma_s: torch.Tensor, policy_type: str = 'dmm') -> torch.Tensor:
        """Direct x0 prediction from policy parameters."""
        if policy_type == 'dx':
            return self._get_direct_x0_dx(params, x_s, sigma_s)
        else:
            return self._get_direct_x0_gmm(params, x_s, sigma_s)
    
    def _get_direct_x0_gmm(self, params: dict, x_s: torch.Tensor, sigma_s: torch.Tensor) -> torch.Tensor:
        """Direct x0 from GMM parameters using low-temperature softmax."""
        mu_s = params['mu_s']
        a_s = params['a_s']
        
        if sigma_s.dim() == 1:
            sigma_s_exp = sigma_s.view(-1, 1, 1, 1, 1, 1)
        else:
            sigma_s_exp = sigma_s.view(sigma_s.shape[0], 1, 1, sigma_s.shape[1], 1, 1)
        x_s_exp = x_s.unsqueeze(1)
        
        x0_s = x_s_exp - sigma_s_exp * mu_s
        low_temp = 0.001
        A_s = (a_s / low_temp).softmax(dim=1)
        x_0 = (A_s * x0_s).sum(dim=1)
        return x_0
    
    def _get_direct_x0_dx(self, params: dict, x_s: torch.Tensor, sigma_s: torch.Tensor) -> torch.Tensor:
        """Direct x0 from DX parameters using interpolation."""
        x_0_grid = params['x_0_grid']
        B, K, C, F, H, W = x_0_grid.shape
        
        if sigma_s.dim() > 1:
            sigma_s = sigma_s[:, 0]
        
        t_normalized = 1.0 - sigma_s
        
        if K == 1:
            return x_0_grid.squeeze(1)
        
        t_scaled = t_normalized.clamp(min=0, max=1) * (K - 1)
        t0 = t_scaled.floor().long().clamp(min=0, max=K - 2)
        t1 = t0 + 1
        
        w1 = (t_scaled - t0.to(x_0_grid.dtype)).view(B, 1, 1, 1, 1)
        w0 = 1 - w1
        
        t0_expanded = t0.view(B, 1, 1, 1, 1, 1).expand(-1, -1, C, F, H, W)
        t1_expanded = t1.view(B, 1, 1, 1, 1, 1).expand(-1, -1, C, F, H, W)
        
        x_0_t0 = torch.gather(x_0_grid, dim=1, index=t0_expanded).squeeze(1)
        x_0_t1 = torch.gather(x_0_grid, dim=1, index=t1_expanded).squeeze(1)
        
        return w0 * x_0_t0 + w1 * x_0_t1

    def unpatchify_gm(self, gm: dict, orig_shape: tuple, grid_sizes: torch.Tensor) -> dict:
        """Unpatchify GMM parameters from sequence to spatial format."""
        B, C, F, H, W = orig_shape
        p_t, p_h, p_w = self.patch_size
        
        f_patches, h_patches, w_patches = grid_sizes[0].tolist()
        
        mu_s = gm['mu_s']
        mu_s = mu_s.reshape(B, self.num_gaussians, f_patches, h_patches, w_patches, p_t, p_h, p_w, C)
        mu_s = mu_s.permute(0, 1, 8, 2, 5, 3, 6, 4, 7)
        mu_s = mu_s.reshape(B, self.num_gaussians, C, F, H, W)
        
        a_s = gm['a_s']
        a_s = a_s.reshape(B, self.num_gaussians, f_patches, h_patches, w_patches, p_t, p_h, p_w, 1)
        a_s = a_s.permute(0, 1, 8, 2, 5, 3, 6, 4, 7)
        a_s = a_s.reshape(B, self.num_gaussians, 1, F, H, W)
        
        return {'mu_s': mu_s, 'a_s': a_s, 'std_s': gm['std_s']}

    def unpatchify_dx(self, dx: torch.Tensor, orig_shape: tuple, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Unpatchify DX parameters from sequence to spatial format."""
        B, C, F, H, W = orig_shape
        p_t, p_h, p_w = self.patch_size
        f_patches, h_patches, w_patches = grid_sizes[0].tolist()
        
        x_0_grid = dx.reshape(B, self.num_grid_points, f_patches, h_patches, w_patches, p_t, p_h, p_w, C)
        x_0_grid = x_0_grid.permute(0, 1, 8, 2, 5, 3, 6, 4, 7)
        x_0_grid = x_0_grid.reshape(B, self.num_grid_points, C, F, H, W)
        return x_0_grid

    def _get_sigma_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Get sigma values from timesteps using the scheduler."""
        original_shape = timestep.shape
        timestep_flat = timestep.flatten()
        
        sigmas = self.scheduler.sigmas.to(timestep.device)
        timesteps = self.scheduler.timesteps.to(timestep.device)
        
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep_flat.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id]
        return sigma.reshape(original_shape)

    def t_to_sigma(self, raw_t: torch.Tensor) -> torch.Tensor:
        """Convert raw timestep t in [0, 1] to sigma."""
        shift = self.scheduler.shift
        return shift * raw_t / (1 + (shift - 1) * raw_t)

    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to raw timestep t in [0, 1]."""
        shift = self.scheduler.shift
        return sigma / (shift - (shift - 1) * sigma)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Convert flow matching prediction to x0."""
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Convert x0 prediction to flow matching prediction."""
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        policy_mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass with optional policy mode."""
        if policy_mode is not None:
            return self._forward_policy_impl(
                noisy_image_or_video, conditional_dict, timestep,
                kv_cache, crossattn_cache, current_start, cache_start, policy_mode
            )
        
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0
    
    def _forward_policy_impl(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        policy_mode: str = 'gmm'
    ):
        """Internal policy forward implementation."""
        if policy_mode == 'dx':
            return self._forward_dx_impl(
                noisy_image_or_video, conditional_dict, timestep,
                kv_cache, crossattn_cache, current_start, cache_start
            )
        else:
            return self._forward_gmm_impl(
                noisy_image_or_video, conditional_dict, timestep,
                kv_cache, crossattn_cache, current_start, cache_start
            )

    def _forward_gmm_impl(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
    ) -> tuple:
        """GMM forward implementation."""
        assert self.has_gmm_heads, "GMM heads not initialized"
        
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
        
        if kv_cache is not None:
            hidden_states, grid_sizes = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, kv_cache=kv_cache,
                crossattn_cache=crossattn_cache, current_start=current_start,
                cache_start=cache_start, return_hidden=True
            )
        else:
            hidden_states, grid_sizes = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, return_hidden=True
            )
        
        B = noisy_image_or_video.shape[0]
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[1] * hidden_states.shape[2]
        
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(1, 2)
        
        out_means = self.proj_out_means(hidden_states)
        out_means = out_means.reshape(B, seq_len, self.num_gaussians, self.latent_c * self.patch_size_total).permute(0, 2, 1, 3)
        
        out_logweights = self.proj_out_logweights(hidden_states).log_softmax(dim=-1)
        out_logweights = out_logweights.unsqueeze(-1).repeat(1, 1, 1, self.patch_size_total).permute(0, 2, 1, 3)
        
        temb = hidden_states.mean(dim=1)
        out_logstds = self.proj_out_logstds(temb.detach()).exp().reshape(B, 1, 1, 1, 1, 1)
        
        gm_params_patchified = {'mu_s': out_means, 'a_s': out_logweights, 'std_s': out_logstds}
        
        orig_shape = noisy_image_or_video.shape
        orig_shape_reordered = (B, orig_shape[2], orig_shape[1], orig_shape[3], orig_shape[4])
        
        params = self.unpatchify_gm(gm_params_patchified, orig_shape_reordered, grid_sizes)
        
        sigma_t = self._get_sigma_from_timestep(timestep)
        x_s_reordered = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        pred_x0 = self.get_direct_x0(params, x_s_reordered, sigma_t, 'gmm')
        pred_x0 = pred_x0.permute(0, 2, 1, 3, 4)
        
        return params, pred_x0

    def _forward_dx_impl(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
    ) -> tuple:
        """DX forward implementation."""
        assert self.has_dx_heads, "DX heads not initialized"
        
        prompt_embeds = conditional_dict["prompt_embeds"]
        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
        
        if kv_cache is not None:
            hidden_states, grid_sizes = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, kv_cache=kv_cache,
                crossattn_cache=crossattn_cache, current_start=current_start,
                cache_start=cache_start, return_hidden=True
            )
        else:
            hidden_states, grid_sizes = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len, return_hidden=True
            )
        
        B = noisy_image_or_video.shape[0]
        seq_len = hidden_states.shape[1] if hidden_states.dim() == 3 else hidden_states.shape[1] * hidden_states.shape[2]
        
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.flatten(1, 2)
        
        out_velocities = self.proj_out_dx(hidden_states)
        out_velocities = out_velocities.reshape(B, seq_len, self.num_grid_points, self.latent_c * self.patch_size_total).permute(0, 2, 1, 3)
        
        orig_shape = noisy_image_or_video.shape
        orig_shape_reordered = (B, orig_shape[2], orig_shape[1], orig_shape[3], orig_shape[4])
        
        v_grid = self.unpatchify_dx(out_velocities, orig_shape_reordered, grid_sizes)
        
        sigma_t = self._get_sigma_from_timestep(timestep)
        if sigma_t.dim() > 1:
            sigma_s = sigma_t[:, 0]
        else:
            sigma_s = sigma_t
        sigma_s_exp = sigma_s.view(-1, 1, 1, 1, 1, 1)
        
        x_s_reordered = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        x_s_exp = x_s_reordered.unsqueeze(1)
        
        x_0_grid = x_s_exp - sigma_s_exp * v_grid
        
        params = {'x_0_grid': x_0_grid}
        pred_x0 = self.get_direct_x0(params, x_s_reordered, sigma_s, 'dx')
        pred_x0 = pred_x0.permute(0, 2, 1, 3, 4)
        
        return params, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """Get scheduler with interface methods."""
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """Post-initialization steps."""
        self.get_scheduler()
