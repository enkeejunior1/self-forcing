"""
Pi-Flow Trainer for distillation training with policy-based ODE integration.
"""
import gc
import logging
import re
import os
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from utils.dataset import ShardingLMDBDataset, TextDataset, cycle
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed, merge_dict_list
from model import Piflow


class PiFlowTrainer:
    """
    Trainer for Pi-Flow distillation with policy-based ODE integration.
    
    Supports:
    - GMM (Gaussian Mixture Model) policy
    - DX (Direct x0) policy
    - L2 velocity matching loss
    - Optional DMD loss for distribution matching
    """
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Initialize distributed training environment
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # Random seed setup
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        # Initialize wandb
        if self.is_main_process and not self.disable_wandb:
            wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=getattr(config, 'wandb_save_dir', 'wandb_logs')
            )
        self.output_path = config.logdir

        # Initialize model
        self.model = Piflow(config, device=self.device)

        # Load pretrained generator checkpoint BEFORE FSDP wrapping
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=False)

        # Initialize policy head weights BEFORE FSDP wrapping
        self.model.initialize_policy_head_weights()

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # Enable gradient checkpointing on the CausalWanModel before FSDP wrapping
        if getattr(config, 'gradient_checkpointing', False):
            self.model.generator.model.gradient_checkpointing = True
            print(f"[Trainer] Enabled gradient checkpointing on generator")

        # FSDP wrapping
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device,
                dtype=torch.bfloat16 if config.mixed_precision else torch.float32
            )

        # Initialize optimizers
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters() if param.requires_grad],
            lr=getattr(config, "lr_critic", config.lr),
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Initialize dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8
        )

        if dist.get_rank() == 0:
            print(f"DATASET SIZE {len(dataset)}")
        self.dataloader = cycle(dataloader)

        # Set up EMA
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue
            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        # Delete EMA params for early steps
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        """Save model checkpoint."""
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                exist_ok=True
            )
            torch.save(
                state_dict,
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt")
            )
            print(f"Model saved to {os.path.join(self.output_path, f'checkpoint_model_{self.step:06d}', 'model.pt')}")

    def fwdbwd_one_step(self, batch, model_type, piflow_loss_scale=0.0, dmd_loss_scale=0.0):
        """
        Forward and backward pass for one training step.
        
        Args:
            batch: Data batch with prompts
            model_type: 'generator' or 'critic'
            piflow_loss_scale: Scale for pi-flow loss (0.0 = disabled)
            dmd_loss_scale: Scale for DMD loss (0.0 = disabled)
        """
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Load data
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype
            )
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Get text embeddings
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size
                )
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Compute loss
        if model_type == 'generator':
            # Prepare input video from shape (generate noise internally in model)
            real_video = torch.zeros(image_or_video_shape, device=self.device, dtype=self.dtype)
            
            generator_loss, generator_log_dict = self.model.generator_loss(
                real_images_or_videos=real_video,
                conditional_dict=conditional_dict,
                piflow_loss_scale=piflow_loss_scale,
                dmd_loss_scale=dmd_loss_scale,
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )

            generator_log_dict.update({
                "generator_loss": generator_loss.detach(),
                "generator_grad_norm": generator_grad_norm
            })
            return generator_log_dict

        if model_type == 'critic':
            # For critic, we need real video data
            real_video = torch.zeros(image_or_video_shape, device=self.device, dtype=self.dtype)
            
            critic_loss, critic_log_dict = self.model.critic_loss(
                real_images_or_videos=real_video,
                conditional_dict=conditional_dict,
            )

            critic_loss.backward()
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic
            )

            critic_log_dict.update({
                "critic_loss": critic_loss.detach(),
                "critic_grad_norm": critic_grad_norm
            })
            return critic_log_dict

    def generate_video(self, prompts, seed=None):
        """Generate video using pi-flow sampling."""
        from pipeline import PiFlowTrainingPipeline
        
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
        batch_size = len(prompts)
        _, _, C, H, W = self.config.image_or_video_shape
        
        # Create pipeline
        # IMPORTANT: For inference/visualization, always use ALL denoising steps
        # last_step_only=True ensures all steps are executed (exit at last step = all 4 NFE used)
        # last_step_only=False would randomly exit early (only some NFE used)
        pipeline = PiFlowTrainingPipeline(
            denoising_step_list=self.model.denoising_step_list,
            scheduler=self.model.scheduler,
            generator=self.model.generator,
            num_frame_per_block=self.model.num_frame_per_block,
            independent_first_frame=self.config.independent_first_frame,
            same_step_across_blocks=self.config.same_step_across_blocks,
            last_step_only=True,  # Exit at last step = use all 4 NFE
            num_max_frames=self.model.num_training_frames,
            context_noise=self.config.context_noise,
            policy_type=getattr(self.config, 'policy_type', ''),
            sampling_type=getattr(self.config, 'sampling_type', 'piflow'),
            integration_nfe=getattr(self.config, 'integration_nfe', 128),
            x0_pred_type=getattr(self.config, 'x0_pred_type', 'dmd'),
        )
        
        # Get text embeddings
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=prompts)
        
        # Prepare noise
        sampled_noise = torch.randn(
            [batch_size, self.model.num_training_frames, C, H, W],
            device="cuda",
            dtype=self.dtype
        )
        
        # Run inference
        with torch.no_grad():
            output, _, _, _ = pipeline.inference_with_trajectory(
                noise=sampled_noise,
                initial_latent=None,
                **conditional_dict
            )
        
        # Decode latents to video
        video = self.model.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    @staticmethod
    def _sanitize_filename(text: str, max_len: int = 120) -> str:
        text = re.sub(r"\s+", "_", text.strip())
        text = re.sub(r"[^A-Za-z0-9._-]+", "", text)
        return text[:max_len] if text else "sample"

    @staticmethod
    def _to_uint8_video(video: np.ndarray) -> np.ndarray:
        if video.dtype != np.uint8:
            video = np.clip(video, 0, 255).astype(np.uint8)
        return video

    @staticmethod
    def _save_gif(video_uint8: np.ndarray, out_path: str, fps: int = 16) -> None:
        duration_ms = int(round(1000.0 / float(fps)))
        
        try:
            import imageio.v2 as imageio
            imageio.mimsave(
                out_path,
                [frame for frame in video_uint8],
                duration=duration_ms,
                loop=0,
            )
            print(f"[GIF] Saved: {out_path}")
            return
        except Exception as e:
            print(f"[GIF] imageio failed: {e}, trying Pillow...")
        
        try:
            from PIL import Image
            frames = [Image.fromarray(frame) for frame in video_uint8]
            if len(frames) == 0:
                print(f"[GIF] No frames to save for {out_path}")
                return
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
            )
            print(f"[GIF] Saved via Pillow: {out_path}")
        except Exception as e:
            print(f"[GIF] Failed to save GIF {out_path}: {e}")

    def visualize_samples(self):
        """Generate and save sample videos."""
        if self.config.no_visualize or not self.is_main_process:
            return
        
        prompts = [
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "She wears sunglasses and red lipstick. She walks confidently and casually. "
            "The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
        ]
        
        with torch.no_grad():
            videos = self.generate_video(prompts, seed=42)
        
        for i, (prompt, video) in enumerate(zip(prompts, videos)):
            video_uint8 = self._to_uint8_video(video)

            if getattr(self.config, "save_gif", True):
                out_dir = os.path.join(self.output_path, "samples")
                os.makedirs(out_dir, exist_ok=True)
                prompt_slug = self._sanitize_filename(prompt, max_len=80)
                out_path = os.path.join(
                    out_dir, f"step_{self.step:06d}_sample_{i}_{prompt_slug}.gif"
                )
                self._save_gif(video_uint8, out_path, fps=16)

    def train(self):
        """Main training loop."""
        start_step = self.step
        max_steps = getattr(self.config, 'max_steps', 100000)
        
        # Loss scales (0.0 = disabled)
        piflow_loss_scale = getattr(self.config, 'piflow_loss_scale', 1.0)
        dmd_loss_scale = getattr(self.config, 'dmd_loss_scale', 0.0)
        
        if self.is_main_process:
            print(f"[Trainer] Loss scales: piflow={piflow_loss_scale}, dmd={dmd_loss_scale}")
            if dmd_loss_scale == 0:
                print("[Trainer] DMD loss disabled, skipping critic training")
        
        # Create progress bar
        pbar = tqdm(
            initial=self.step,
            total=max_steps,
            desc="Training",
            disable=not self.is_main_process,
            dynamic_ncols=True
        )

        while self.step < max_steps:
            iter_start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            
            # Generator training
            # Compute effective scales for this step
            step_piflow_scale = piflow_loss_scale
            step_dmd_scale = dmd_loss_scale if (self.step % self.config.dfake_gen_update_ratio == 0) else 0.0

            if step_piflow_scale > 0 or step_dmd_scale > 0:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(
                    batch, 'generator',
                    piflow_loss_scale=step_piflow_scale,
                    dmd_loss_scale=step_dmd_scale,
                )
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Critic training (skip when dmd_loss_scale=0)
            critic_log_dict = {}
            if dmd_loss_scale > 0:
                self.critic_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, 'critic')
                extras_list.append(extra)
                critic_log_dict = merge_dict_list(extras_list)
                self.critic_optimizer.step()

            self.step += 1

            # Create EMA params
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save model
            if (not self.config.no_save) and (self.step - start_step) > 0 and \
                    self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Visualize samples
            if (self.step == 0 or self.step % 10 == 0) and self.is_main_process:
                self.visualize_samples()

            # Get loss values for display
            piflow_loss = generator_log_dict.get("piflow_loss", 0.0)
            if isinstance(piflow_loss, torch.Tensor):
                piflow_loss = piflow_loss.mean().item()
            
            gen_grad_norm = generator_log_dict.get("generator_grad_norm", 0.0)
            if isinstance(gen_grad_norm, torch.Tensor):
                gen_grad_norm = gen_grad_norm.mean().item()
            
            iter_time = time.time() - iter_start_time
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            # Update progress bar
            pbar.set_postfix({
                'pi_loss': f'{piflow_loss:.4f}',
                'gen_gnorm': f'{gen_grad_norm:.4f}',
                'iter_time': f'{iter_time:.2f}s',
                'peak_mem': f'{peak_mem_gb:.1f}GB'
            }, refresh=False)
            pbar.update(1)

            # Logging
            if self.is_main_process and not self.disable_wandb:
                wandb_loss_dict = {
                    "piflow_loss": piflow_loss,
                    "generator_grad_norm": gen_grad_norm,
                    "peak_memory_gb": peak_mem_gb,
                }
                
                if dmd_loss_scale > 0:
                    critic_loss = critic_log_dict.get("critic_loss", 0.0)
                    if isinstance(critic_loss, torch.Tensor):
                        critic_loss = critic_loss.mean().item()
                    wandb_loss_dict["critic_loss"] = critic_loss
                    
                    dmd_loss = generator_log_dict.get("dmd_loss", 0.0)
                    if isinstance(dmd_loss, torch.Tensor):
                        dmd_loss = dmd_loss.mean().item()
                    wandb_loss_dict["dmd_loss"] = dmd_loss
                
                wandb.log(wandb_loss_dict, step=self.step)

            # Garbage collection
            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log(
                            {"per iteration time": current_time - self.previous_time},
                            step=self.step
                        )
                    self.previous_time = current_time
        
        pbar.close()
        if self.is_main_process:
            print(f"Training completed at step {self.step}")
