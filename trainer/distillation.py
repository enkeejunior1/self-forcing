import gc
import logging
import os

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list,
    print_gpu_tensors,
    print_gpu_memory_summary
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time

# Enable via: DEBUG_GPU_MEMORY=1
DEBUG_GPU_MEMORY = os.environ.get('DEBUG_GPU_MEMORY', '0') == '1'


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
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

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

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
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
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

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

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
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def _sanitize_filename(self, text, max_len=80):
        """Convert text to a safe filename string."""
        import re
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text[:max_len]

    def _to_uint8_video(self, video):
        """Convert video tensor to uint8 numpy array."""
        # video: (C, T, H, W) or (T, H, W, C)
        if isinstance(video, torch.Tensor):
            video = video.cpu().numpy()
        video = (video * 255.0).clip(0, 255).astype('uint8')
        return video

    def _save_gif(self, video_uint8, out_path, fps=16):
        """Save video as animated GIF."""
        try:
            from PIL import Image
            # Assume video_uint8 is (T, H, W, C)
            if video_uint8.ndim == 3:  # (T, H, W)
                video_uint8 = video_uint8[..., None]
            frames = [Image.fromarray(video_uint8[i]) for i in range(video_uint8.shape[0])]
            duration_ms = int(1000 / fps)
            frames[0].save(
                out_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"[GIF] Saved via Pillow: {out_path}")
        except Exception as e:
            print(f"[GIF] Failed to save GIF {out_path}: {e}")

    def visualize_samples(self):
        """Generate and save sample videos. All ranks must call this for FSDP."""
        # Skip if visualization is disabled
        if self.config.no_visualize:
            return
        
        # Print only on main process
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"[Visualize Step {self.step}] Starting qualitative sample generation...")
            print(f"{'='*60}")
        
        prompts = [
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
            "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
            "She wears sunglasses and red lipstick. She walks confidently and casually. "
            "The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
        ]
        
        batch_size = len(prompts)
        _, _, C, H, W = self.config.image_or_video_shape
        
        # Get text embeddings (ALL RANKS must call FSDP model)
        if self.is_main_process:
            print("[1/4] Encoding text prompts...")
        start = time.time()
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=prompts)
        if self.is_main_process:
            print(f"  └─ Text encoding: {time.time()-start:.2f}s")
        
        # Prepare noise
        if self.is_main_process:
            print("[2/4] Running inference (denoising blocks)...")
        sampled_noise = torch.randn(
            [batch_size, self.model.num_training_frames, C, H, W],
            device=self.device,
            dtype=self.dtype
        )
        
        # Use model's existing inference pipeline (ALL RANKS must call FSDP model)
        start = time.time()
        with torch.no_grad():
            output, _, _ = self.model._consistency_backward_simulation(
                noise=sampled_noise,
                **conditional_dict  # Unpack dict as kwargs
            )
        if self.is_main_process:
            print(f"  └─ Inference: {time.time()-start:.2f}s")
        
        # Decode latents to video (ALL RANKS must call FSDP VAE)
        if self.is_main_process:
            print("[3/4] Decoding latents to pixels (VAE)...")
        start = time.time()
        video = self.model.vae.decode_to_pixel(output, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if self.is_main_process:
            print(f"  └─ VAE decoding: {time.time()-start:.2f}s")
        
        # Convert to uint8 format: (B, T, H, W, C)
        videos = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        videos = videos.clip(0, 255).astype('uint8')
        
        # Save GIF only on main process
        if self.is_main_process:
            print("[4/4] Saving GIF...")
            start = time.time()
            for i, (prompt, video_array) in enumerate(zip(prompts, videos)):
                if getattr(self.config, "save_gif", True):
                    out_dir = os.path.join(self.output_path, "samples")
                    os.makedirs(out_dir, exist_ok=True)
                    prompt_slug = self._sanitize_filename(prompt, max_len=80)
                    out_path = os.path.join(
                        out_dir, f"step_{self.step:06d}_sample_{i}_{prompt_slug}.gif"
                    )
                    self._save_gif(video_array, out_path, fps=16)
                    print(f"  └─ Saved: {out_path}")
            print(f"  └─ GIF saving: {time.time()-start:.2f}s")
            
            print(f"[Visualize Step {self.step}] Complete!")
            print(f"{'='*60}\n")

    def train(self):
        start_step = self.step

        while True:
            # Visualize samples every 10 iterations (before training to see current model state)
            # ALL RANKS must call this because it uses FSDP models
            if ((self.step - start_step) % 10 == 0):
                torch.cuda.empty_cache()
                self.visualize_samples()  # All ranks execute
                torch.cuda.empty_cache()
                dist.barrier()  # Ensure all ranks finish before continuing

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Memory debug: log before training step
            if DEBUG_GPU_MEMORY and self.is_main_process and self.step <= start_step + 3:
                print_gpu_tensors(logging, tag=f"BEFORE step={self.step}")
                print_gpu_memory_summary(logging, tag=f"BEFORE step={self.step}")

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                
                # Memory debug: before generator forward/backward
                if DEBUG_GPU_MEMORY and self.is_main_process and self.step <= start_step + 3:
                    print_gpu_memory_summary(logging, tag=f"BEFORE generator fwdbwd step={self.step}")
                
                extra = self.fwdbwd_one_step(batch, True)
                
                # Memory debug: after generator forward/backward
                if DEBUG_GPU_MEMORY and self.is_main_process and self.step <= start_step + 3:
                    print_gpu_tensors(logging, tag=f"AFTER generator fwdbwd step={self.step}")
                    print_gpu_memory_summary(logging, tag=f"AFTER generator fwdbwd step={self.step}")
                
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            
            # Memory debug: before critic forward/backward
            if DEBUG_GPU_MEMORY and self.is_main_process and self.step <= start_step + 3:
                print_gpu_memory_summary(logging, tag=f"BEFORE critic fwdbwd step={self.step}")
            
            extra = self.fwdbwd_one_step(batch, False)
            
            # Memory debug: after critic forward/backward
            if DEBUG_GPU_MEMORY and self.is_main_process and self.step <= start_step + 3:
                print_gpu_tensors(logging, tag=f"AFTER critic fwdbwd step={self.step}")
                print_gpu_memory_summary(logging, tag=f"AFTER critic fwdbwd step={self.step}")
            
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
