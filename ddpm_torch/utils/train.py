import os
import re
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image as _save_image
import weakref
from tqdm import tqdm
from functools import partial
from contextlib import nullcontext

from ddpm_torch.functions import flat_mean

import torch.nn.functional as F
import copy
from copy import deepcopy

from sklearn.metrics import r2_score
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import beta
from torch.distributions import Beta
import numpy as np
import matplotlib.pyplot as plt

import torch.multiprocessing as mp
import time  # Added for wall clock time tracking


class DummyScheduler:
    @staticmethod
    def step():
        pass

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def state_dict():
        return None


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v / self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


save_image = partial(_save_image, normalize=True, value_range=(-1., 1.))


class Trainer:
    def __init__(
            self,
            model,
            replay_buffer,
            value_function,
            policy,
            optimizer,
            optimizer_v,
            optimizer_pi,
            diffusion,
            epochs,
            trainloader,
            sampler=None,
            scheduler=None,
            scheduler_v=None,
            scheduler_pi=None,
            num_accum=1,
            use_ema=False,
            grad_norm=1.0,
            shape=None,
            device=torch.device("cpu"),
            chkpt_intv=5,
            image_intv=1,
            fid_intv=None,     # New Argument
            num_samples=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0,  # process id for distributed training
            dry_run=False,
            alg='reinforce',
            use_baseline=False,
            max_buffer_length=10,
            n_features_to_select=5,
            update_policy=20,
            ent_coef=0.0,
            clip_ratio=0.2,
            sampler_type="adaptive"  # NEW ARGUMENT
    ):
        self.model = model
        self.value_function = value_function
        self.policy = policy
        self.sampler_type = sampler_type
        self.optimizer = optimizer
        self.optimizer_v = optimizer_v
        self.optimizer_pi = optimizer_pi

        self.diffusion = diffusion
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler
        self.scheduler_v = DummyScheduler() if scheduler_v is None else scheduler_v
        self.scheduler_pi = DummyScheduler() if scheduler_pi is None else scheduler_pi

        self.num_accum = num_accum
        self.grad_norm = grad_norm
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.image_intv = image_intv
        self.fid_intv = fid_intv if fid_intv is not None else self.epochs + 1 # Disable if not set
        self.num_samples = num_samples

        self.non_zero_coef_timesteps = torch.arange(3, dtype=torch.int64, device=self.device)
        self.alg = alg
        self.use_baseline = use_baseline
        self.max_buffer_length = max_buffer_length 
        self.n_features_to_select = n_features_to_select
        self.clip_ratio = clip_ratio
        
        self.replay_buffer = replay_buffer
        self.update_policy = update_policy
        self.ent_coef = ent_coef
        self.mse_loss = nn.MSELoss()

        self.states = []
        self.actions = []
        self.log_probs = []
        self.advantages = []

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.rank = rank
        self.dry_run = dry_run
        self.is_leader = rank == 0
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        # maintain a process-specific generator
        self.generator = torch.Generator(device).manual_seed(8191 + self.rank)

        self.sample_seed = 131071 + self.rank  # process-specific seed

        self.use_ema = use_ema
        if use_ema:
            if isinstance(model, DDP):
                self.ema = EMA(model.module, decay=ema_decay)
            else:
                self.ema = EMA(model, decay=ema_decay)
        else:
            self.ema = nullcontext()

        self.stats = RunningStatistics(loss=None)

    @property
    def timesteps(self):
        return self.diffusion.timesteps

    def loss(self, x, t):
        x = x.to(self.device)
        noise = torch.empty_like(x).normal_(generator=self.generator)
        loss, sampled_xt, sampled_t = self.diffusion.train_losses(self.model, x_0=x, t=t, noise=noise)
        assert loss.shape == (x.shape[0],)
        return loss, sampled_xt, sampled_t

    @torch.no_grad()
    def compute_singlestep_KL(self, x, sampled_t):
        mse_losses, sampled_xt, sampled_t = self.diffusion.train_losses(self.model, x, sampled_t)
        
        if hasattr(self.diffusion, "loss_type") and self.diffusion.loss_type == "flow_matching":
            # For Flow Matching, use the MSE loss directly as the "cost" metric
            # If sampled_t came from integer indices (adaptive sampler), normalize it to [0, 1]
            if sampled_t.dtype in [torch.int32, torch.int64, torch.long]:
                 t_in = sampled_t.float() / (self.diffusion.timesteps - 1)
                 # Recalculate loss with correct float time
                 mse_losses, _, _ = self.diffusion.train_losses(self.model, x, t_in)
            
            return mse_losses

        kl_divergence = self.compute_kl_divergence(x, sampled_xt, sampled_t)
        return kl_divergence

    @torch.no_grad()
    def calculate_kl_for_all_x0_at_t(self, sampling_timestep, x):
        kl_divergence_tensor_list = []
        
        # Robustly handle timestep input: convert Tensor/nested lists to flat list of scalars
        ts = sampling_timestep
        if isinstance(ts, torch.Tensor):
            ts = ts.tolist()
        if isinstance(ts, (list, tuple)) and len(ts) > 0 and isinstance(ts[0], (list, tuple)):
            ts = [item for sublist in ts for item in sublist]

        for j in ts:
            # Ensure j is a python scalar (int/float)
            val = j.item() if hasattr(j, "item") else j
            
            t_full = torch.full((x.shape[0],), val, device=self.device)
            kl_divergence = self.compute_singlestep_KL(x, t_full)
            kl_divergence_tensor_list.append(kl_divergence)
            
        kl_divergence_tensor = torch.stack(kl_divergence_tensor_list).transpose(0, 1)
        return kl_divergence_tensor

    def sample_timesteps(self, x):
        alpha, beta = self.policy(x)
        alpha = alpha.squeeze()
        beta = beta.squeeze()

        dist = Beta(alpha, beta)
        dist_sampled = dist.sample()
    
        timesteps = dist_sampled * (self.timesteps - 1) # Scale to [0, 99]
        timesteps = torch.round(timesteps).long()

        log_probs = dist.log_prob(dist_sampled)
        entropy = dist.entropy()
        
        return timesteps.detach(), log_probs, alpha, beta, entropy, dist_sampled

    def store_trajectory(self, state, action, log_prob, ret):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.advantages.append(ret)

    def feature_selector(self, kl_diff_lasso, kl_diff_sum_lasso):
        X = kl_diff_lasso.cpu().numpy()  
        y = kl_diff_sum_lasso.cpu().numpy().ravel()
        
        non_zero_columns = np.any(X != 0, axis=1)
        X = X[non_zero_columns]
        y = y[non_zero_columns]

        # Select the best features using SelectKBest
        k_best = SelectKBest(score_func=f_regression, k=self.n_features_to_select)
        k_best.fit(X, y)
        selected_features_indices = k_best.get_support(indices=True)

        return selected_features_indices.tolist(),

    def compute_kl_divergence(self, x, sampled_xt, sampled_t):
        q_mean, q_var, q_logvar = self.diffusion.q_posterior_mean_var(x, sampled_xt, sampled_t)
        p_mean, p_var, p_logvar = self.diffusion.p_mean_var(self.model, sampled_xt, sampled_t, clip_denoised=False, return_pred=False)
        kl = self.normal_kl(q_mean, q_logvar, p_mean, p_logvar)
        kl = flat_mean(kl) / math.log(2.)  # natural base to base 2
        return kl

    @torch.jit.script
    def normal_kl(mean1, logvar1, mean2, logvar2):
        diff_logvar = logvar1 - logvar2
        kl = (-1.0 - diff_logvar).add(
            (mean1 - mean2).pow(2) * torch.exp(-logvar2)).add(
            torch.exp(diff_logvar)).mul(0.5)
        return kl

    def step(self, x, e, i, global_steps=1, logger=None, start_time=None):
        B = x.shape[0]
        T = self.diffusion.timesteps
        is_flow_matching = hasattr(self.diffusion, "loss_type") and self.diffusion.loss_type == "flow_matching"
        is_adaptive_like = self.sampler_type in ["adaptive", "a2c_ats"]

        sampled_t_input = None
        log_probs, entropy, sampled_act, alpha_value, beta_value = None, None, None, None, None
        
        # 1. Sample Timesteps based on strategy
        if is_adaptive_like:
            sampled_t, log_probs, alpha_value, beta_value, entropy, sampled_act = self.sample_timesteps(x)
            # Use continuous sample [0,1] for FM, else discrete indices
            sampled_t_input = sampled_act if is_flow_matching else sampled_t
        elif self.sampler_type == "uniform":
            if is_flow_matching:
                sampled_t_input = torch.rand((B,), device=self.device)
            else:
                sampled_t_input = torch.randint(0, T, (B,), device=self.device)
            sampled_t = sampled_t_input 
        elif self.sampler_type == "ln":
            z = torch.randn((B,), device=self.device)
            t_float = torch.sigmoid(z)
            # Logit-Normal
            sampled_t_input = t_float if is_flow_matching else (t_float * (T - 1)).long()
            sampled_t = sampled_t_input
        elif self.sampler_type == "bernoulli90":
            # Static Bernoulli / Piecewise Constant Sampling
            # Zone A: [0, 0.2) with prob 0.1
            # Zone B: [0.2, 1.0] with prob 0.9
            
            # 1. Decide which zone for each batch element
            zone_mask = torch.rand((B,), device=self.device) < 0.9  # True = Zone B (90%)
            
            # 2. Sample uniformly within zones
            t_vals = torch.empty((B,), device=self.device)
            
            # Fill Zone B [0.2, 1.0]
            # range is 0.8 wide, starts at 0.2
            t_vals[zone_mask] = (torch.rand(zone_mask.sum(), device=self.device) * 0.8) + 0.2
            
            # Fill Zone A [0.0, 0.2)
            # range is 0.2 wide, starts at 0.0
            t_vals[~zone_mask] = (torch.rand((~zone_mask).sum(), device=self.device) * 0.2)
            
            sampled_t_input = t_vals if is_flow_matching else (t_vals * (T - 1)).long()
            sampled_t = sampled_t_input

        elif self.sampler_type == "bernoulli95":
            # Zone A: [0, 0.2) with prob 0.05
            # Zone B: [0.2, 1.0] with prob 0.95
            
            zone_mask = torch.rand((B,), device=self.device) < 0.95  # True = Zone B (95%)
            t_vals = torch.empty((B,), device=self.device)
            t_vals[zone_mask] = (torch.rand(zone_mask.sum(), device=self.device) * 0.8) + 0.2
            t_vals[~zone_mask] = (torch.rand((~zone_mask).sum(), device=self.device) * 0.2)
            
            sampled_t_input = t_vals if is_flow_matching else (t_vals * (T - 1)).long()
            sampled_t = sampled_t_input

        elif self.sampler_type == "bernoulli_inv90":
            # Inverse Bernoulli 90%
            # Zone A: [0, 0.8) with prob 0.9 (Noise area)
            # Zone B: [0.8, 1.0] with prob 0.1
            
            zone_mask = torch.rand((B,), device=self.device) < 0.9  # True = Zone A (90%)
            t_vals = torch.empty((B,), device=self.device)
            
            # Fill Zone A [0.0, 0.8)
            t_vals[zone_mask] = (torch.rand(zone_mask.sum(), device=self.device) * 0.8)
            
            # Fill Zone B [0.8, 1.0]
            t_vals[~zone_mask] = (torch.rand((~zone_mask).sum(), device=self.device) * 0.2) + 0.8
            
            sampled_t_input = t_vals if is_flow_matching else (t_vals * (T - 1)).long()
            sampled_t = sampled_t_input

        elif self.sampler_type == "bernoulli_inv95":
            # Inverse Bernoulli 95%
            # Zone A: [0, 0.8) with prob 0.95 (Noise area)
            # Zone B: [0.8, 1.0] with prob 0.05
            
            zone_mask = torch.rand((B,), device=self.device) < 0.95  # True = Zone A (95%)
            t_vals = torch.empty((B,), device=self.device)
            
            # Fill Zone A [0.0, 0.8)
            t_vals[zone_mask] = (torch.rand(zone_mask.sum(), device=self.device) * 0.8)
            
            # Fill Zone B [0.8, 1.0]
            t_vals[~zone_mask] = (torch.rand((~zone_mask).sum(), device=self.device) * 0.2) + 0.8
            
            sampled_t_input = t_vals if is_flow_matching else (t_vals * (T - 1)).long()
            sampled_t = sampled_t_input

        # NEW: Beta Noise Strategy (Biased towards t=0, which is NOISE)
        elif self.sampler_type == "beta_noise":
            # Fixed Beta Distribution (alpha=0.8, beta=1.0)
            # PDF ~ x^-0.2, high prob near 0
            beta_dist = torch.distributions.Beta(
                torch.tensor([0.8], device=self.device), 
                torch.tensor([1.0], device=self.device)
            )
            t_vals = beta_dist.sample((B,)).squeeze(-1)
            sampled_t_input = t_vals if is_flow_matching else (t_vals * (T - 1)).long()
            sampled_t = sampled_t_input

        # NEW: Beta Data Strategy (Biased towards t=1, which is DATA)
        elif self.sampler_type == "beta_data":
            # Fixed Beta Distribution (alpha=1.0, beta=0.8)
            # PDF ~ (1-x)^-0.2, high prob near 1
            beta_dist = torch.distributions.Beta(
                torch.tensor([1.0], device=self.device), 
                torch.tensor([0.8], device=self.device)
            )
            # Sample B times
            t_vals = beta_dist.sample((B,)).squeeze(-1)
            
            sampled_t_input = t_vals if is_flow_matching else (t_vals * (T - 1)).long()
            sampled_t = sampled_t_input

        # 2. Pre-update calculations for Adaptive Policy (Feature Selection / Replay Buffer)
        # We only do this if we are going to update the policy later
        if is_adaptive_like and i % self.update_policy == 0:
            total_T = torch.arange(T, dtype=torch.int64, device=self.device)
            chunked_T = torch.chunk(total_T, self.world_size)
            range_T = chunked_T[self.rank]

            if self.is_leader:
                random_index = torch.randint(0, x.shape[0], (1,), device=self.device)
                leader_x_sampled = x[random_index]
                self.replay_buffer.add_x_sampled(leader_x_sampled)
            
            if self.distributed:
                dist.barrier()

            x_sampled = self.replay_buffer.x_sampled.repeat(len(range_T), 1, 1, 1)
            x_sampled = x_sampled.to(self.device)
            kl_before_for_lasso = self.compute_singlestep_KL(x_sampled, range_T)
            
            # KL_sum before for reward (using all coefficients of interest)
            kl_divergence_tensor_before = self.calculate_kl_for_all_x0_at_t(self.non_zero_coef_timesteps, x)

        # 3. Diffusion Update
        # DETACH sampled_t_input to prevent gradients flowing back into policy/sampler
        # during diffusion loss backward pass.
        if torch.is_tensor(sampled_t_input):
            t_input_for_diffusion = sampled_t_input.detach()
        else:
            t_input_for_diffusion = sampled_t_input

        dif_loss, _, _ = self.loss(x, t_input_for_diffusion)
        
        if torch.isnan(dif_loss).any():
             print(f"NaN loss detected! Steps: {global_steps}, Sampler: {self.sampler_type}")
        
        dif_loss.mean().backward()
        
        if global_steps % self.num_accum == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.use_ema and hasattr(self.ema, "update"):
                self.ema.update()
        
        dif_loss = dif_loss.detach()
        if self.distributed:
            dist.reduce(dif_loss, dst=0, op=dist.ReduceOp.SUM)
            dif_loss.div_(self.world_size)
        self.stats.update(B, loss=dif_loss.mean().item() * B)
        
        # 4. Policy Update (Adaptive Only)
        if is_adaptive_like and i % self.update_policy == 0:
            kl_after_for_lasso = self.compute_singlestep_KL(x_sampled, range_T)
            kl_diff_lasso = kl_before_for_lasso - kl_after_for_lasso
            kl_diff_sum_lasso = kl_diff_lasso.sum()

            self.replay_buffer.sum_gpus(kl_diff_lasso, kl_diff_sum_lasso, self.rank)
            if self.distributed:
                dist.barrier()

            if self.is_leader:
                self.replay_buffer.add()
            if self.distributed:
                dist.barrier()

            kl_divergence_tensor_after = self.calculate_kl_for_all_x0_at_t(self.non_zero_coef_timesteps, x)
            kl_diff = kl_divergence_tensor_before - kl_divergence_tensor_after
            kl_diff_sum = kl_diff.sum(dim=1)

            if self.replay_buffer.size.item() > 1:
                self.non_zero_coef_timesteps = self.feature_selector(self.replay_buffer.buffer_X, self.replay_buffer.buffer_y)
                
            reward = kl_diff_sum.detach()
            if entropy is not None:
                reward += self.ent_coef * entropy

            if self.sampler_type == "a2c_ats":
                # Value Network estimates expected reward of image x
                baseline_value = self.value_function(x).squeeze()

                # 1. Critic Update (MSE Loss)
                # target for value function should be detached to prevent backprop into policy (via entropy in reward)
                value_loss = self.mse_loss(baseline_value, reward.detach())
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_function.parameters(), max_norm=self.grad_norm)
                self.optimizer_v.step()
                self.optimizer_v.zero_grad()
                self.scheduler_v.step()
                
                value_loss_item = value_loss.detach().item()
            
            else:
                baseline_value = torch.zeros_like(reward)
                value_loss_item = 0.0
    
            advantage = reward - baseline_value.detach()
            # 3. The Entropy Injection (Coupled into the Advantage)
            if entropy is not None:
                reward_eff = advantage + (self.ent_coef * entropy)
            else:
                reward_eff = advantage
            
            # policy grad
            if log_probs is not None:
                actor_loss = -(log_probs * reward_eff).mean()
                actor_loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.grad_norm)
                self.optimizer_pi.step()
                self.optimizer_pi.zero_grad()
                self.scheduler_pi.step()
                
                actor_loss = actor_loss.detach()
                if self.distributed:
                    dist.reduce(actor_loss, dst=0, op=dist.ReduceOp.SUM)
                    actor_loss.div_(self.world_size)
                
                # Log policy metrics
                if logger and self.is_leader:
                    log_payload = {
                        "policy/alpha_mean": alpha_value.mean().item() if alpha_value is not None else 0,
                        "policy/beta_mean": beta_value.mean().item() if beta_value is not None else 0,
                        "policy/entropy_mean": entropy.mean().item() if entropy is not None else 0,
                        "policy/kl_diff_sum_mean": kl_diff_sum.mean().item(),
                        "policy/advantage_mean": advantage.mean().item(),
                        "policy/reward_eff_mean": reward_eff.mean().item(),
                        "policy/actor_loss": actor_loss.item(),
                        "policy/value_loss": value_loss_item,
                        "epoch": e,
                        "global_step": global_steps
                    }
                    if start_time is not None:
                        log_payload["wall_clock_time"] = time.time() - start_time
                    
                    logger.log(log_payload, step=global_steps)

    def sample_fn(self, sample_size=None, noise=None, diffusion=None, sample_seed=None):
        if noise is None:
            shape = (sample_size // self.world_size,) + self.shape
        else:
            shape = noise.shape
        if diffusion is None:
            diffusion = self.diffusion
        with self.ema:
            sample = diffusion.p_sample(
                denoise_fn=self.model, shape=shape,
                device=self.device, noise=noise, seed=sample_seed)
        if self.distributed:
            # equalizes GPU memory usages across all processes within the same process group
            sample_list = [torch.zeros(shape, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(sample_list, sample)
            sample = torch.cat(sample_list, dim=0)
        assert sample.grad is None
        return sample


    def train(self, evaluator=None, chkpt_path=None, image_dir=None, logger=None):
        nrow = math.floor(math.sqrt(self.num_samples))
        if self.num_samples:
            assert self.num_samples % self.world_size == 0, "Number of samples should be divisible by WORLD_SIZE!"

        if self.dry_run:
            self.start_epoch, self.epochs = 0, 1

        global_steps = 0
        start_time = time.time()  # Start tracking time

        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            self.value_function.train()
            
            results = dict()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(self.trainloader, desc=f"{e + 1}/{self.epochs} epochs", disable=not self.is_leader) as t:
                for i, x in enumerate(t): # i: index, x : [128, 3, 32, 32]
                    if isinstance(x, (list, tuple)):
                        x = x[0]  # unconditional model -> discard labels
                    global_steps += 1
                    self.step(x.to(self.device), e=e, i=i, global_steps=global_steps, logger=logger, start_time=start_time)
                    t.set_postfix(self.current_stats)
                    results.update(self.current_stats)
                    
                    # Log training loss and time immediately if logger is present
                    if logger and self.is_leader:
                        current_loss = self.stats.extract().get("loss", 0)
                        elapsed_time = time.time() - start_time
                        logger.log({
                            "train_loss": current_loss,
                            "wall_clock_time": elapsed_time,
                            "epoch": e,
                            "global_step": global_steps
                        }, step=global_steps)

                    if self.dry_run and not global_steps % self.num_accum:
                        break

            # --- Independent Evaluation Loops ---
            is_last_epoch = (e == self.epochs - 1)
            
            # A. Image Generation
            if (not (e + 1) % self.image_intv) or is_last_epoch:
                if self.num_samples and image_dir:
                    self.model.eval()
                    self.value_function.eval()
                    torch.cuda.empty_cache()
                    
                    x = self.sample_fn(sample_size=self.num_samples, sample_seed=self.sample_seed)

                    if self.is_leader:
                        save_image(x, os.path.join(image_dir, f"{e + 1}.jpg"), nrow=nrow)
                        if logger:
                            log_data = {}
                            images = x.cpu().numpy().transpose(0, 2, 3, 1)
                            wandb_images = [logger.Image(image, caption=f"Image {i}") for i, image in enumerate(images)]
                            log_data.update({
                                "images": wandb_images,
                                "wall_clock_time": time.time() - start_time,
                                "epoch": e,
                                "global_step": global_steps
                            })
                            logger.log(log_data, step=global_steps)

            # B. FID Evaluation
            if (not (e + 1) % self.fid_intv) or is_last_epoch:
                if evaluator is not None:
                    # Memory cleanup before large FID calculation
                    torch.cuda.empty_cache() 
                    self.model.eval()
                    self.value_function.eval()

                    eval_results = evaluator.eval(self.sample_fn, is_leader=self.is_leader)
                    
                    if self.is_leader:
                        elapsed_time = time.time() - start_time
                        logger.log({
                            "FID": eval_results['fid'],
                            "wall_clock_time": elapsed_time,
                            "epoch": e,
                            "global_step": global_steps
                        }, step=global_steps)
                        print(f"FID: {eval_results['fid']}") 
                    results.update(eval_results)

            if logger and self.is_leader:
                # Redundant but ensures epoch-end logging aligns if step logging was skipped/conditional
                elapsed_time = time.time() - start_time
                logger.log({
                    "global_steps": global_steps,
                    "epoch": e,
                    "diffusion_timesteps": self.diffusion.timesteps,
                    "wall_clock_time": elapsed_time
                }, step=global_steps)

            if self.distributed:
                dist.barrier()  # synchronize all processes here

    @property
    def trainees(self):
        roster = ["model", "optimizer", "optimizer_v", "optimizer_pi"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None: 
            roster.append("scheduler")
        if self.scheduler_v is not None: 
            roster.append("scheduler_v")
        if self.scheduler_pi is not None:
            roster.append("scheduler_pi")
            
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def load_checkpoint(self, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.startswith("module."):
                        _chkpt[k.split(".", maxsplit=1)[1]] = _chkpt.pop(k)
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class EMA:
    """
    exponential moving average
    inspired by:
    [1] https://github.com/fadel/pytorch_ema
    [2] https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L281-L685
    """

    def __init__(self, model, decay=0.9999):
        shadow = []
        refs = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                shadow.append((k, v.detach().clone()))
                refs.append((k, weakref.ref(v)))
        self.shadow = dict(shadow)
        self._refs = dict(refs)
        self.decay = decay
        self.num_updates = -1
        self.backup = None

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, _ref in self._refs.items():
            assert _ref() is not None, "referenced object no longer exists!"
            self.shadow[k] += (1 - decay) * (_ref().data - self.shadow[k])

    def apply(self):
        self.backup = dict([
            (k, _ref().detach().clone()) for k, _ref in self._refs.items()])
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.shadow[k])

    def restore(self):
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.backup[k])
        self.backup = None

    def __enter__(self):
        self.apply()

    def __exit__(self, *exc):
        self.restore()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "num_updates": self.num_updates
        }

    @property
    def extra_states(self):
        return {"decay", "num_updates"}

    def load_state_dict(self, state_dict, strict=True):
        _dict_keys = set(self.__dict__["shadow"]).union(self.extra_states)
        dict_keys = set(state_dict["shadow"]).union(self.extra_states)
        incompatible_keys = set.symmetric_difference(_dict_keys, dict_keys) \
            if strict else set.difference(_dict_keys, dict_keys)
        if incompatible_keys:
            raise RuntimeError(
                "Key mismatch!\n"
                f"Missing key(s): {', '.join(set.difference(_dict_keys, dict_keys))}."
                f"Unexpected key(s): {', '.join(set.difference(dict_keys, _dict_keys))}"
            )
        self.__dict__.update(state_dict)


class ModelWrapper(nn.Module):
    def __init__(
            self,
            model,
            pre_transform=None,
            post_transform=None
    ):
        super().__init__()
        self._model = model
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def forward(self, x, *args, **kwargs):
        if self.pre_transform is not None:
            x = self.pre_transform(x)
        out = self._model(x, *args, **kwargs)
        if self.post_transform is not None:
            out = self.post_transform(out)
        return out
