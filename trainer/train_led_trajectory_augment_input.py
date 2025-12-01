#!/usr/bin/env python3

import os
import gc

import time
import torch


import random
import numpy as np
import torch.nn as nn

from utils.config import Config
from utils.utils import print_log
from utils.memory import print_mem


from torch.utils.data import DataLoader
from data.dataloader_nba import NBADataset, seq_collate


from models.model_led_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as CoreDenoisingModel

from tqdm.auto import tqdm  # <- added tqdm import
from datetime import datetime

import pdb
from torch.cuda import amp

NUM_Tau = 5


class Trainer:
    def __init__(self, config):
        if torch.cuda.is_available():
            torch.cuda.set_device(config.gpu)
            torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda") if config.cuda else torch.device("cpu")
        self.cfg = Config(config.cfg, config.info)

        # ------------------------- prepare train/test data loader -------------------------
        train_dset = NBADataset(
            obs_len=self.cfg.past_frames, pred_len=self.cfg.future_frames, training=True
        )
        test_dset = NBADataset(
            obs_len=self.cfg.past_frames,
            pred_len=self.cfg.future_frames,
            training=False,
        )

        # choose workers (honors explicit config, otherwise cores-based)
        num_workers = self._choose_num_workers(config)
        if num_workers > 0:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=self.cfg.train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=seq_collate,
                pin_memory=(num_workers > 0),
                persistent_workers=(num_workers > 0),
                prefetch_factor=4,
            )
            self.test_loader = DataLoader(
                test_dset,
                batch_size=self.cfg.test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=seq_collate,
                pin_memory=(num_workers > 0),
                persistent_workers=(num_workers > 0),
                prefetch_factor=4,
            )
        else:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=self.cfg.train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=seq_collate,
                pin_memory=False,
                persistent_workers=False,
            )
            self.test_loader = DataLoader(
                test_dset,
                batch_size=self.cfg.test_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=seq_collate,
                pin_memory=False,
                persistent_workers=False,
            )

        # data normalization parameters (create directly on target device)
        self.traj_mean = (
            torch.tensor(self.cfg.traj_mean, dtype=torch.float, device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.traj_scale = self.cfg.traj_scale

        # ------------------------- define diffusion parameters -------------------------
        self.n_steps = self.cfg.diffusion.steps  # define total diffusion steps

        # make beta schedule and calculate the parameters used in denoising process.
        # Create schedule directly on the trainer device to avoid copies
        self.betas = self.make_beta_schedule(
            schedule=self.cfg.diffusion.beta_schedule,
            n_timesteps=self.n_steps,
            start=self.cfg.diffusion.beta_start,
            end=self.cfg.diffusion.beta_end,
        ).to(self.device)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        # Precompute timestep-dependent constants on device to avoid repeated
        # arithmetic inside inner sampling loops.
        self._inv_sqrt_alphas = (1.0 / self.alphas.sqrt()).to(self.device)
        self._sigma = self.betas.sqrt().to(self.device)
        self._eps_factor = ((1 - self.alphas) / self.one_minus_alphas_bar_sqrt).to(
            self.device
        )

        # Prebuild a small cache of timestep scalars on device to avoid
        # allocating a full-size index tensor on every inner sampling step.
        # We'll expand these single-element tensors to the batch size when used.
        self._t_idx_cache = [
            torch.tensor([i], dtype=torch.long, device=self.device)
            for i in range(self.n_steps)
        ]

        # ------------------------- define models -------------------------
        self.model = CoreDenoisingModel().to(self.device)
        # load pretrained models
        model_cp = torch.load(
            self.cfg.pretrained_core_denoising_model, map_location="cpu"
        )
        self.model.load_state_dict(model_cp["model_dict"])

        # Freeze pretrained core denoiser (train initializer only, per paper)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.model_initializer = InitializationModel(
            t_h=10, d_h=6, t_f=20, d_f=2, k_pred=20
        ).to(self.device)

        # AMP configuration: enable only if requested and CUDA available
        self.use_amp = bool(
            getattr(config, "use_amp", False) and torch.cuda.is_available()
        )
        if self.use_amp:
            self.scaler = amp.GradScaler()

        self.opt = torch.optim.AdamW(
            self.model_initializer.parameters(), lr=config.learning_rate
        )
        self.scheduler_model = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma
        )

        # ------------------------- prepare logs -------------------------
        self.log = open(os.path.join(self.cfg.log_dir, "log.txt"), "a+")
        print_log(
            f"NEW RUN AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.log
        )
        self.print_model_param(self.model, name="Core Denoising Model")
        self.print_model_param(self.model_initializer, name="Initialization Model")

        # temporal reweight in the loss, create on-device to avoid copies
        self.temporal_reweight = (
            torch.tensor(
                [21 - i for i in range(1, 21)], dtype=torch.float, device=self.device
            )
            .unsqueeze(0)
            .unsqueeze(0)
            / 10
        )

    def print_model_param(self, model: nn.Module, name: str = "Model") -> None:
        """
        Count the trainable/total parameters in `model`.
        """
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_log(
            "[{}] Trainable/Total: {}/{}".format(name, trainable_num, total_num),
            self.log,
        )
        return None

    def tensor_stats(self):
        """Compute a lightweight summary of live torch tensors and log it.

        This is intended as a diagnostic to see whether live tensor count/bytes
        grows across batches (indicating a leak) vs allocator-reserved memory.
        """
        try:
            import gc

            gc.collect()
            objs = gc.get_objects()
        except Exception:
            print_log("tensor_stats: gc.get_objects() failed", self.log)
            return

        cnt = 0
        bytes_total = 0
        by_shape = {}
        for o in objs:
            try:
                # torch.is_tensor sometimes throws for non-torch objects; guard
                if torch.is_tensor(o):
                    cnt += 1
                    bytes_total += o.element_size() * o.nelement()
                    s = tuple(o.size())
                    by_shape[s] = by_shape.get(s, 0) + 1
            except Exception:
                continue

        msg = (
            f"Live torch tensors: {cnt}, approx bytes: {bytes_total / (1024**2):.1f} MB"
        )
        print_log(msg, self.log)
        if len(by_shape) > 0:
            top = sorted(by_shape.items(), key=lambda x: -x[1])[:10]
            print_log(f"Top tensor shapes (count): {top}", self.log)

    def _choose_num_workers(self, config):
        """Simplified workers selection.

        Rules:
        - If `config.workers` is provided and >= 0, use it (explicit override).
        - Otherwise choose based on CPU cores: `min(cores-1, cap)` with a small cap.
        """
        # explicit override from CLI/config
        try:
            w_cfg = getattr(config, "workers", None)
            if w_cfg is not None:
                w = int(w_cfg)
                if w >= 0:
                    print(f"Using user-provided workers={w}")
                    return w
        except Exception:
            pass

        cores = os.cpu_count() or 2
        cap = 8
        w = max(1, min(cores - 1, cap))
        print(f"Auto-selected workers={w} based on cores={cores}")
        return w

    def make_beta_schedule(
        self,
        schedule: str = "linear",
        n_timesteps: int = 1000,
        start: float = 1e-5,
        end: float = 1e-2,
    ) -> torch.Tensor:
        """
        Make beta schedule.

        Parameters
        ----
        schedule: str, in ['linear', 'quad', 'sigmoid'],
        n_timesteps: int, diffusion steps,
        start: float, beta start, `start<end`,
        end: float, beta end,

        Returns
        ----
        betas: Tensor with the shape of (n_timesteps)

        """
        if schedule == "linear":
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start**0.5, end**0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        # Avoid calling .to(...) on every inner loop iteration. Only move
        # `t` to the input device if it's not already there. In our sampling
        # code we ensure `t` is created on the correct device, so this branch
        # will almost always be a no-op and avoids repeated small copies.
        if t.device != input.device:
            t = t.to(input.device)
        out = torch.gather(input, 0, t)
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def noise_estimation_loss(self, x, y_0, mask):
        batch_size = x.shape[0]
        # Select a random step for each example
        t = torch.randint(0, self.n_steps, size=(batch_size // 2 + 1,), device=x.device)
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:batch_size]
        # x0 multiplier
        a = self.extract(self.alphas_bar_sqrt, t, y_0)
        beta = self.extract(self.betas, t, y_0)
        # eps multiplier
        am1 = self.extract(self.one_minus_alphas_bar_sqrt, t, y_0)
        e = torch.randn_like(y_0)
        # model input
        y = y_0 * a + e * am1
        # Use autocast for model forward if AMP is enabled
        with amp.autocast(enabled=self.use_amp):
            output = self.model(y, beta, x, mask)
        # batch_size, 20, 2
        return (e - output).square().mean()

    def p_sample(self, x, mask, cur_y, t):
        # Use cached single-element timestep tensor and expand to batch size
        t_idx = self._t_idx_cache[t].expand(x.shape[0])

        # Use precomputed per-timestep constants and reshape for broadcasting
        reshape = [t_idx.shape[0]] + [1] * (cur_y.dim() - 1)
        eps_factor = self._eps_factor[t_idx].reshape(*reshape)
        beta = self.betas[t_idx].reshape(*reshape)
        inv_sqrt_alpha = self._inv_sqrt_alphas[t_idx].reshape(*reshape)

        # Model output
        with amp.autocast(enabled=self.use_amp):
            eps_theta = self.model(cur_y, beta, x, mask)
        mean = inv_sqrt_alpha * (cur_y - (eps_factor * eps_theta))

        # Generate z in-place to avoid extra allocations
        z = torch.empty_like(cur_y).normal_()

        # Fixed sigma (precomputed)
        sigma_t = self._sigma[t_idx].reshape(*reshape)
        sample = mean + sigma_t * z
        return sample

    def p_sample_accelerate(self, cur_y, t, context_encoded, z=None):
        """Accelerated sample step that accepts a precomputed context encoding.

        Arguments:
        - cur_y: current noisy sample tensor
        - t: timestep (int)
        - context_encoded: output of self.model.encoder_context(x, mask)
        """
        # Use cached single-element timestep tensor and expand to batch size
        t_idx = self._t_idx_cache[t].expand(cur_y.shape[0])

        reshape = [t_idx.shape[0]] + [1] * (cur_y.dim() - 1)
        eps_factor = self._eps_factor[t_idx].reshape(*reshape)
        beta = self.betas[t_idx].reshape(*reshape)
        inv_sqrt_alpha = self._inv_sqrt_alphas[t_idx].reshape(*reshape)

        # Model output using pre-encoded context (avoids re-running encoder)
        with amp.autocast(enabled=self.use_amp):
            eps_theta = self.model.generate_accelerate_encoded(
                cur_y, beta, context_encoded
            )
        mean = inv_sqrt_alpha * (cur_y - (eps_factor * eps_theta))

        # Use caller-provided noise if available (pre-generated), else create
        if z is None:
            z = torch.empty_like(cur_y).normal_()

        # Fixed sigma (precomputed)
        sigma_t = self._sigma[t_idx].reshape(*reshape)
        sample = mean + sigma_t * z * 0.00001

        # Progress bar update: if an outer tqdm has been set by the caller, update it for this substep
        outer = getattr(self, "_active_tqdm", None)
        if outer is not None:
            try:
                outer.update(1)
            except Exception:
                pass

        return sample

    def p_sample_loop(self, x, mask, shape):
        self.model.eval()
        predictions = []
        for _ in range(20):
            cur_y = torch.randn(shape, device=x.device)
            for i in reversed(range(self.n_steps)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            predictions.append(cur_y.unsqueeze(1))
        prediction_total = torch.cat(predictions, dim=1)
        return prediction_total

    def p_sample_loop_mean(self, x, mask, loc):
        predictions = []
        for loc_i in range(1):
            cur_y = loc
            for i in reversed(range(NUM_Tau)):
                cur_y = self.p_sample(x, mask, cur_y, i)
            predictions.append(cur_y.unsqueeze(1))
        if len(predictions) > 0:
            prediction_total = torch.cat(predictions, dim=1)
        else:
            prediction_total = torch.empty(0, device=x.device)
        return prediction_total

    def p_sample_loop_accelerate(self, x, mask, loc):
        """
        Batch operation to accelerate the denoising process.

        x: [11, 10, 6]
        mask: [11, 11]
        cur_y: [11, 10, 20, 2]
        """
        # Precompute the encoder context once for the whole batch to avoid repeated encoder work
        context_encoded = self.model.encoder_context(x, mask)

        cur_y = loc[:, :10]
        # Pre-generate noise for the accelerated small loop to avoid per-step allocations
        noise_seq = torch.randn((NUM_Tau,) + cur_y.shape, device=cur_y.device)
        for i in reversed(range(NUM_Tau)):
            cur_y = self.p_sample_accelerate(cur_y, i, context_encoded, z=noise_seq[i])
        cur_y_ = loc[:, 10:]
        noise_seq2 = torch.randn((NUM_Tau,) + cur_y_.shape, device=cur_y_.device)
        for i in reversed(range(NUM_Tau)):
            cur_y_ = self.p_sample_accelerate(
                cur_y_, i, context_encoded, z=noise_seq2[i]
            )
        # shape: B=b*n, K=10, T, 2 -- concatenate the two halves once
        prediction_total = torch.cat((cur_y_, cur_y), dim=1)
        return prediction_total

    def fit(self):
        # Training loop
        for epoch in range(0, self.cfg.num_epochs):
            loss_total, loss_distance, loss_uncertainty = self._train_single_epoch(
                epoch
            )
            print_log(
                "[{}] Epoch: {}\t\tLoss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f}".format(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    epoch,
                    loss_total,
                    loss_distance,
                    loss_uncertainty,
                ),
                self.log,
            )

            if (epoch + 1) % self.cfg.test_interval == 0:
                performance, samples = self._test_single_epoch()
                for time_i in range(4):
                    print_log(
                        "--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}".format(
                            time_i + 1,
                            performance["ADE"][time_i] / samples,
                            time_i + 1,
                            performance["FDE"][time_i] / samples,
                        ),
                        self.log,
                    )
                cp_path = self.cfg.model_path % (epoch + 1)
                model_cp = {
                    "model_initializer_dict": self.model_initializer.state_dict()
                }
                torch.save(model_cp, cp_path)
            self.scheduler_model.step()

        torch.cuda.empty_cache()

        gc.collect()
        print("Training complete.")

    def data_preprocess(self, data):
        """
        pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
        fut_motion_3D: torch.Size([32, 11, 20, 2])
        fut_motion_mask: torch.Size([32, 11, 20])
        pre_motion_mask: torch.Size([32, 11, 10])
        traj_scale: 1
        pred_mask: None
        seq: nba
        """
        batch_size = data["pre_motion_3D"].shape[0]

        device = self.device
        # build block-diagonal trajectory mask directly on device
        block = torch.ones((11, 11), device=device)
        eye = torch.eye(batch_size, device=device)
        traj_mask = torch.kron(eye, block)

        # Move batch tensors to device once and reuse local references
        # Use non-blocking transfers from pinned memory (DataLoader has pin_memory=True)
        pre_motion = data["pre_motion_3D"].to(device, non_blocking=True)
        fut_motion = data["fut_motion_3D"].to(device, non_blocking=True)

        initial_pos = pre_motion[:, :, -1:]

        # augment input: absolute position, relative position, velocity
        past_traj_abs = (
            ((pre_motion - self.traj_mean) / self.traj_scale)
            .contiguous()
            .view(-1, 10, 2)
        )
        past_traj_rel = (
            ((pre_motion - initial_pos) / self.traj_scale).contiguous().view(-1, 10, 2)
        )

        past_traj_vel = torch.cat(
            (
                past_traj_rel[:, 1:] - past_traj_rel[:, :-1],
                torch.zeros_like(past_traj_rel[:, -1:]),
            ),
            dim=1,
        )

        past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1)

        fut_traj = (
            ((fut_motion - initial_pos) / self.traj_scale).contiguous().view(-1, 20, 2)
        )

        return batch_size, traj_mask, past_traj, fut_traj

    def _train_single_epoch(self, epoch):
        self.model.train()
        self.model_initializer.train()
        loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0

        # Use tqdm for the training data loader
        train_iter = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for data in train_iter:
            batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

            # Use autocast for initializer forward when AMP is enabled
            with amp.autocast(enabled=self.use_amp):
                sample_prediction, mean_estimation, variance_estimation = (
                    self.model_initializer(past_traj, traj_mask)
                )
            sample_prediction = (
                torch.exp(variance_estimation / 2)[..., None, None]
                * sample_prediction
                / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
            )
            loc = sample_prediction + mean_estimation[:, None]

            generated_y = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

            loss_dist = (
                (
                    (generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1)
                    * self.temporal_reweight
                )
                .mean(dim=-1)
                .min(dim=1)[0]
                .mean()
            )
            loss_uncertainty = (
                torch.exp(-variance_estimation)
                * (generated_y - fut_traj.unsqueeze(dim=1))
                .norm(p=2, dim=-1)
                .mean(dim=(1, 2))
                + variance_estimation
            ).mean()

            loss = loss_dist * 50 + loss_uncertainty
            loss_total += loss.item()
            loss_dt += loss_dist.item() * 50
            loss_dc += loss_uncertainty.item()

            self.opt.zero_grad()
            if self.use_amp:
                # scaled backward + unscale for gradient clipping
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.0)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_initializer.parameters(), 1.0)
                self.opt.step()
            count += 1

            # update tqdm with current loss
            train_iter.set_postfix(loss=loss.item())

            if self.cfg.debug and count == 2:
                break

        return loss_total / count, loss_dt / count, loss_dc / count

    def _test_single_epoch(self):
        # Accumulate metrics on device to avoid repeated CPU syncs
        performance_acc = {
            "FDE": [
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            ],
            "ADE": [
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            ],
        }
        samples = 0

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        with torch.no_grad():
            # BEGIN MODIFICATION: create manually-sized outer tqdm and let p_sample_accelerate update it
            substeps_per_batch = NUM_Tau * 2
            total_batches = len(self.test_loader)
            outer = tqdm(total=total_batches * substeps_per_batch, desc="Testing")
            self._active_tqdm = outer
            # END MODIFICATION: create manually-sized outer tqdm and let p_sample_accelerate update it

            for data in self.test_loader:
                batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

                with amp.autocast(enabled=self.use_amp):
                    sample_prediction, mean_estimation, variance_estimation = (
                        self.model_initializer(past_traj, traj_mask)
                    )
                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

                fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
                # b*n, K, T, 2
                distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (
                        (distances[:, :, : 5 * time_i])
                        .mean(dim=-1)
                        .min(dim=-1)[0]
                        .sum()
                    )
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    # Accumulate on device tensors to avoid CPU sync on each batch
                    performance_acc["ADE"][time_i - 1] += ade
                    performance_acc["FDE"][time_i - 1] += fde
                samples += distances.shape[0]
                count += 1

                # update tqdm with processed sample count
                outer.set_postfix(samples=samples)

                # if count==100:
                # 	break
            outer.close()
            delattr(self, "_active_tqdm")
        # Convert accumulated tensors to Python floats before returning (fits existing callers)
        performance = {
            "ADE": [performance_acc["ADE"][i].cpu().item() for i in range(4)],
            "FDE": [performance_acc["FDE"][i].cpu().item() for i in range(4)],
        }
        return performance, samples

    def save_data(self):
        """
        Save the visualization data.
        """
        model_path = "./results/checkpoints/led_vis.p"
        model_dict = torch.load(model_path, map_location=torch.device("cpu"))[
            "model_initializer_dict"
        ]
        self.model_initializer.load_state_dict(model_dict)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        root_path = "./visualization/data/"

        with torch.no_grad():
            for data in self.test_loader:
                _, traj_mask, past_traj, _ = self.data_preprocess(data)

                with amp.autocast(enabled=self.use_amp):
                    sample_prediction, mean_estimation, variance_estimation = (
                        self.model_initializer(past_traj, traj_mask)
                    )
                torch.save(sample_prediction, root_path + "p_var.pt")
                torch.save(mean_estimation, root_path + "p_mean.pt")
                torch.save(variance_estimation, root_path + "p_sigma.pt")

                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)
                pred_mean = self.p_sample_loop_mean(
                    past_traj, traj_mask, mean_estimation
                )

                torch.save(data["pre_motion_3D"], root_path + "past.pt")
                torch.save(data["fut_motion_3D"], root_path + "future.pt")
                torch.save(pred_traj, root_path + "prediction.pt")
                torch.save(pred_mean, root_path + "p_mean_denoise.pt")

                raise ValueError

    def test_single_model(self, path_model=None):
        model_path = path_model if path_model else "./results/checkpoints/led_new.p"
        model_dict = torch.load(model_path, map_location=torch.device("cpu"))[
            "model_initializer_dict"
        ]
        self.model_initializer.load_state_dict(model_dict)
        # Use torch tensors for accumulation to avoid many .item() CPU syncs
        # Create accumulation tensors directly on the trainer device to avoid
        # small CPU->GPU allocations during repeated adds in the inner loop.
        performance = {
            "FDE": [
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            ],
            "ADE": [
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
                torch.tensor(0.0, device=self.device),
            ],
        }
        samples = 0
        print_log(model_path, log=self.log)

        def prepare_seed(rand_seed):
            np.random.seed(rand_seed)
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)

        prepare_seed(0)
        count = 0
        with torch.no_grad():
            # BEGIN MODIFICATION: create manually-sized outer tqdm and let p_sample_accelerate update it
            substeps_per_batch = NUM_Tau * 2
            total_batches = len(self.test_loader)
            outer = tqdm(
                total=total_batches * substeps_per_batch, desc="Testing (single model)"
            )
            self._active_tqdm = outer
            # END MODIFICATION: create manually-sized outer tqdm and let p_sample_accelerate update it

            for data in self.test_loader:
                batch_size, traj_mask, past_traj, fut_traj = self.data_preprocess(data)

                sample_prediction, mean_estimation, variance_estimation = (
                    self.model_initializer(past_traj, traj_mask)
                )
                sample_prediction = (
                    torch.exp(variance_estimation / 2)[..., None, None]
                    * sample_prediction
                    / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
                )
                loc = sample_prediction + mean_estimation[:, None]

                pred_traj = self.p_sample_loop_accelerate(past_traj, traj_mask, loc)

                fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
                # b*n, K, T, 2
                distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
                for time_i in range(1, 5):
                    ade = (
                        (distances[:, :, : 5 * time_i])
                        .mean(dim=-1)
                        .min(dim=-1)[0]
                        .sum()
                    )
                    fde = (distances[:, :, 5 * time_i - 1]).min(dim=-1)[0].sum()
                    # accumulate as tensors (on GPU) then move minimal data only when needed
                    performance["ADE"][time_i - 1] = (
                        performance["ADE"][time_i - 1] + ade
                    )
                    performance["FDE"][time_i - 1] = (
                        performance["FDE"][time_i - 1] + fde
                    )
                samples += distances.shape[0]
                count += 1

                # update tqdm with processed sample count
                outer.set_postfix(samples=samples)
            outer.close()
            delattr(self, "_active_tqdm")
            # if count==2:
            # 	break
        # Move accumulated tensors to CPU and log once
        for time_i in range(4):
            ade_val = performance["ADE"][time_i].cpu().item() / samples
            fde_val = performance["FDE"][time_i].cpu().item() / samples
            print_log(
                "--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}".format(
                    time_i + 1,
                    ade_val,
                    time_i + 1,
                    fde_val,
                ),
                log=self.log,
            )
