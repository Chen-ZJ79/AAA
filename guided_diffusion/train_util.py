import copy
import functools
import os
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from .losses import (
    dice_loss, focal_loss_multiclass, combined_segmentation_loss,
    UncertaintyWeightedLoss
)

# Suppress sklearn warnings for single-class batches
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
# from visdom import Visdom
# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        data,
        dataloader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_cls_head=False,
        class_weights=None,
        focal_gamma=2.0,
        seg_focal_gamma=1.5,
        seg_focal_lambda=0.5,
    ):
        self.model = model
        self.dataloader=dataloader
        self.classifier = classifier
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        
        # ==================== Classification-related ====================
        self.use_cls_head = use_cls_head
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.seg_focal_gamma = seg_focal_gamma
        self.seg_focal_lambda = seg_focal_lambda
        
        # Initialize uncertainty-weighted loss if using classification
        if self.use_cls_head:
            self.uncertainty_loss = UncertaintyWeightedLoss()
            if th.cuda.is_available():
                self.uncertainty_loss = self.uncertainty_loss.to(dist_util.dev())
        else:
            self.uncertainty_loss = None
        
        # Metrics tracking
        self.epoch_metrics = {
            'dice': [], 'iou': [], 'auc': [], 'f1': [], 'acc': [],
            'loss_seg': [], 'loss_cls': [], 'loss_total': [],
            'sigma_seg': [], 'sigma_cls': []
        }
        # ============================================================

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        # Add uncertainty loss parameters to optimizer if using classification
        opt_params = list(self.mp_trainer.master_params)
        if self.use_cls_head and self.uncertainty_loss is not None:
            opt_params += list(self.uncertainty_loss.parameters())

        self.opt = AdamW(
            opt_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_part_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.dataloader)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):

            try:
                if self.use_cls_head:
                  batch, cond, grade_label, name = next(data_iter)
                else:
                  batch, cond, name = next(data_iter)
                  grade_label = None
            except StopIteration:
                  # StopIteration is thrown if dataset ends
                  # reinitialize data loader
                data_iter = iter(self.dataloader)
                if self.use_cls_head:
                  batch, cond, grade_label, name = next(data_iter)
                else:
                  batch, cond, name = next(data_iter)
                  grade_label = None

            self.run_step(batch, cond, grade_label)

           
            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, grade_label=None):
        # 安全的batch拼接：只在cond是有效tensor时才拼接
        if cond is not None and isinstance(cond, th.Tensor) and cond.shape[0] == batch.shape[0]:
            batch = th.cat((batch, cond), dim=1)
        
        cond = {}
        sample = self.forward_backward(batch, cond, grade_label)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond, grade_label=None):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            
            # Handle grade labels if using classification
            micro_grade = None
            if self.use_cls_head and grade_label is not None:
                micro_grade = grade_label[i : i + self.microbatch].to(dist_util.dev())

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
                grade_labels=micro_grade,
                use_cls_head=self.use_cls_head,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses1[0]["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            # Compute segmentation loss (already in losses["loss"])
            loss_seg = (losses["loss"] * weights + losses['loss_cal'] * 10).mean()
            
            # Ensure loss_seg is a scalar
            if loss_seg.numel() > 1:
                loss_seg = loss_seg.mean()
            
            # Initialize total loss with segmentation loss
            total_loss = loss_seg
            loss_cls = th.tensor(0.0).to(dist_util.dev())
            
            # Add classification loss if enabled
            if self.use_cls_head and "cls_logits" in losses and "grade_labels" in losses:
                cls_logits = losses["cls_logits"]
                grade_labels = losses["grade_labels"]
                
                # Compute classification loss with focal loss and class weighting
                class_weights_device = None
                if self.class_weights is not None:
                    class_weights_device = self.class_weights.to(dist_util.dev())
                
                loss_cls = focal_loss_multiclass(
                    cls_logits, 
                    grade_labels.long(), 
                    alpha=class_weights_device,
                    gamma=self.focal_gamma
                )
                
                # Uncertainty-weighted multi-task loss
                if self.uncertainty_loss is not None:
                    # Stabilize losses before uncertainty weighting
                    loss_seg_stable = th.clamp(loss_seg, 0, 10.0)  # Clamp to prevent explosion
                    loss_cls_stable = th.clamp(loss_cls, 0, 10.0)
                    
                    total_loss, sigma_seg, sigma_cls = self.uncertainty_loss(loss_seg_stable, loss_cls_stable)
                    
                    # Log uncertainty parameters
                    logger.logkv("sigma_seg", sigma_seg.item())
                    logger.logkv("sigma_cls", sigma_cls.item())
                    self.epoch_metrics['sigma_seg'].append(sigma_seg.item())
                    self.epoch_metrics['sigma_cls'].append(sigma_cls.item())
                else:
                    # Simple weighted sum if no uncertainty weighting
                    total_loss = loss_seg + loss_cls
                
                # Compute classification metrics
                with th.no_grad():
                    pred_probs = th.softmax(cls_logits, dim=1)
                    pred_labels = th.argmax(pred_probs, dim=1)
                    
                    # Accuracy
                    acc = (pred_labels == grade_labels).float().mean().item()
                    
                    # F1 Score
                    try:
                        f1_raw = f1_score(grade_labels.cpu().numpy(), pred_labels.cpu().numpy(), average='binary')
                        f1 = float(f1_raw)  # Ensure it's a Python float
                    except:
                        f1 = 0.0
                    
                    # AUC (requires probability of positive class)
                    try:
                        auc_raw = roc_auc_score(grade_labels.cpu().numpy(), pred_probs[:, 1].cpu().numpy())
                        auc = float(auc_raw)  # Ensure it's a Python float
                    except:
                        auc = 0.5
                    
                    # Log classification metrics
                    logger.logkv("cls_acc", float(acc))
                    logger.logkv("cls_f1", float(f1))
                    logger.logkv("cls_auc", float(auc))
                    self.epoch_metrics['acc'].append(float(acc))
                    self.epoch_metrics['f1'].append(float(f1))
                    self.epoch_metrics['auc'].append(float(auc))
                
                # Log losses
                logger.logkv("loss_cls", loss_cls.item())
                self.epoch_metrics['loss_cls'].append(loss_cls.item())
            
            # Compute segmentation metrics (Dice and IoU)
            with th.no_grad():
                # Get ground truth mask
                gt_mask = micro[:, -1:, ...]
                gt_mask = th.where(gt_mask > 0, 1, 0).float()
                
                # Predicted mask from sample
                pred_mask = th.sigmoid(sample)
                pred_mask_binary = (pred_mask > 0.5).float()
                
                # Dice coefficient
                intersection = (pred_mask_binary * gt_mask).sum()
                dice = (2.0 * intersection + 1e-5) / (pred_mask_binary.sum() + gt_mask.sum() + 1e-5)
                
                # IoU
                union = pred_mask_binary.sum() + gt_mask.sum() - intersection
                iou = (intersection + 1e-5) / (union + 1e-5)
                
                logger.logkv("seg_dice", dice.item())
                logger.logkv("seg_iou", iou.item())
                self.epoch_metrics['dice'].append(dice.item())
                self.epoch_metrics['iou'].append(iou.item())
            
            # Log losses
            # Ensure all losses are scalars before logging
            loss_seg_scalar = loss_seg.item() if loss_seg.numel() == 1 else loss_seg.mean().item()
            total_loss_scalar = total_loss.item() if total_loss.numel() == 1 else total_loss.mean().item()
            
            logger.logkv("loss_seg", loss_seg_scalar)
            logger.logkv("loss_total", total_loss_scalar)
            self.epoch_metrics['loss_seg'].append(loss_seg_scalar)
            self.epoch_metrics['loss_total'].append(total_loss_scalar)

            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )
            # 只记录包含 "loss" 的键，避免把 logits/labels 传进去
            filtered_losses = {k: (v * weights) for k, v in losses.items() if "loss" in k}
            log_loss_dict(self.diffusion, t, filtered_losses)

            # Ensure total_loss is a scalar before backward
            if total_loss.numel() > 1:
                total_loss = total_loss.mean()
            
            self.mp_trainer.backward(total_loss)
            
            ###########################################
            # Gradient clipping to prevent explosion
            th.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)
            
            # 调试：检查冻结参数是否有梯度（仅在需要时启用）
            # for name, param in self.ddp_model.named_parameters():
            #     if param.grad is not None and not param.requires_grad:
            #         print(f"⚠️  冻结参数有梯度: {name}")
            
            return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", float(sub_loss))
