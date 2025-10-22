
import sys
import random
sys.path.append(".")
from guided_diffusion.utils import staple

import numpy
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
import math
from PIL import Image
import matplotlib.pyplot as plt
from guided_diffusion.utils import staple
import argparse

import collections
import logging
import math
import os
import time
from datetime import datetime
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

import dateutil.tz


def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_seg(pred, true_mask_p, threshold=(0.1, 0.3, 0.5, 0.7, 0.9)):
    '''
    Evaluate segmentation metrics for BraTS dataset
    threshold: a int or a tuple of int
    masks: [b,1,h,w] for BraTS (single channel)
    pred: [b,1,h,w] for BraTS
    '''
    b, c, h, w = pred.size()
    
    # BraTS uses single channel (tumor segmentation)
    eiou, edice = 0, 0
    for th_val in threshold:
        gt_vmask_p = (true_mask_p > th_val).float()
        vpred = (pred > th_val).float()
        vpred_cpu = vpred.cpu()
        pred_np = vpred_cpu[:, 0, :, :].numpy().astype('int32')
        
        gt_mask_np = gt_vmask_p[:, 0, :, :].squeeze(1).cpu().numpy().astype('int32')
        
        # IoU for numpy
        eiou += iou(pred_np, gt_mask_np)
        
        # Dice for torch
        edice += dice_coeff(vpred[:, 0, :, :], gt_vmask_p[:, 0, :, :]).item()
        
    return eiou / len(threshold), edice / len(threshold)


def load_classification_results(csv_path):
    """Load classification results from CSV file"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Classification evaluation will be skipped.")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def create_comparison_visualization(pred_img, gt_img, original_img, sample_id, output_dir, 
                                 cls_pred=None, cls_true=None, cls_prob=None):
    """Create side-by-side comparison visualization"""
    
    # Convert tensors to numpy arrays
    if isinstance(pred_img, torch.Tensor):
        pred_np = pred_img.squeeze().cpu().numpy()
    else:
        pred_np = np.array(pred_img)
        
    if isinstance(gt_img, torch.Tensor):
        gt_np = gt_img.squeeze().cpu().numpy()
    else:
        gt_np = np.array(gt_img)
        
    if isinstance(original_img, torch.Tensor):
        orig_np = original_img.squeeze().cpu().numpy()
    else:
        orig_np = np.array(original_img)
    
    # Normalize to [0, 1]
    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)
    orig_np = (orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(orig_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(gt_np, cmap='hot')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred_np, cmap='hot')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = np.zeros((pred_np.shape[0], pred_np.shape[1], 3))
    overlay[:, :, 0] = pred_np  # Red for prediction
    overlay[:, :, 1] = gt_np    # Green for ground truth
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red=Pred, Green=GT)')
    axes[3].axis('off')
    
    # Add classification info if available
    if cls_pred is not None and cls_true is not None:
        grade_names = {0: 'LGG', 1: 'HGG'}
        title = f'Sample {sample_id}\nTrue: {grade_names.get(cls_true, "Unknown")}, Pred: {grade_names.get(cls_pred, "Unknown")}'
        if cls_prob is not None:
            title += f'\nProb(HGG): {cls_prob:.3f}'
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    # Save comparison image
    comparison_path = os.path.join(output_dir, f'{sample_id}_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return comparison_path

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--inp_pth", help="Path to prediction images folder")
    argParser.add_argument("--out_pth", help="Path to ground truth images folder")
    argParser.add_argument("--data_name", default="BRATS", help="Dataset name (BRATS)")
    argParser.add_argument("--csv_path", default=None, help="Path to CSV file with grade labels")
    argParser.add_argument("--comparison_dir", default="./comparisons", help="Directory to save comparison visualizations")
    argParser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary segmentation")
    args = argParser.parse_args()
    
    # Create comparison directory
    os.makedirs(args.comparison_dir, exist_ok=True)
    
    # Load classification results if available
    cls_results = None
    if args.csv_path and os.path.exists(args.csv_path):
        cls_results = load_classification_results(args.csv_path)
        print(f"Loaded classification results from {args.csv_path}")
    
    mix_res = (0, 0)
    num = 0
    pred_path = args.inp_pth
    gt_path = args.out_pth
    
    # Lists to store metrics for detailed analysis
    all_iou = []
    all_dice = []
    all_cls_pred = []
    all_cls_true = []
    all_cls_probs = []
    
    print(f"Evaluating predictions from: {pred_path}")
    print(f"Ground truth from: {gt_path}")
    print(f"Comparison images will be saved to: {args.comparison_dir}")
    print("=" * 60)
    
    for root, dirs, files in os.walk(pred_path, topdown=False):
        for name in files:
            if 'ens' in name:  # Ensemble results
                num += 1
                
                # Extract sample ID from filename
                if args.data_name == 'BRATS':
                    # For BraTS: filename format like "BraTS20_Training_001_slice001_output_ens.jpg"
                    parts = name.split('_')
                    if len(parts) >= 4:
                        sample_id = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
                    else:
                        sample_id = name.split('_output_ens')[0]
                else:
                    # For ISIC: original format
                    sample_id = name.split('_')[0]
                
                print(f"Processing sample {num}: {sample_id}")
                
                # Load prediction image
                pred_img_path = os.path.join(root, name)
                pred = Image.open(pred_img_path).convert('L')
                pred = torchvision.transforms.PILToTensor()(pred)
                pred = torch.unsqueeze(pred, 0).float()
                pred = pred / pred.max()
                
                # Load ground truth
                if args.data_name == 'BRATS':
                    # For BraTS, look for corresponding ground truth
                    # Try different naming patterns
                    gt_candidates = [
                        f"{sample_id}_ground_truth.jpg",
                        f"{sample_id}_gt.jpg",
                        f"{sample_id}_mask.jpg",
                        f"{sample_id}_seg.jpg"
                    ]
                    
                    gt_img_path = None
                    for candidate in gt_candidates:
                        candidate_path = os.path.join(gt_path, candidate)
                        if os.path.exists(candidate_path):
                            gt_img_path = candidate_path
                            break
                    
                    if gt_img_path is None:
                        print(f"Warning: Ground truth not found for {sample_id}, skipping...")
                        continue
                        
                    gt = Image.open(gt_img_path).convert('L')
                else:
                    # For ISIC: original format
                    gt_name = "ISIC_" + sample_id + "_Segmentation.png"
                    gt_img_path = os.path.join(gt_path, gt_name)
                    if not os.path.exists(gt_img_path):
                        print(f"Warning: Ground truth not found for {sample_id}, skipping...")
                        continue
                    gt = Image.open(gt_img_path).convert('L')
                
                gt = torchvision.transforms.PILToTensor()(gt)
                gt = torchvision.transforms.Resize((pred.shape[2], pred.shape[3]))(gt)
                gt = torch.unsqueeze(gt, 0).float() / 255.0
                
                # Evaluate segmentation
                iou_val, dice_val = eval_seg(pred, gt)
                all_iou.append(iou_val)
                all_dice.append(dice_val)
                
                # Get classification info if available
                cls_pred = None
                cls_true = None
                cls_prob = None
                
                if cls_results is not None:
                    # Try to find classification results for this sample
                    try:
                        # Look for sample in classification results
                        sample_row = cls_results[cls_results['sample_id'].str.contains(sample_id.split('_')[-1], na=False)]
                        if not sample_row.empty:
                            cls_pred = sample_row['prediction'].iloc[0]
                            cls_true = sample_row['true_label'].iloc[0]
                            cls_prob = sample_row['probability'].iloc[0]
                            all_cls_pred.append(cls_pred)
                            all_cls_true.append(cls_true)
                            all_cls_probs.append(cls_prob)
                    except Exception as e:
                        print(f"Warning: Could not load classification results for {sample_id}: {e}")
                
                # Create comparison visualization
                try:
                    # Load original image for comparison (if available)
                    original_img = None
                    if args.data_name == 'BRATS':
                        # Try to find original image
                        orig_candidates = [
                            f"{sample_id}_original.jpg",
                            f"{sample_id}_input.jpg",
                            f"{sample_id}_image.jpg"
                        ]
                        
                        for candidate in orig_candidates:
                            candidate_path = os.path.join(gt_path, candidate)
                            if os.path.exists(candidate_path):
                                original_img = Image.open(candidate_path).convert('L')
                                original_img = torchvision.transforms.PILToTensor()(original_img)
                                original_img = torchvision.transforms.Resize((pred.shape[2], pred.shape[3]))(original_img)
                                original_img = torch.unsqueeze(original_img, 0).float() / 255.0
                                break
                    
                    if original_img is None:
                        # Use ground truth as placeholder for original
                        original_img = gt
                    
                    comparison_path = create_comparison_visualization(
                        pred, gt, original_img, sample_id, args.comparison_dir,
                        cls_pred, cls_true, cls_prob
                    )
                    print(f"  Comparison saved: {comparison_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not create comparison for {sample_id}: {e}")
                
                # Update running totals
                mix_res = tuple([sum(a) for a in zip(mix_res, (iou_val, dice_val))])
                
                print(f"  IoU: {iou_val:.4f}, Dice: {dice_val:.4f}")
                if cls_pred is not None:
                    grade_names = {0: 'LGG', 1: 'HGG'}
                    print(f"  Classification: True={grade_names.get(cls_true, 'Unknown')}, Pred={grade_names.get(cls_pred, 'Unknown')}, Prob={cls_prob:.3f}")
                print("-" * 40)
    
    if num == 0:
        print("No prediction files found!")
        return
    
    # Calculate final metrics
    mean_iou, mean_dice = tuple([a/num for a in mix_res])
    std_iou = np.std(all_iou)
    std_dice = np.std(all_dice)
    
    print("=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples evaluated: {num}")
    print(f"Segmentation Metrics:")
    print(f"  Mean IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    
    # Classification metrics
    if len(all_cls_pred) > 0:
        print(f"\nClassification Metrics:")
        acc = accuracy_score(all_cls_true, all_cls_pred)
        f1 = f1_score(all_cls_true, all_cls_pred, average='binary')
        try:
            auc = roc_auc_score(all_cls_true, all_cls_probs)
        except:
            auc = 0.5
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC:      {auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_cls_true, all_cls_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    print("=" * 60)
    print(f"Comparison visualizations saved to: {args.comparison_dir}")
    
    # Save detailed results to JSON
    results = {
        'segmentation': {
            'mean_iou': float(mean_iou),
            'std_iou': float(std_iou),
            'mean_dice': float(mean_dice),
            'std_dice': float(std_dice),
            'individual_iou': all_iou,
            'individual_dice': all_dice
        }
    }
    
    if len(all_cls_pred) > 0:
        results['classification'] = {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'auc': float(auc),
            'confusion_matrix': cm.tolist(),
            'predictions': all_cls_pred,
            'true_labels': all_cls_true,
            'probabilities': all_cls_probs
        }
    
    results_file = os.path.join(args.comparison_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
