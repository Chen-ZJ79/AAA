

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    # ==================== Classification Support ====================
    use_cls_head = args.use_cls_head if hasattr(args, 'use_cls_head') else False
    csv_path = args.csv_path if hasattr(args, 'csv_path') else None
    
    # Metrics accumulation
    all_dice = []
    all_iou = []
    all_cls_pred = []
    all_cls_true = []
    all_cls_probs = []
    # ============================================================

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(
            args.data_dir,
            transform_test,
            test_flag=False,  # Set to False to get ground truth masks
            csv_path=csv_path,
            use_cls_head=use_cls_head
        )
        args.in_ch = 5
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)

        ds = CustomDataset(args, args.data_dir, transform_test, mode = 'Test')
        args.in_ch = 4

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)  # No shuffle for consistent evaluation
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Enable classification head in model
    if use_cls_head:
        if hasattr(model, 'use_cls_head'):
            model.use_cls_head = True
            logger.log("Classification head enabled in model for evaluation")
    
    all_images = []


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    num_samples = min(len(data), args.num_eval_samples if hasattr(args, 'num_eval_samples') else len(data))
    logger.log(f"Evaluating on {num_samples} samples...")
    
    for sample_idx in range(num_samples):
        # Load data based on whether classification is enabled
        if use_cls_head:
            b, m, grade_label, path = next(data)
            grade_label = grade_label.to(dist_util.dev())
        else:
            b, m, path = next(data)
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        if args.data_name == 'ISIC':
            slice_ID=path[0].split("_")[-1].split('.')[0]
        elif args.data_name == 'BRATS':
            # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
            slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        
        # ==================== Classification Evaluation ====================
        # Get classification prediction once (not per ensemble)
        cls_pred_probs = None
        if use_cls_head:
            with th.no_grad():
                try:
                    # Create a clean input for classification (no noise)
                    clean_img = b.to(dist_util.dev())
                    # Add a dummy noise channel (zeros) for model input format
                    dummy_noise = th.zeros_like(clean_img[:, :1, ...])
                    clean_input = th.cat((clean_img, dummy_noise), dim=1)
                    # Use timestep=0 for clean prediction
                    t_zero = th.zeros(clean_img.shape[0], dtype=th.long, device=dist_util.dev())
                    
                    # Get model output
                    model_output = model(clean_input, t_zero)
                    if len(model_output) == 3:
                        _, cls_logits, _ = model_output
                        cls_pred_probs = th.softmax(cls_logits, dim=1)
                        cls_pred = th.argmax(cls_pred_probs, dim=1)
                        
                        # Store for metrics
                        all_cls_pred.append(cls_pred.cpu().numpy())
                        all_cls_true.append(grade_label.cpu().numpy())
                        all_cls_probs.append(cls_pred_probs[:, 1].cpu().numpy())
                        
                        logger.log(f"Sample {sample_idx}: True={grade_label.item()}, Pred={cls_pred.item()}, Prob_HGG={cls_pred_probs[0, 1].item():.3f}")
                    else:
                        logger.log(f"⚠️  Warning: Model did not return classification results (got {len(model_output)} outputs)")
                except Exception as e:
                    logger.log(f"❌ Classification prediction error: {e}")
                    import traceback
                    traceback.print_exc()
        # ============================================================

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            co = th.tensor(cal_out)
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
            else:
                enslist.append(co)

            if args.debug:
                # print('sample size is',sample.size())
                # print('org size is',org.size())
                # print('cal size is',cal.size())
                if args.data_name == 'ISIC':
                    # s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
                    o = th.tensor(org)[:,:-1,:,:]
                    c = th.tensor(cal).repeat(1, 3, 1, 1)
                    # co = co.repeat(1, 3, 1, 1)

                    s = sample[:,-1,:,:]
                    b,h,w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)

                    tup = (ss,o,c)
                elif args.data_name == 'BRATS':
                    s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
                    m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
                    o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
                    o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
                    o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
                    o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
                    c = th.tensor(cal)

                    tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

                compose = th.cat(tup,0)
                vutils.save_image(compose, fp = os.path.join(args.out_dir, str(slice_ID)+'_output'+str(i)+".jpg"), nrow = 1, padding = 10)
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
        vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)
        
        # ==================== Compute Segmentation Metrics ====================
        # Compute Dice and IoU for this sample
        with th.no_grad():
            pred_mask = (ensres > 0.5).float()
            gt_mask = m.to(dist_util.dev())
            gt_mask = th.where(gt_mask > 0, 1, 0).float()
            
            intersection = (pred_mask * gt_mask).sum()
            dice = (2.0 * intersection + 1e-5) / (pred_mask.sum() + gt_mask.sum() + 1e-5)
            union = pred_mask.sum() + gt_mask.sum() - intersection
            iou = (intersection + 1e-5) / (union + 1e-5)
            
            all_dice.append(dice.item())
            all_iou.append(iou.item())
            
            logger.log(f"Sample {sample_idx}: Dice={dice.item():.4f}, IoU={iou.item():.4f}")
        # ============================================================
    
    # ==================== Print Final Statistics ====================
    logger.log("\n" + "=" * 60)
    logger.log("EVALUATION RESULTS")
    logger.log("=" * 60)
    
    # Segmentation metrics
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    mean_iou = np.mean(all_iou)
    std_iou = np.std(all_iou)
    
    logger.log(f"Segmentation Metrics:")
    logger.log(f"  Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    logger.log(f"  IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    
    # Classification metrics
    if use_cls_head and len(all_cls_true) > 0:
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
        
        all_cls_pred = np.concatenate(all_cls_pred)
        all_cls_true = np.concatenate(all_cls_true)
        all_cls_probs = np.concatenate(all_cls_probs)
        
        acc = accuracy_score(all_cls_true, all_cls_pred)
        f1 = f1_score(all_cls_true, all_cls_pred, average='binary')
        try:
            auc = roc_auc_score(all_cls_true, all_cls_probs)
        except:
            auc = 0.5
        
        logger.log(f"\nClassification Metrics:")
        logger.log(f"  Accuracy: {acc:.4f}")
        logger.log(f"  F1 Score: {f1:.4f}")
        logger.log(f"  AUC:      {auc:.4f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_cls_true, all_cls_pred)
        logger.log(f"\nConfusion Matrix:")
        logger.log(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.log(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    logger.log("=" * 60)

def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",         #path to pretrain model
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results/',
        multi_gpu = None, #"0,1,2"
        debug = False,
        # ==================== Classification parameters ====================
        use_cls_head=False,  # Enable classification evaluation
        csv_path=None,  # Path to CSV file with grade labels
        num_eval_samples=100,  # Number of samples to evaluate (set to large number for all)
        # ============================================================
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
