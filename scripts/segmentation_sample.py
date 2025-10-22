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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def has_tumor_slice(mask, threshold=0.001):
    """Check if slice contains tumor (non-background) regions"""
    # Count non-zero pixels in mask
    non_zero_pixels = th.sum(mask > 0).float()
    total_pixels = mask.numel()
    tumor_ratio = non_zero_pixels / total_pixels
    return tumor_ratio > threshold

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def calculate_dice_iou(pred, target):
    """Calculate Dice and IoU for binary segmentation (background vs tumor)"""
    # 确保输入是numpy数组
    if not isinstance(pred, np.ndarray):
        pred = pred.numpy()
    if not isinstance(target, np.ndarray):
        target = target.numpy()
    
    # 转换为二分类：0=背景，1=肿瘤
    pred_binary = (pred > 0).astype(np.float32)
    target_binary = (target > 0).astype(np.float32)
    
    # 计算交集和并集
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary)
    
    # 计算Dice和IoU
    if union > 0:
        dice = 2 * intersection / union
        iou = intersection / (union - intersection + 1e-8)
    else:
        dice = 1.0 if intersection == 0 else 0.0
        iou = 1.0 if intersection == 0 else 0.0
    
    return float(dice), float(iou)

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        # Use validation split for evaluation (with masks and grades)
        ds = BRATSDataset3D(args.data_dir, transform_test, csv_path=args.csv_path, use_cls_head=True, split_mode='validation')
        args.in_ch = 5
    else:
        raise ValueError(f"Unsupported data_name: {args.data_name}. Only 'BRATS' is supported.")

    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)  # Don't shuffle for evaluation
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Enable classification head
    model.use_cls_head = True
    
    all_images = []
    all_seg_preds = []
    all_seg_targets = []
    all_cls_preds = []
    all_cls_targets = []
    all_cls_probs = []

    logger.log(f"Loading model from {args.model_path}")
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    
    logger.log(f"Starting evaluation on validation set (case-wise analysis)...")
    
    # 按病例组织数据
    case_results = {}
    processed_cases = 0
    target_cases = args.num_eval_cases  # 评估的病例数量
    
    with th.no_grad():
        for batch_idx in range(len(datal)):
            if args.data_name == 'BRATS' and args.use_cls_head:
                b, m, grade_label, path = next(data)  # Get grade label
            else:
                b, m, path = next(data)  # Original format
                grade_label = None
            
            # Skip empty slices (background only)
            if not has_tumor_slice(m):
                # logger.log(f"Skipping empty slice: {path[0]}")
                continue
            
            # 提取病例ID并清理特殊字符
            if args.data_name == 'BRATS':
                # 路径格式: Data/BraTS/MICCAI_BraTS2020_TrainingData/BraTS20_Training_004/BraTS20_Training_004_t1_slice58.nii
                # 需要提取: BraTS20_Training_004
                path_parts = path[0].split('/')
                
                # 找到包含BraTS20_Training的目录
                case_id = None
                for part in path_parts:
                    if 'BraTS20_Training' in part and not part.endswith('.nii'):
                        case_id = part
                        break
                
                if case_id is None:
                    # 备用方案：从文件名提取
                    filename = path[0].split('/')[-1]
                    if 'BraTS20_Training' in filename:
                        # 提取到slice之前的部分
                        case_id = filename.split('_slice')[0].split('_t')[0]
                    else:
                        case_id = filename.split('_slice')[0]
            else:
                case_id = path[0].split('_')[0]
            
            # 清理病例ID中的特殊字符
            case_id = case_id.replace('/', '_').replace('\\', '_').replace(':', '_')
            
            # 初始化病例结果
            if case_id not in case_results:
                case_results[case_id] = {
                    'grade_label': grade_label.item() if grade_label is not None else None,
                    'slices': [],
                    'seg_predictions': [],
                    'seg_targets': [],
                    'cls_predictions': [],
                    'cls_probs': []
                }
                processed_cases += 1
                logger.log(f"Processing case {processed_cases}/{target_cases}: {case_id}")
            
            if processed_cases > target_cases:
                break
            
            c = th.randn_like(b[:, :1, ...])
            img = th.cat((b, c), dim=1)     #add a noise channel
            
            # 创建切片ID
            if args.data_name == 'BRATS':
                # 从路径中提取切片号
                slice_id_raw = path[0].split('slice')[-1].split('.nii')[0]
                slice_id_clean = slice_id_raw.replace('/', '_').replace('\\', '_').replace(':', '_')
                slice_ID = f"{case_id}_{slice_id_clean}"
            # 从路径中提取实际切片序号和简化病例ID
            slice_number = path[0].split('slice')[-1].split('.nii')[0]
            # 提取简化的病例ID (如 004, 007)
            simple_case_id = case_id.split('_')[-1] if '_' in case_id else case_id
            
            logger.log(f"Processing slice {slice_number} of case {simple_case_id}: {slice_ID}")

            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            enslist = []

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
                print(f'Time for sample {i+1}: {start.elapsed_time(end):.2f}ms')

                co = th.tensor(cal_out)
                if args.version == 'new':
                    enslist.append(sample[:,-1,:,:])
                else:
                    enslist.append(co)

                if args.debug:
                    if args.data_name == 'BRATS':
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
            
            # Ensemble result
            ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
            vutils.save_image(ensres, fp = os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".jpg"), nrow = 1, padding = 10)
            
            # Store predictions for this case
            case_results[case_id]['slices'].append(slice_ID)
            case_results[case_id]['seg_predictions'].append(ensres.cpu().numpy())
            case_results[case_id]['seg_targets'].append(m.cpu().numpy())
            
            # Classification prediction (use clean input at t=0)
            if args.use_cls_head and grade_label is not None:
                try:
                    # Use clean input for classification - ensure GPU placement and correct dtype
                    clean_img = th.cat((b, th.zeros_like(b[:, :1, ...])), dim=1).to(dist_util.dev()).float()
                    t_zeros = th.zeros(clean_img.shape[0], device=clean_img.device, dtype=th.float32)
                    model_output = model(clean_img, t_zeros)
                    
                    if len(model_output) == 3:
                        _, cls_logits, _ = model_output
                        cls_probs = th.softmax(cls_logits, dim=1)
                        pred_label = th.argmax(cls_logits, dim=1)
                        
                        case_results[case_id]['cls_predictions'].append(pred_label.item())
                        case_results[case_id]['cls_probs'].append(cls_probs[0].cpu().numpy())
                        
                        logger.log(f"Case {case_id} Slice {len(case_results[case_id]['slices'])}: Pred={pred_label.item()}, True={grade_label.item()}, Prob={cls_probs[0].cpu().numpy()}")
                except Exception as e:
                    logger.log(f"Classification error for case {case_id} slice {len(case_results[case_id]['slices'])}: {e}")
                    case_results[case_id]['cls_predictions'].append(-1)  # 标记错误
                    case_results[case_id]['cls_probs'].append([0.5, 0.5])  # 默认概率

    # Calculate final metrics
    logger.log("="*60)
    logger.log("FINAL EVALUATION RESULTS")
    logger.log(f"Processed {len(case_results)} cases")
    logger.log("="*60)
    
    # 按病例计算指标
    all_seg_preds = []
    all_seg_targets = []
    all_cls_preds = []
    all_cls_targets = []
    all_cls_probs = []
    
    for case_id, case_data in case_results.items():
        logger.log(f"Case {case_id}: {len(case_data['slices'])} slices, Grade={case_data['grade_label']}")
        
        # 收集该病例的所有切片数据
        all_seg_preds.extend(case_data['seg_predictions'])
        all_seg_targets.extend(case_data['seg_targets'])
        
        # 分类：每个病例一个结果（取第一个切片的预测）
        if case_data['cls_predictions'] and case_data['cls_predictions'][0] != -1:
            all_cls_preds.append(case_data['cls_predictions'][0])
            all_cls_targets.append(case_data['grade_label'])
            all_cls_probs.append(case_data['cls_probs'][0])
    
    # Segmentation metrics
    if all_seg_preds:
        all_seg_preds = np.concatenate(all_seg_preds, axis=0)
        all_seg_targets = np.concatenate(all_seg_targets, axis=0)
        
        # 逐个切片计算Dice和IoU，然后取平均
        dice_scores = []
        iou_scores = []
        for i in range(len(all_seg_preds)):
            dice, iou = calculate_dice_iou(all_seg_preds[i], all_seg_targets[i])
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        
        logger.log(f"Segmentation Metrics:")
        logger.log(f"  Dice Score: {mean_dice:.4f}")
        logger.log(f"  IoU Score: {mean_iou:.4f}")
        logger.log(f"  Total slices evaluated: {len(all_seg_preds)}")
    
    # Classification metrics
    if all_cls_preds and args.use_cls_head:
        all_cls_preds = np.array(all_cls_preds)
        all_cls_targets = np.array(all_cls_targets)
        all_cls_probs = np.array(all_cls_probs)
        
        acc = accuracy_score(all_cls_targets, all_cls_preds)
        f1 = f1_score(all_cls_targets, all_cls_preds, average='binary')
        
        try:
            auc = roc_auc_score(all_cls_targets, all_cls_probs[:, 1])
        except:
            auc = 0.5
            
        cm = confusion_matrix(all_cls_targets, all_cls_preds)
        
        logger.log(f"Classification Metrics:")
        logger.log(f"  Accuracy: {acc:.4f}")
        logger.log(f"  F1 Score: {f1:.4f}")
        logger.log(f"  AUC Score: {auc:.4f}")
        logger.log(f"  Confusion Matrix:")
        logger.log(f"    {cm}")
        
        # Class distribution
        lgg_count = np.sum(all_cls_targets == 0)
        hgg_count = np.sum(all_cls_targets == 1)
        total = len(all_cls_targets)
        logger.log(f"  Test Set Distribution:")
        logger.log(f"    LGG: {lgg_count} ({100.0 * lgg_count / total:.1f}%)")
        logger.log(f"    HGG: {hgg_count} ({100.0 * hgg_count / total:.1f}%)")
        logger.log(f"    Total cases evaluated: {total}")
    
    logger.log("="*60)
    logger.log("Evaluation completed!")

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
        use_cls_head = True,
        csv_path = "",
        num_eval_cases = 20
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
