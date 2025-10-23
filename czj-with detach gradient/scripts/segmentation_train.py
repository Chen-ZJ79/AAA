
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
from guided_diffusion.custom_dataset_loader import CustomDataset,CustomDataset3D
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from pathlib import Path
from guided_diffusion.train_util import TrainLoop
# from visdom import Visdom
# viz = Visdom(port=8850)
import torchvision.transforms as transforms

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    # ==================== Classification ====================
    use_cls_head = args.use_cls_head if hasattr(args, 'use_cls_head') else False
    csv_path = args.csv_path if hasattr(args, 'csv_path') else None
    class_weights = None

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(
            args.data_dir, 
            transform_train, 
            test_flag=False,#是否把标签加入到数据集中
            csv_path=csv_path,
            use_cls_head=use_cls_head
        )
        args.in_ch = 5
        
        # Get class weights
        if use_cls_head and hasattr(ds, 'class_weights'):
            class_weights = ds.class_weights
            
            # class distribution
            if hasattr(ds, 'grade_mapping'):
                from collections import Counter
                grade_counts = Counter(ds.grade_mapping.values())
                n_lgg = grade_counts[0]
                n_hgg = grade_counts[1]
                total = n_lgg + n_hgg
                
                logger.log("=" * 60)
                logger.log("Dataset Statistics:")
                logger.log(f"  Total samples: {total}")
                logger.log(f"  LGG samples: {n_lgg} ({100.0 * n_lgg / total:.2f}%)")
                logger.log(f"  HGG samples: {n_hgg} ({100.0 * n_hgg / total:.2f}%)")
                logger.log(f"  Alpha_LGG: {class_weights[0]:.4f}")
                logger.log(f"  Alpha_HGG: {class_weights[1]:.4f}")
                logger.log("=" * 60)
                
    elif any(Path(args.data_dir).glob("**/*.nii.gz")):
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset3D(args, args.data_dir, transform_train)
        args.in_ch = 4
    else:
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ",args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
        
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,           # Use 4 workers for parallel data loading
        pin_memory=True,         # Speed up CPU to GPU transfer
        prefetch_factor=2,       # Prefetch 2 batches per worker
        persistent_workers=True  # Keep workers alive between epochs
    )
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Enable classification head in model
    if use_cls_head:
        if hasattr(model, 'use_cls_head'):
            model.use_cls_head = True
            logger.log("Classification head enabled in model")
        elif hasattr(model, 'module') and hasattr(model.module, 'use_cls_head'):
            model.module.use_cls_head = True
            logger.log("Classification head enabled in model (DataParallel)")
    
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
        # Enable cls_head after DataParallel wrapping
        if use_cls_head and hasattr(model.module, 'use_cls_head'):
            model.module.use_cls_head = True
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)

    # ==================== Display Parameter Training Status ====================
    logger.log("="*80)
    logger.log("PARAMETER TRAINING STATUS")
    logger.log("="*80)
    
    trainable_params = []
    frozen_params = []
    trainable_count = 0
    frozen_count = 0
    trainable_numel = 0
    frozen_numel = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            trainable_count += 1
            trainable_numel += param.numel()
        else:
            frozen_params.append(name)
            frozen_count += 1
            frozen_numel += param.numel()
    
    # Display summary
    logger.log(f"\n📊 Summary:")
    logger.log(f"   Trainable parameters: {trainable_count} ({trainable_numel:,} elements)")
    logger.log(f"   Frozen parameters:    {frozen_count} ({frozen_numel:,} elements)")
    logger.log(f"   Total parameters:     {trainable_count + frozen_count} ({trainable_numel + frozen_numel:,} elements)")
    
    if trainable_numel + frozen_numel > 0:
        trainable_pct = 100.0 * trainable_numel / (trainable_numel + frozen_numel)
        logger.log(f"   Trainable percentage: {trainable_pct:.2f}%")
    
    # Display trainable parameters (grouped by module)
    if trainable_params:
        logger.log(f"\n✅ Trainable parameters ({len(trainable_params)}):")
        
        # Group by module prefix
        grouped_trainable = {}
        for name in trainable_params:
            # Extract module name (e.g., "input_blocks.0" from "input_blocks.0.0.weight")
            parts = name.split('.')
            if len(parts) >= 2:
                module_key = '.'.join(parts[:2])
            else:
                module_key = parts[0]
            
            if module_key not in grouped_trainable:
                grouped_trainable[module_key] = []
            grouped_trainable[module_key].append(name)
        
        # Display grouped
        for module_key in sorted(grouped_trainable.keys()):
            params_in_module = grouped_trainable[module_key]
            if len(params_in_module) <= 3:
                # Show all if few parameters
                for param_name in params_in_module:
                    logger.log(f"   - {param_name}")
            else:
                # Show first and last, and count
                logger.log(f"   - {module_key}.* ({len(params_in_module)} parameters)")
                logger.log(f"     ├─ {params_in_module[0]}")
                logger.log(f"     ├─ ...")
                logger.log(f"     └─ {params_in_module[-1]}")
    else:
        logger.log(f"\n❌ No trainable parameters!")
    
    # Display frozen parameters (only summary for brevity)
    if frozen_params:
        logger.log(f"\n🔒 Frozen parameters ({len(frozen_params)}):")
        
        # Group by module prefix
        grouped_frozen = {}
        for name in frozen_params:
            parts = name.split('.')
            if len(parts) >= 2:
                module_key = '.'.join(parts[:2])
            else:
                module_key = parts[0]
            
            if module_key not in grouped_frozen:
                grouped_frozen[module_key] = 0
            grouped_frozen[module_key] += 1
        
        # Display grouped summary
        for module_key in sorted(grouped_frozen.keys()):
            count = grouped_frozen[module_key]
            logger.log(f"   - {module_key}.* ({count} parameters)")
    
    # Check gradient detachment status for classification head
    if use_cls_head:
        logger.log(f"\n🔍 Classification Head Status:")
        if hasattr(model, 'cls_detach_features'):
            detach_status = model.cls_detach_features
        elif hasattr(model, 'module') and hasattr(model.module, 'cls_detach_features'):
            detach_status = model.module.cls_detach_features
        else:
            detach_status = None
        
        if detach_status is not None:
            if detach_status:
                logger.log(f"   ✅ Gradient detachment: ENABLED")
                logger.log(f"      → Classification gradients will NOT affect segmentation network")
                logger.log(f"      → Training mode: Stage 1 (Segmentation-focused)")
            else:
                logger.log(f"   ❌ Gradient detachment: DISABLED")
                logger.log(f"      → Classification gradients WILL affect segmentation network")
                logger.log(f"      → Training mode: Stage 3 (Joint fine-tuning)")
        else:
            logger.log(f"   ⚠️  Cannot determine gradient detachment status")
    
    logger.log("="*80)
    logger.log("")
    # ============================================================

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_cls_head=use_cls_head,
        class_weights=class_weights,
        focal_gamma=args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0,
        seg_focal_gamma=args.seg_focal_gamma if hasattr(args, 'seg_focal_gamma') else 1.5,
        seg_focal_lambda=args.seg_focal_lambda if hasattr(args, 'seg_focal_lambda') else 0.5,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_name = 'BRATS',
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = None, #"0,1,2"
        out_dir='./results/',
        # ==================== Classification parameters ====================
        use_cls_head=False,  # Enable classification head for HGG/LGG grading
        csv_path=None,  # Path to CSV file with grade labels
        focal_gamma=2.0,  # Gamma parameter for classification focal loss
        seg_focal_gamma=1.5,  # Gamma parameter for segmentation focal loss
        seg_focal_lambda=0.5,  # Weight for segmentation focal loss component
        # ============================================================
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
