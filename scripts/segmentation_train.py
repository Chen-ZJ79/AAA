
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
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

    # ==================== Classification Support ====================
    use_cls_head = args.use_cls_head if hasattr(args, 'use_cls_head') else False
    csv_path = args.csv_path if hasattr(args, 'csv_path') else None
    class_weights = None
    # ============================================================

    if args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_train = transforms.Compose(tran_list)

        # Create dataset with classification support
        ds = BRATSDataset3D(
            args.data_dir, 
            transform_train, 
            test_flag=False,
            csv_path=csv_path,
            use_cls_head=use_cls_head
        )
        args.in_ch = 5
        
        # Get class weights from dataset
        if use_cls_head and hasattr(ds, 'class_weights'):
            class_weights = ds.class_weights
            
            # Calculate and print class distribution
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
                
    else:
        raise ValueError(f"Unsupported data_name: {args.data_name}. Only 'BRATS' is supported.")
        
    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
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
