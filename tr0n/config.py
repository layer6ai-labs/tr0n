import os
import sys
from pathlib import Path
code_folder = Path(__file__).parent
sys.path.append(str(code_folder/"modules/BigGAN_utils"))
sys.path.append(str(code_folder/"modules/BLIP_utils"))
sys.path.append(str(code_folder/"modules/StyleGAN_utils"))
sys.path.append(str(code_folder/"modules/NVAE_utils"))
sys.path.append(str(code_folder/"modules/sgxl_utils"))
from argparse import ArgumentParser
from tr0n.modules.utils.basic_utils import mkdir, deletedir
import tr0n.modules.BigGAN_utils.utils as utils

def parse_args(keep_prev=False, is_demo=False):
    parser = utils.prepare_parser()

    # data parameters
    parser.add_argument('--dataset_name', type=str, default='coco', help='Dataset name')
    parser.add_argument('--train_image_dir', type=str, default='data/coco/train2014', help='Directory containing train images')
    parser.add_argument('--train_ann_file', type=str, default='data/coco/annotations/captions_train2014.json', help='Training annotation file')
    parser.add_argument('--val_image_dir', type=str, default='data/coco/val2014', help='Directory containing val images')
    parser.add_argument('--val_ann_file', type=str, default='data/coco/annotations/captions_val2014.json', help='Validation annotation file')
    parser.add_argument('--clip_image_size', type=int, default=224, help='Size to transform images')

    # experiment parameters
    parser.add_argument('--exp_name', type=str, required=not is_demo, help="Name of the current experiment")
    parser.add_argument('--load_exp_name', type=str, help="Name of the experiment to load from if different than the current one")
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_every', type=int, default=1, help="Save model every n epochs")
    parser.add_argument('--log_step', type=int, default=10, help="Print training log every n steps")
    parser.add_argument('--eval_every', type=int, default=1, help="Evaluate model every n epochs")
    parser.add_argument('--load_epoch', type=int, help="Epoch to load from exp_name or load_exp_name, or -1 to load model_best.pth")    
    parser.add_argument('--dec_type', type=str, default='biggan', help='the decoder architecture to use') # [biggan|stylegan|nvae]
    parser.add_argument('--generating_syn_data', action='store_true', default=False)

    # clip parameters
    parser.add_argument('--clip_arch', type=str, default='ViT-B/32', help='CLIP arch')
    parser.add_argument('--with_open_clip', action='store_true', default=False, help='Whether to use CLIP implementation from Open CLIP')
    parser.add_argument('--clip_pretrained', type=str, default='openai', help='CLIP pretrained model when using Open CLIP')
    parser.add_argument('--interp_mode', type=str, default='bilinear')
    parser.add_argument('--add_clip_noise', action='store_true', default=False)
    parser.add_argument('--clip_noise_scale', type=float, default=1., help='Amount to scale noise added to clip latents')
    parser.add_argument('--with_covariance', action='store_true', default=False)
    parser.add_argument('--with_adaptive_noise', action='store_true', default=False)
    parser.add_argument('--alpha_aesthetic', type=float, default=1., help='Weight for aesthetic loss')
 
    # decoder parameters
    parser.add_argument('--biggan_resolution', type=int, default=512, help='BigGAN resolution')
    parser.add_argument('--stylegan_gen', type=str, default='sg2-ffhq-1024', help='StyleGAN pre-trained generator name')
    parser.add_argument('--nvae_model', type=str, default='ffhq', help='NVAE pre-trained model name')
    parser.add_argument('--z_thresh', type=float, default=2., help='max/min threshold for BigGAN latent')
    parser.add_argument('--z_dist_type', type=str, default='truncate') # [truncate|normal|uniform]
    parser.add_argument('--y_dist_type', type=str, default='categorical') #[categorical|normal]
    parser.add_argument('--z_truncation', type=float, default=0.4)
    
    # training parameters
    parser.add_argument('--train_loss', type=str, default='mse_latent', help='Loss function to train the translator with')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train the translator')
    parser.add_argument('--optim', type=str, default='sgd', help='Optimizer to train with')
    parser.add_argument('--scheduler', type=str, default='cosine', help='LR scheduler')
    parser.add_argument('--weight_decay', type=float, default=0., help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0., help='Momentum for SGD')
    parser.add_argument('--max_grad_norm', type=float, default=1., help='Max gradient norm for clipping')
    
    # eval parameters
    parser.add_argument('--val_loss', type=str, default='cosine_latent', help='Loss (energy) function for Langevin dynamics')
    parser.add_argument('--times_augment_pred_image', type=int, default=0, help='Number of augmentations to run for a generated image')
    parser.add_argument('--num_hybrid_iters', type=int, default=0, help='Number of iterations to perform Langevin dynamics for')
    parser.add_argument('--lr_latent', type=float, default=1e-3, help='LR for Langevin dynamics')
    parser.add_argument('--optim_latent', type=str, default='sgld', help='Optimizer for Langevin dynamics')
    parser.add_argument('--sgld_noise_std', type=float, help='Langevin dynamics noise std')
    parser.add_argument('--stylegan_eval_mode', type=str, default='eval', help='Mode for StyleGAN eval') #[eval|test|landscape]
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to generate')
    parser.add_argument('--num_interps', type=int, default=0, help='Number of interpolations to generate')

    # gmm paramters
    parser.add_argument('--with_gmm', action='store_true', default=False, help='Use GMM')
    parser.add_argument('--num_mixtures', type=int, default=1, help='Number of mixtures for GMM')
    parser.add_argument('--alpha_nll', type=float, default=1., help='Weight for NLL loss in GMM')
    parser.add_argument('--gumbel_softmax_temp', type=float, default=1., help='Temperature of Gumbel-Softmax when reparam sampling from GMM')
    
    # synthetic data
    parser.add_argument('--datasize_syn', type=int, default=10000, help='number of data points to generate')
    parser.add_argument('--synthetic_latents_path', type=str, default=None, help='path name for synthetic latents')
    parser.add_argument('--with_caption_model', action='store_true', default=False, help='Use caption model to generate pairs')
    parser.add_argument('--with_nucleus_sampling', action='store_true', default=False, help='Use nucleus sampling with caption model')
    
    # system parameters
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=24, help='Random seed')
    parser.add_argument('--no_tensorboard', action='store_true', default=False)
    parser.add_argument('--tb_log_dir', type=str, default='logs')

    args = parser.parse_args()

    if is_demo:
        return args

    if not args.generating_syn_data: # we're not generating synthetic data
        args.model_path = os.path.join(args.output_dir, args.exp_name)
        if args.load_exp_name is not None:
            args.load_path = os.path.join(args.output_dir, args.load_exp_name)
        else:
            args.load_path = args.model_path
        args.save_image_path = os.path.join(args.model_path, 'generated_images')
        args.tb_log_dir = os.path.join(args.tb_log_dir, args.exp_name)

        if not keep_prev:
            deletedir(args.model_path)
            mkdir(args.model_path)
            deletedir(args.tb_log_dir)
            mkdir(args.tb_log_dir)

    return args
