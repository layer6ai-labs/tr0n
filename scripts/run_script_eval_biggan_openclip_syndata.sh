python eval.py --exp_name='hybrid_eval_50aug_gmm_syn_biggan1000000_z_norml_y_cat_fixed_laionclip_10ep_16bs_1e4lr_cosine_0wd_skip_10mix_1e4_05noise_100iter_z_sgld_5lr_adam_5e3lr_001noise_099mo_0wd' \
        --dataset_name='synthetic_biggan' --dec_type='biggan' --biggan_resolution=512 --z_thresh=2. \
        --val_image_dir='data/coco/val2014' --val_ann_file='data/coco/annotations/captions_val2014.json' \
        --batch_size=1 --val_loss='aug_cosine_latent' --times_augment_pred_image=50 --load_epoch=-1 \
        --with_gmm --num_mixtures=10 \
        --with_open_clip --clip_arch='ViT-B-32' --clip_pretrained='laion2b_s34b_b79k' \
        --load_exp_name='gmm_syn_biggan1000000_z_norml_y_cat_fixed_laionclip_10ep_16bs_1e4lr_cosine_0wd_skip_10mix_1e4_05noise' \
        --lr_latent=5.0 --optim_latent='sgld' --lr=5e-3 --optim='adam' --momentum=0.99 --weight_decay=0.0 --sgld_noise_std=0.01 --num_hybrid_iters=100
