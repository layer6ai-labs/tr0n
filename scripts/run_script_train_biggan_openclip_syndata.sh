python train.py --exp_name='gmm_syn_biggan1000000_z_norml_y_cat_fixed_laionclip_10ep_16bs_1e4lr_cosine_0wd_skip_10mix_1e4_05noise' \
        --dataset_name='synthetic_biggan' --dec_type='biggan' --biggan_resolution=512 --z_thresh=2. \
        --num_epochs=10 --save_every=10 --log_step=100 --eval_every=5 \
        --val_image_dir='data/coco/val2014' --val_ann_file='data/coco/annotations/captions_val2014.json' \
        --batch_size=16 --train_loss='nll_latent' --val_loss='cosine_latent' --lr=1e-4 --optim='adam' \
        --scheduler='cosine' --weight_decay=0.0 --synthetic_latents_path='syn_biggan1000000_z_norml_y_cat_fixed_laionclip' \
        --with_gmm --num_mixtures=10 --alpha_nll=1e-4 \
        --add_clip_noise --clip_noise_scale=0.5 \
        --with_open_clip --clip_arch='ViT-B-32' --clip_pretrained='laion2b_s34b_b79k'
