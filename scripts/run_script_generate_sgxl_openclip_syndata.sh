python generate_synthetic_latent_codes.py --exp_name='syn_sgxl' --dataset_name='random_gen_sgxl' --generating_syn_data \
        --dec_type='sgxl' --stylegan_gen='sgxl-imagenet-512' --z_dist_type='normal' --y_dist_type='categorical_fixed' \
        --batch_size=16 --datasize_syn=1000000 --synthetic_latents_path='syn_imagenet_sgxl1000000_z_norml_blip2_opt27B' \
        --with_open_clip --clip_arch='ViT-B-32' --clip_pretrained='laion2b_s34b_b79k' \
        --with_caption_model --with_nucleus_sampling
