python generate_synthetic_latent_codes.py --exp_name='syn_biggan' --dataset_name='random_gen_biggan' --generating_syn_data \
        --dec_type='biggan' --biggan_resolution=512 --z_thresh=2. --z_dist_type='normal' --y_dist_type='categorical_fixed' \
        --batch_size=16 --datasize_syn=1000000 --synthetic_latents_path='syn_biggan1000000_z_norml_y_cat_fixed_cap_model_nuc' \
        --with_caption_model --with_nucleus_sampling
