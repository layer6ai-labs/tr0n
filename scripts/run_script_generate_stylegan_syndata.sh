python generate_synthetic_latent_codes.py --exp_name='syn_stylegan' --dataset_name='random_gen_stylegan' --generating_syn_data \
        --dec_type='stylegan' --stylegan_gen='sg2-ffhq-1024' --z_dist_type='normal' \
        --batch_size=16 --datasize_syn=1000000 --synthetic_latents_path='syn_ffhq_stylegan1000000_z_norml'
