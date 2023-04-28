python generate_synthetic_latent_codes.py --exp_name='syn_nvae' --dataset_name='random_gen_nvae' --generating_syn_data \
        --dec_type='nvae' --nvae_model='ffhq' --z_truncation=0.6 \
        --batch_size=2 --datasize_syn=1000000 --synthetic_latents_path='syn_ffhq_nvae1000000_z_temp_06_mean7'
