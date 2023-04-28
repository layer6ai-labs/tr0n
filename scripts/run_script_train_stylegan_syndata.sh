python train.py --exp_name='gmm_syn_ffhq_stylegan1000000_z_norml_test_10ep_16bs_1e4lr_cosine_0wd_skip_10mix_1e4_02noise' \
        --dataset_name='synthetic_stylegan' --dec_type='stylegan' --stylegan_gen='sg2-ffhq-1024' \
        --num_epochs=10 --save_every=10 --log_step=100 --eval_every=5 \
        --stylegan_eval_mode='test' \
        --batch_size=16 --train_loss='nll_latent' --val_loss='cosine_latent' --lr=1e-4 --optim='adam' \
        --scheduler='cosine' --weight_decay=0.0 --synthetic_latents_path='syn_ffhq_stylegan1000000_z_norml' \
        --with_gmm --num_mixtures=10 --alpha_nll=1e-4 \
        --add_clip_noise --clip_noise_scale=0.2
