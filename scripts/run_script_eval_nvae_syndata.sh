python eval.py --exp_name='hybrid_text_eval_50aug_gmm_syn_ffhq_nvae1000000_z_temp_06_mean7_10ep_16bs_1e4lr_cosine_0wd_skip_10mix_1e4_02noise_100iter_w_sgld_1e1lr_adam_5e3lr_001noise_099mo_0wd' \
        --dataset_name='text' --dec_type='nvae' --nvae_model='ffhq' --z_truncation=0.6 \
        --batch_size=1 --val_loss='aug_cosine_latent' --times_augment_pred_image=50 --load_epoch=-1 \
        --with_gmm --num_mixtures=10 \
        --load_exp_name='gmm_syn_ffhq_nvae1000000_z_temp_06_mean7_10ep_16bs_1e4lr_cosine_0wd_skip_10mix_1e4_02noise' \
        --lr_latent=1e-1 --optim_latent='sgld' --lr=5e-3 --optim='adam' --momentum=0.99 --weight_decay=0.0 --sgld_noise_std=0.01 --num_hybrid_iters=100
