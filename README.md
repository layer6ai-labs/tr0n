<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p> 

<div align="center">
<h1>
<b>
TR0N: Translator Networks for 0-Shot Plug-and-Play Conditional Generation
</b>
</h1>
  
<p align="center">
  <a href='https://arxiv.org/abs/2304.13742'><img src='https://img.shields.io/badge/arXiv-2304.13742-b31b1b.svg' /></a>
  <a href='https://huggingface.co/spaces/Layer6/TR0N'><img src='https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-orange' /></a>
</p>
  
<h4>
<b>
<a href="https://www.linkedin.com/in/zhaoyan-liu-9309aa180/">Zhaoyan Liu*</a>, <a href="https://www.cs.toronto.edu/~nvouitsis/">Noël Vouitsis*</a>, <a href="https://www.cs.toronto.edu/~satyag/">Satya Krishna Gorti</a>, <a href="https://jimmylba.github.io/">Jimmy Ba</a>, Gabriel Loaiza-Ganem    
</b>
</h4>
</div>

<a name="intro"/>

## Introduction
This repository contains the official implementation of our **ICML 2023** paper. It includes both training and sampling code.

<a name="install"/>

## Installation
```bash
git clone https://github.com/layer6ai-labs/tr0n.git
cd tr0n
pip install --upgrade pip
pip install -r requirements.txt
```

<a name="dwnld"/>

## Downloads

For BigGAN, you must first download its pre-trained weights from this <a href="https://drive.google.com/drive/folders/1nJ3HmgYgeA9NZr-oU-enqbYeO7zBaANs?usp=sharing">link</a>, and move them to `tr0n/modules/BigGAN_utils/weights/`. Similarly, for NVAE, you must download any pre-trained checkpoint from this <a href="https://drive.google.com/drive/folders/1KVpw12AzdVjvbfEYM_6_3sxTy93wWkbe">link</a>, and move it to `tr0n/modules/NVAE_utils/ckpt/{model_name}/`. For NVAE models, this repo only supports FFHQ and CelebA256 NVAE models. We also note that we updated the pre-trained NVAE weights of the batch norm layers for sampling as suggested in the NVAE paper. For all StyleGAN* models, the checkpoints are automatically downloaded in this repo. Finally, for all FID experiments, you must also download MS-COCO 2014 from this <a href="https://cocodataset.org/#download">link</a>. 

<a name="usage"/>

## Usage

Scripts to run TR0N with the default hyperparameters listed in our paper's appendix can be found under `scripts/`. These scripts can also be modified for experimental usage. All arguments are listed in our config at `tr0n/config.py`.

<a name="train"/>
  
### TR0N training

The first step is to generate a synthetic dataset of latent/condition pairs with `scripts/run_script_generate_*`. Then, you can train a translator network on said pairs with `scripts/run_script_train_*`.

For text conditioning, we propose two options to generate synthetic data, as mentioned in our paper. The first involves using a caption model followed by CLIP's text encoder. With this choice, the same CLIP encoder is used both to train the translator network and to obtain conditions, which we find empirically produces higher quality images. This choice can be enabled by adding `--with_caption_model` (and optionally `--with_nucleus_sampling`) to the synthetic dataset generation script. We note that this choice is best suited (and only supported) for natural image generation, not facial images. The second option for synthetic data generation is to use CLIP's image encoder; to address the modality gap in CLIP space and simulate "pseudo-text embeddings", we add Gaussian noise to the CLIP image embeddings during translator training (see details in our paper). This choice can be enabled by adding `--add_clip_noise --clip_noise_scale={noise_std}` to the training script (we set `noise_std=0.2` for CLIP).

For image semantic conditioning, we train the translator network directly on the CLIP image embeddings (i.e no added noise) since the CLIP image encoder is also used to obtain conditions.

<a name="sample"/>

### TR0N sampling

For sampling, run `scripts/run_script_eval_*`. Specifying the argument `--dataset_name` as either `text` or `image` allows for text or image semantic conditioning respectively. Note that the translator used to initialize TR0N sampling in both cases would differ, as noted in the appendix of our paper, so ensure to specify the intended translator checkpoint to use for sampling in the `--load_exp_name` argument. If you specify text conditioning, the text prompts to condition with are taken from `tr0n/modules/datasets/text_dataset.py` (the source of our natural image prompts can be found at this <a href="https://docs.google.com/spreadsheets/d/18yN-E2sESmpnwmOgw2atWxpoA45LNkBRQkNANGvkFww/edit#gid=0">link</a>). If you specify image semantic conditioning, you must also specify a folder containing the images to condition with using the `--val_image_dir` argument. Moreover, specifying a positive value to the argument `--num_interps` generates semantic interpolations for two images.

We note that $\sqrt{\lambda}$ in our paper is specified by the argument `--sgld_noise_std`, $\frac{\beta \lambda}{2}$ is specified by `--lr_latent` and $T$ is specified by `--num_hybrid_iters`.

<a name="cite"/>

## Citation

If you find this work useful in your research, please cite the following paper:

```
@inproceedings{liu2023tr0n,
  title={TR0N: Translator Networks for 0-Shot Plug-and-Play Conditional Generation},
  author={Liu, Zhaoyan and Vouitsis, No{\"e}l and Gorti, Satya Krishna and Ba, Jimmy and Loaiza-Ganem, Gabriel},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2023}
}
```
