from pathlib import Path                                                                      
module_folder = Path(__file__).parent.parent 
from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import tr0n.modules.BigGAN_utils.utils as utils
import clip
import open_clip
from tr0n.modules.utils.diff_augment import DiffAugment


def get_G(config, device):
    if config.biggan_resolution == 256:
        config = vars(config)

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
        config["resolution"] = 256
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "128"
        config["D_attn"] = "128"
        config["G_ch"] = 96
        config["D_ch"] = 96
        config["hier"] = True
        config["dim_z"] = 140
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = device

        # Set up cudnn.benchmark for free speed.
        #torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)

        # Load weights.
        weights_path = str(module_folder/"BigGAN_utils/weights/biggan-256.pth")
        G.load_state_dict(torch.load(weights_path), strict=False)
    elif config.biggan_resolution == 512:
        config = vars(config)

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs128x8.sh.
        config["resolution"] = 512
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "64"
        config["D_attn"] = "64"
        config["G_ch"] = 96
        config["D_ch"] = 64
        config["hier"] = True
        config["dim_z"] = 128
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = device

        # Set up cudnn.benchmark for free speed.
        #torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)
        
        # Load weights.
        weights_path = str(module_folder/"BigGAN_utils/weights/biggan-512.pth")
        G.load_state_dict(torch.load(weights_path), strict=False)
    else:
        raise NotImplementedError

    return G


class Translator(nn.Module):
    def __init__(self, clip_dim, z_dim, y_dim, z_thresh, with_gmm: bool=False, num_mixtures=None, 
            gumbel_softmax_temp=None):
        super().__init__()
        self.z_dim = z_dim
        self.z_thresh = z_thresh
        self.with_gmm = with_gmm
        self.num_mixtures = num_mixtures
        self.gumbel_softmax_temp = gumbel_softmax_temp
        self.min_std = 1e-6
        self.shared1 = nn.Linear(clip_dim, 4 * clip_dim)
        self.shared2 = nn.Linear(4 * clip_dim, 4 * clip_dim)
        if self.with_gmm:
            assert num_mixtures is not None
            self.z_layer = nn.Linear(4 * clip_dim, num_mixtures * z_dim)
            self.z_mixture_logits = nn.Linear(4 * clip_dim, num_mixtures)
            self.z_log_stds = nn.Parameter(torch.log(torch.ones(1, 1, z_dim)*0.01))
        else:
            self.z_layer = nn.Linear(4 * clip_dim, z_dim)
        self.y_layer = nn.Linear(4 * clip_dim, y_dim)

    def forward(self, x: torch.Tensor, no_sample: bool=False, reparam_sample: bool=False):
        x = F.relu(self.shared1(x))
        x = x + F.relu(self.shared2(x))
        if self.z_thresh is not None:
            z = self.z_thresh * torch.tanh(self.z_layer(x))
        else:
            z = self.z_layer(x)
        y = self.y_layer(x)
        if self.with_gmm:
            bs = z.shape[0]
            z_mixture_logits = self.z_mixture_logits(x)
            z_means = z.reshape(bs, self.num_mixtures, self.z_dim)
            z_stds = self.z_log_stds.exp() + self.min_std
            z_stds = z_stds.expand(bs, self.num_mixtures, -1)
            if no_sample:
                return z_mixture_logits, z_means, y
            elif reparam_sample:
                z_mix_gumbel_softmax = D.RelaxedOneHotCategorical(self.gumbel_softmax_temp, logits=z_mixture_logits)
                z_mix_gumbel_softmax_sample = z_mix_gumbel_softmax.rsample().unsqueeze(-1) # bs x num_mixtures x 1
                z_comp = D.Independent(D.Normal(z_means, z_stds), 1)
                z_comp_sample = z_comp.rsample() # bs x num_mixtures x z_dim
                z = (z_mix_gumbel_softmax_sample * z_comp_sample).sum(1) # bs x z_dim
                return None, z, y
            else:
                z_mix = D.Categorical(logits=z_mixture_logits)
                z_comp = D.Independent(D.Normal(z_means, z_stds), 1)
                z_gmm = D.MixtureSameFamily(z_mix, z_comp)
                #z = z_gmm.sample() # bs x z_dim
                z_index = torch.argmax(z_mixture_logits, dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.z_dim) # bs x 1 x z_dim
                z = z_means.gather(dim=1, index=z_index).squeeze(1) # bs x z_dim 
                return z_gmm, z, y
        else:
            return z, y


class Model(nn.Module):
    def __init__(self, config, device, clip_noise_cov=None):
        super().__init__()
        self.config = config
        self.with_gmm = config.with_gmm
        self.with_open_clip = config.with_open_clip
        self.with_caption_model = config.with_caption_model
        self.with_covariance = config.with_covariance
        self.with_adaptive_noise = config.with_adaptive_noise

        self.G = get_G(config, device)
        if self.with_open_clip:
            self.clip = open_clip.create_model(config.clip_arch, config.clip_pretrained, device=torch.device(device))
        else:
            self.clip = clip.load(config.clip_arch, device)[0]
        self.translator = Translator(self.clip.visual.output_dim, self.G.dim_z, self.G.shared_dim, config.z_thresh,
                self.with_gmm, config.num_mixtures, config.gumbel_softmax_temp)

        if self.with_caption_model:
            from tr0n.modules.BLIP_utils.models.blip import blip_decoder
            blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
            self.captioner = blip_decoder(pretrained=blip_url, image_size=config.biggan_resolution, vit='base').to(device)
            self.captioner.eval()
            for p in self.captioner.parameters():
                p.requires_grad = False
            self.with_nucleus_sampling = config.with_nucleus_sampling

        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

        self.clip_noise_scale = config.clip_noise_scale
        if self.with_covariance:
            assert clip_noise_cov is not None
            self.clip_noise_dist = D.MultivariateNormal(torch.zeros(self.clip.visual.output_dim).to(device), clip_noise_cov.to(device))

        self.G.eval()
        self.clip.eval()
        self.eval()

        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.clip.parameters():
            p.requires_grad = False

    
    def train(self, mode=True):
        self.training = mode
        for module in self.translator.children():
            module.train(mode)
        return self

    
    def eval(self):
        return self.train(False)


    def with_clip_noise(self, clip_latent: torch.Tensor):
        if self.with_covariance:
            bs = clip_latent.shape[0]
            noise = self.clip_noise_dist.sample((bs,))
        elif self.with_adaptive_noise:
            noise = torch.randn_like(clip_latent)
            noise_norm = noise.norm(dim=-1, keepdim=True)
            clip_latent_norm = clip_latent.norm(dim=-1, keepdim=True)
            adaptive_scale = (clip_latent_norm / noise_norm)
            noise *= adaptive_scale
        else:
            noise = torch.randn_like(clip_latent)
        return clip_latent + (self.clip_noise_scale * noise) 


    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
            x_type: str, add_clip_noise: bool=False, return_after_translator: bool=False,
            no_sample: bool=False, reparam_sample: bool=False, times_augment_pred_image: int=0):
        
        if x_type == 'text':
            assert not isinstance(x, tuple)
            clip_latent = self.clip.encode_text(x).detach().float()
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                z_gmm, z, y = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                z, y = self.translator(clip_latent)
        elif x_type == 'image': 
            assert not isinstance(x, tuple)
            clip_latent = self.clip.encode_image(x).detach().float()
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                z_gmm, z, y = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                z, y = self.translator(clip_latent)
        elif x_type == 'clip_latent':
            assert not isinstance(x, tuple)
            clip_latent = x
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                z_gmm, z, y = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                z, y = self.translator(clip_latent)
        elif x_type == 'gan_latent':
            clip_latent = None
            z, y = x
        else:
            raise NotImplementedError

        if return_after_translator:
            if self.with_gmm:
                return clip_latent, z_gmm, z, y
            else:
                return clip_latent, z, y

        g_image = self.G(z, y)
        g_image_raw = g_image.clone()
        if times_augment_pred_image > 0:
            assert g_image.shape[0] == 1
            g_image = DiffAugment(g_image.expand(times_augment_pred_image, -1, -1, -1), policy='color,translation,resize,cutout')
        g_image = (g_image+1.)/2. #[-1,1] to [0,1]
        
        if self.with_caption_model:
            g_image.sub_(self.clip_mean[None, :, None, None]).div_(self.clip_std[None, :, None, None])
            if self.with_nucleus_sampling:
                g_caption = self.captioner.generate(g_image, sample=True, top_p=0.9, max_length=20, min_length=5)
            else:
                g_caption = self.captioner.generate(g_image, sample=False, num_beams=3, max_length=20, min_length=5)
            if self.with_open_clip:
                g_caption_tok = open_clip.tokenize(g_caption).to(g_image.device)
            else:
                g_caption_tok = clip.tokenize(g_caption, truncate=True).to(g_image.device)
            g_clip_latent = self.clip.encode_text(g_caption_tok).float()
        else:
            if self.with_open_clip:
                g_image = F.interpolate(g_image, size=self.clip.visual.image_size, mode=self.config.interp_mode)
            else:
                g_image = F.interpolate(g_image, size=self.clip.visual.input_resolution, mode=self.config.interp_mode)
            g_image.sub_(self.clip_mean[None, :, None, None]).div_(self.clip_std[None, :, None, None])
            g_clip_latent = self.clip.encode_image(g_image).float()
        return z, y, clip_latent, g_clip_latent, g_image, g_image_raw
