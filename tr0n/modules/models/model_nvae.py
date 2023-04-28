from pathlib import Path                                                                      
module_folder = Path(__file__).parent.parent 
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.cuda.amp import autocast
import clip
import open_clip
from tr0n.modules.utils.diff_augment import DiffAugment
from tr0n.modules.NVAE_utils.distributions import Normal


class NVAE_Dec_Wrapper(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.temp = config.z_truncation

        models = {
                    "ffhq": partial(self.load_nvae, str(module_folder/'NVAE_utils/ckpt/ffhq/checkpoint.pt')),
                    "celeba_256": partial(self.load_nvae, str(module_folder/'NVAE_utils/ckpt/celeba_256/checkpoint.pt')),
                }
        self.model = models[config.nvae_model](config, device)
        self.z_size = self.model.z0_size
        self.z_dim = self.z_size[0] * self.z_size[1] * self.z_size[2]
    
    def load_nvae(self, checkpoint_path, config, device):
        import tr0n.modules.NVAE_utils.tools as utils
        from tr0n.modules.NVAE_utils.model import AutoEncoder
        
        checkpoint = torch.load(checkpoint_path)
        args = checkpoint['args']

        if not hasattr(args, 'ada_groups'):
            print('old NVA model, no ada groups was found.')
            args.ada_groups = False

        if not hasattr(args, 'min_groups_per_scale'):
            print('old NVAE model, no min_groups_per_scale was found.')
            args.min_groups_per_scale = 1

        if not hasattr(args, 'num_mixture_dec'):
            print('old NVAE model, no num_mixture_dec was found.')
            args.num_mixture_dec = 10

        args.batch_size = config.batch_size
        arch_instance = utils.get_arch_cells(args.arch_instance)
        model = AutoEncoder(args, None, arch_instance)
        # Loading is not strict because of self.weight_normalized in Conv2D class in neural_operations. This variable
        # is only used for computing the spectral normalization and it is safe not to load it. Some of our earlier models
        # did not have this variable.
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model = model.to(device)
        return model

    def forward(self, z):
        with autocast():
            scale_ind = 0
            idx_dec = 0
            s = self.model.prior_ftr0.unsqueeze(0)
            batch_size = z.size(0)
            z = z.reshape(batch_size, *self.z_size)
            s = s.expand(batch_size, -1, -1, -1)
            for cell in self.model.dec_tower:
                if cell.cell_type == 'combiner_dec':
                    if idx_dec > 0:
                        # form prior
                        param = self.model.dec_sampler[idx_dec - 1](s)
                        mu, log_sigma = torch.chunk(param, 2, dim=1)
                        dist = Normal(mu, log_sigma, self.temp)
                        if idx_dec > 7:
                            z, _ = dist.sample()
                        else:
                            z = dist.mean()

                    # 'combiner_dec'
                    s = cell(s, z)
                    idx_dec += 1
                else:
                    s = cell(s)
                    if cell.cell_type == 'up_dec':
                        scale_ind += 1

            if self.model.vanilla_vae:
                s = self.model.stem_decoder(z)

            for cell in self.model.post_process:
                s = cell(s)

            logits = self.model.image_conditional(s)

        output = self.model.decoder_output(logits)
        output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
                    else output.mean()
        output_img = (output_img*2.)-1. # bring image back to [-1,1] for better compatibility with code
        return output_img


def get_dec(config, device):
    dec = NVAE_Dec_Wrapper(config, device)
    config = vars(config)
    config["dim_z"] = dec.z_dim
    return dec


class Translator(nn.Module):
    def __init__(self, clip_dim, z_dim, z_thresh, with_gmm: bool=False, num_mixtures=None, 
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

    def forward(self, x: torch.Tensor, no_sample: bool=False, reparam_sample: bool=False):
        x = F.relu(self.shared1(x))
        x = x + F.relu(self.shared2(x))
        if self.z_thresh is not None:
            z = self.z_thresh * torch.tanh(self.z_layer(x))
        else:
            z = self.z_layer(x)
        if self.with_gmm:
            bs = z.shape[0]
            z_mixture_logits = self.z_mixture_logits(x)
            z_means = z.reshape(bs, self.num_mixtures, self.z_dim)
            z_stds = self.z_log_stds.exp() + self.min_std
            z_stds = z_stds.expand(bs, self.num_mixtures, -1)
            if no_sample:
                return z_mixture_logits, z_means
            elif reparam_sample:
                z_mix_gumbel_softmax = D.RelaxedOneHotCategorical(self.gumbel_softmax_temp, logits=z_mixture_logits)
                z_mix_gumbel_softmax_sample = z_mix_gumbel_softmax.rsample().unsqueeze(-1) # bs x num_mixtures x 1
                z_comp = D.Independent(D.Normal(z_means, z_stds), 1)
                z_comp_sample = z_comp.rsample() # bs x num_mixtures x z_dim
                z = (z_mix_gumbel_softmax_sample * z_comp_sample).sum(1) # bs x z_dim
                return None, z
            else:
                z_mix = D.Categorical(logits=z_mixture_logits)
                z_comp = D.Independent(D.Normal(z_means, z_stds), 1)
                z_gmm = D.MixtureSameFamily(z_mix, z_comp)
                #z = z_gmm.sample() # bs x z_dim
                z_index = torch.argmax(z_mixture_logits, dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.z_dim) # bs x 1 x z_dim
                z = z_means.gather(dim=1, index=z_index).squeeze(1) # bs x z_dim 
                return z_gmm, z
        else:
            return z


class Model(nn.Module):
    def __init__(self, config, device, clip_noise_cov=None):
        super().__init__()
        self.config = config
        self.with_gmm = config.with_gmm
        self.with_open_clip = config.with_open_clip
        self.with_covariance = config.with_covariance
        self.with_adaptive_noise = config.with_adaptive_noise

        self.dec = get_dec(config, device)
        if self.with_open_clip:
            self.clip = open_clip.create_model(config.clip_arch, config.clip_pretrained, device=torch.device(device))
        else:
            self.clip = clip.load(config.clip_arch, device)[0]
        self.translator = Translator(self.clip.visual.output_dim, self.dec.z_dim, None,
                self.with_gmm, config.num_mixtures, config.gumbel_softmax_temp)

        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)

        self.clip_noise_scale = config.clip_noise_scale
        if self.with_covariance:
            assert clip_noise_cov is not None
            self.clip_noise_dist = D.MultivariateNormal(torch.zeros(self.clip.visual.output_dim).to(device), clip_noise_cov.to(device))

        self.dec.eval()
        self.clip.eval()
        self.eval()

        for p in self.dec.parameters():
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


    def forward(self, x: torch.Tensor, x_type: str, add_clip_noise: bool=False, 
            return_after_translator: bool=False, no_sample: bool=False, 
            reparam_sample: bool=False, times_augment_pred_image: int=0):
        
        if x_type == 'text':
            clip_latent = self.clip.encode_text(x).detach().float()
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                z_gmm, z = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                z = self.translator(clip_latent)
        elif x_type == 'image': 
            clip_latent = self.clip.encode_image(x).detach().float()
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                z_gmm, z = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                z = self.translator(clip_latent)
        elif x_type == 'clip_latent':
            clip_latent = x
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                z_gmm, z = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                z = self.translator(clip_latent)
        elif x_type == 'gan_latent':
            clip_latent = None
            z = x
        else:
            raise NotImplementedError

        if return_after_translator:
            if self.with_gmm:
                return clip_latent, z_gmm, z
            else:
                return clip_latent, z

        g_image = self.dec(z)
        g_image_raw = g_image.clone()
        if times_augment_pred_image > 0:
            assert g_image.shape[0] == 1
            g_image = DiffAugment(g_image.expand(times_augment_pred_image, -1, -1, -1), policy='color,translation,resize,cutout')
        g_image = (g_image+1.)/2. #[-1,1] to [0,1]

        if self.with_open_clip:
            g_image = F.interpolate(g_image, size=self.clip.visual.image_size, mode=self.config.interp_mode)
        else:
            g_image = F.interpolate(g_image, size=self.clip.visual.input_resolution, mode=self.config.interp_mode)
        g_image.sub_(self.clip_mean[None, :, None, None]).div_(self.clip_std[None, :, None, None])
        g_clip_latent = self.clip.encode_image(g_image).float()
        return z, clip_latent, g_clip_latent, g_image, g_image_raw
