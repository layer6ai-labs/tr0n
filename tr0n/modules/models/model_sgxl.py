from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision.transforms import ToPILImage
import clip
import open_clip
from tr0n.modules.utils.diff_augment import DiffAugment


class StyleGAN_Wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        generators = {
            "sgxl-imagenet-512": partial(self.load_stylegan, 'https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet512.pkl')
        }
        self.G = generators[config.stylegan_gen]()
        self.mapping = self.G.mapping
        self.z_dim = self.G.z_dim
        self.w_dim = self.G.mapping.w_dim
        self.num_ws = self.G.mapping.num_ws
        self.noise_mode = 'const'
    
    def load_stylegan(self, network_pkl):
        from tr0n.modules.sgxl_utils.dnnlib.util import open_url
        from tr0n.modules.sgxl_utils.legacy import load_network_pkl

        with open_url(network_pkl) as f:
            G = load_network_pkl(f)['G_ema'] # type: ignore
        return G

    def forward(self, w):
        # w (bs, w_dim) -> (bs, num_ws, w_dim)
        w = w.unsqueeze(1).expand(-1, self.mapping.num_ws, -1)
        out = self.G.synthesis(w, noise_mode=self.noise_mode).clamp(-1., 1.)
        return out


def get_G(config, device):
    G = StyleGAN_Wrapper(config)
    config = vars(config)
    config["dim_z"] = G.z_dim
    return G.to(device)


class Translator(nn.Module):
    def __init__(self, clip_dim, w_dim, w_thresh, with_gmm: bool=False, num_mixtures=None, 
            gumbel_softmax_temp=None):
        super().__init__()
        self.w_dim = w_dim
        self.w_thresh = w_thresh
        self.with_gmm = with_gmm
        self.num_mixtures = num_mixtures
        self.gumbel_softmax_temp = gumbel_softmax_temp
        self.min_std = 1e-6
        self.shared1 = nn.Linear(clip_dim, 4 * clip_dim)
        self.shared2 = nn.Linear(4 * clip_dim, 4 * clip_dim)
        if self.with_gmm:
            assert num_mixtures is not None
            self.w_layer = nn.Linear(4 * clip_dim, num_mixtures * w_dim)
            self.w_mixture_logits = nn.Linear(4 * clip_dim, num_mixtures)
            self.w_log_stds = nn.Parameter(torch.log(torch.ones(1, 1, w_dim)*0.01))
        else:
            self.w_layer = nn.Linear(4 * clip_dim, w_dim)

    def forward(self, x: torch.Tensor, no_sample: bool=False, reparam_sample: bool=False):
        x = F.relu(self.shared1(x))
        x = x + F.relu(self.shared2(x))
        if self.w_thresh is not None:
            w = self.w_thresh * torch.tanh(self.w_layer(x))
        else:
            w = self.w_layer(x)
        if self.with_gmm:
            bs = w.shape[0]
            w_mixture_logits = self.w_mixture_logits(x)
            w_means = w.reshape(bs, self.num_mixtures, self.w_dim)
            w_stds = self.w_log_stds.exp() + self.min_std
            w_stds = w_stds.expand(bs, self.num_mixtures, -1)
            if no_sample:
                return w_mixture_logits, w_means
            elif reparam_sample:
                w_mix_gumbel_softmax = D.RelaxedOneHotCategorical(self.gumbel_softmax_temp, logits=w_mixture_logits)
                w_mix_gumbel_softmax_sample = w_mix_gumbel_softmax.rsample().unsqueeze(-1) # bs x num_mixtures x 1
                w_comp = D.Independent(D.Normal(w_means, w_stds), 1)
                w_comp_sample = w_comp.rsample() # bs x num_mixtures x w_dim
                w = (w_mix_gumbel_softmax_sample * w_comp_sample).sum(1) # bs x w_dim
                return None, w
            else:
                w_mix = D.Categorical(logits=w_mixture_logits)
                w_comp = D.Independent(D.Normal(w_means, w_stds), 1)
                w_gmm = D.MixtureSameFamily(w_mix, w_comp)
                #w = w_gmm.sample() # bs x w_dim
                w_index = torch.argmax(w_mixture_logits, dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.w_dim) # bs x 1 x w_dim
                w = w_means.gather(dim=1, index=w_index).squeeze(1) # bs x w_dim 
                return w_gmm, w
        else:
            return w


class Model(nn.Module):
    def __init__(self, config, device, clip_noise_cov=None):
        super().__init__()
        self.config = config
        self.device = device
        self.with_gmm = config.with_gmm
        self.with_open_clip = config.with_open_clip
        self.with_covariance = config.with_covariance
        self.with_adaptive_noise = config.with_adaptive_noise
        self.caption_model_type = 'blip'
        self.G = get_G(config, device)
        if self.with_open_clip:
            self.clip = open_clip.create_model(config.clip_arch, config.clip_pretrained, device=torch.device(device))
        else:
            self.clip = clip.load(config.clip_arch, device)[0]
        self.translator = Translator(self.clip.visual.output_dim, self.G.mapping.w_dim, None,
                self.with_gmm, config.num_mixtures, config.gumbel_softmax_temp)

        self.with_caption_model = config.with_caption_model

        if self.with_caption_model:
            # print('with caption model')
            if self.caption_model_type == 'blip':
                from tr0n.modules.BLIP_utils.models.blip import blip_decoder
                blip_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
                self.captioner = blip_decoder(pretrained=blip_url, image_size=config.biggan_resolution, vit='base').to(device)
                self.captioner.eval()
                for p in self.captioner.parameters():
                    p.requires_grad = False
                self.with_nucleus_sampling = config.with_nucleus_sampling
            elif self.caption_model_type == 'blip2':
                from lavis.models import load_model_and_preprocess
                # Out of CUDA Memory
                # self.captioner, vis_processors, _ = load_model_and_preprocess(
                #     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
                # )

                # no support for bfloat16, float Out of CUDA Memory
                # self.captioner, self.vis_processors, _ = load_model_and_preprocess(
                #     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
                # )

                self.captioner, self.vis_processors, _ = load_model_and_preprocess(
                    name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
                )
                
                # self.captioner = self.captioner.float()
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


    def forward(self, x: torch.Tensor, x_type: str, add_clip_noise: bool=False, 
            return_after_translator: bool=False, no_sample: bool=False, 
            reparam_sample: bool=False, times_augment_pred_image: int=0):
        
        if x_type == 'text':
            clip_latent = self.clip.encode_text(x).detach().float()
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                w_gmm, w = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                w = self.translator(clip_latent)
        elif x_type == 'image': 
            clip_latent = self.clip.encode_image(x).detach().float()
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                w_gmm, w = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                w = self.translator(clip_latent)
        elif x_type == 'clip_latent':
            clip_latent = x
            if add_clip_noise:
                clip_latent = self.with_clip_noise(clip_latent)
            if self.with_gmm:
                w_gmm, w = self.translator(clip_latent, no_sample, reparam_sample)
            else:
                w = self.translator(clip_latent)
        elif x_type == 'gan_latent':
            clip_latent = None
            w = x
        else:
            raise NotImplementedError

        if return_after_translator:
            if self.with_gmm:
                return clip_latent, w_gmm, w
            else:
                return clip_latent, w

        g_image = self.G(w)
        g_image_raw = g_image.clone()
        if times_augment_pred_image > 0:
            assert g_image.shape[0] == 1
            g_image = DiffAugment(g_image.expand(times_augment_pred_image, -1, -1, -1), policy='color,translation,resize,cutout')
        g_image = (g_image+1.)/2. #[-1,1] to [0,1]
        
        if self.with_caption_model:
            if self.caption_model_type == 'blip':
                g_image.sub_(self.clip_mean[None, :, None, None]).div_(self.clip_std[None, :, None, None])
                if self.with_nucleus_sampling:
                    g_caption = self.captioner.generate(g_image, sample=True, top_p=0.9, max_length=20, min_length=5)
                else:
                    g_caption = self.captioner.generate(g_image, sample=False, num_beams=3, max_length=20, min_length=5)
                
            elif self.caption_model_type == 'blip2':
                g_image_cap = g_image.clone()
                g_image_cap = F.interpolate(g_image_cap, size=224, mode=self.config.interp_mode)
                g_image_cap.sub_(self.clip_mean[None, :, None, None]).div_(self.clip_std[None, :, None, None])
                
                if self.with_nucleus_sampling:
                    g_caption = self.captioner.generate({"image": g_image_cap}, use_nucleus_sampling=True)
                else:
                    g_caption = self.captioner.generate({"image": g_image_cap})
            else:
                print(f'No Implementation for the caption model {self.caption_model} !')

            g_image.sub_(self.clip_mean[None, :, None, None]).div_(self.clip_std[None, :, None, None])
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
        return w, clip_latent, g_clip_latent, g_image, g_image_raw
