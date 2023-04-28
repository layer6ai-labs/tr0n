import os
import sys
from pathlib import Path
code_folder = Path(__file__).parent      
sys.path.append(str(code_folder/"tr0n/modules/BigGAN_utils"))      
sys.path.append(str(code_folder/"tr0n/modules/BLIP_utils"))      
sys.path.append(str(code_folder/"tr0n/modules/StyleGAN_utils"))      
sys.path.append(str(code_folder/"tr0n/modules/NVAE_utils"))
sys.path.append(str(code_folder/"tr0n/modules/sgxl_utils"))
import torch
import random
import numpy as np
from tr0n.config import parse_args
from tr0n.modules.datasets.data_factory import DataFactory
from tr0n.modules.models.model_factory import ModelFactory
from tqdm import tqdm
import pickle as pkl
import torch.nn.functional as F

def generate_synthetic_from_biggan(config, device):
    # generate synthetic data using biggan
    model = ModelFactory.get_model(config, device)
    data_loader = DataFactory.get_loader(config, codebook=model.G.shared.weight.cpu())

    save_list = []
    with torch.no_grad():
        for z, y in tqdm(data_loader):
            z = z.to(device)
            y = y.to(device)

            _, _, _, g_clip_latent, _, _ = model(x=(z,y), x_type='gan_latent') 

            for z_, y_, g_clip_latent_ in zip(z, y, g_clip_latent):
                save_dict = {
                    'clip_latent': g_clip_latent_.cpu().numpy(),
                    'z': z_.cpu().numpy(),
                    'y': y_.cpu().numpy(),
                }
                save_list.append(save_dict)

    with open(os.path.join('data', '{}.pkl'.format(config.synthetic_latents_path)), 'wb') as handle:
        pkl.dump(save_list, handle, protocol=pkl.HIGHEST_PROTOCOL)


def generate_synthetic_from_stylegan(config, device):
    # generate synthetic data using stylegan
    model = ModelFactory.get_model(config, device)
    data_loader = DataFactory.get_loader(config, mapper=model.G.mapping.cpu())

    save_list = []
    with torch.no_grad():
        for w in tqdm(data_loader):
            w = w.to(device)

            _, _, g_clip_latent, _, _ = model(x=w, x_type='gan_latent')

            for w_, g_clip_latent_ in zip(w, g_clip_latent):
                save_dict = {
                    'clip_latent': g_clip_latent_.cpu().numpy(),
                    'w': w_.cpu().numpy(),
                }
                save_list.append(save_dict)
                
    with open(os.path.join('data', '{}.pkl'.format(config.synthetic_latents_path)), 'wb') as handle:
        pkl.dump(save_list, handle, protocol=pkl.HIGHEST_PROTOCOL)


def generate_synthetic_from_sgxl(config, device):
    # generate synthetic data using stylegan-xl
    model = ModelFactory.get_model(config, device)
    data_loader = DataFactory.get_loader(config, mapper=model.G.mapping.cpu(), codebook=model.G.mapping.w_avg.cpu())

    save_list = []
    with torch.no_grad():
        for w, z, y in tqdm(data_loader):
            w = w.to(device)

            _, _, g_clip_latent, _, _ = model(x=w, x_type='gan_latent')

            for w_, g_clip_latent_, z_, y_ in zip(w, g_clip_latent, z, y):
                save_dict = {
                    'clip_latent': g_clip_latent_.cpu().numpy(),
                    'w': w_.cpu().numpy(),
                    'z': z_.cpu().numpy(),
                    'y': y_.cpu().numpy(),
                }
                save_list.append(save_dict)
                
    with open(os.path.join('data', '{}.pkl'.format(config.synthetic_latents_path)), 'wb') as handle:
        pkl.dump(save_list, handle, protocol=pkl.HIGHEST_PROTOCOL)



def generate_synthetic_from_nvae(config, device):
    # generate synthetic data using nvae
    model = ModelFactory.get_model(config, device)
    data_loader = DataFactory.get_loader(config)

    save_list = []
    with torch.no_grad():
        for z in tqdm(data_loader):
            z = z.to(device)

            _, _, g_clip_latent, _, _ = model(x=z, x_type='gan_latent')

            for z_, g_clip_latent_ in zip(z, g_clip_latent):
                if torch.isnan(g_clip_latent).any():
                    continue
                save_dict = {
                    'clip_latent': g_clip_latent_.cpu().numpy(),
                    'z': z_.cpu().numpy(),
                }
                save_list.append(save_dict)
                
    with open(os.path.join('data', '{}.pkl'.format(config.synthetic_latents_path)), 'wb') as handle:
        pkl.dump(save_list, handle, protocol=pkl.HIGHEST_PROTOCOL)

    print(f'Successfully generated {len(save_list)} synthetic pairs.')


def main():
    config = parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if config.dec_type == 'biggan':
        generate_synthetic_from_biggan(config, device)
    elif config.dec_type == 'stylegan':
        generate_synthetic_from_stylegan(config, device)
    elif config.dec_type == 'sgxl':
        generate_synthetic_from_sgxl(config, device)
    elif config.dec_type == 'nvae':
        generate_synthetic_from_nvae(config, device)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
