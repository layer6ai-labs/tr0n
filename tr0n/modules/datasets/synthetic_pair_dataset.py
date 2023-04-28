import os
from torch.utils.data import Dataset
import torch
import pickle as pkl


class SyntheticBigGANDataset(Dataset):
    def __init__(self, synthetic_latents_path):
        super().__init__()
        
        self.synthetic_latents_path = os.path.join('data', '{}.pkl'.format(synthetic_latents_path))
        assert os.path.exists(self.synthetic_latents_path)
        
        self.z_list = []
        self.y_list = []
        self.clip_latent_list = []
        with open(self.synthetic_latents_path, 'rb') as f:
            latent_list = pkl.load(f)
            for latent_dict in latent_list:
                self.z_list.append(torch.tensor(latent_dict['z']))
                self.y_list.append(torch.tensor(latent_dict['y']))
                self.clip_latent_list.append(torch.tensor(latent_dict['clip_latent']))

        self.cov = torch.cov(torch.stack(self.clip_latent_list, dim=1))
 
        
    def __len__(self):
        return len(self.z_list)

    
    def __getitem__(self, idx):
        return self.clip_latent_list[idx], self.z_list[idx], self.y_list[idx]


class SyntheticStyleGANDataset(Dataset):
    def __init__(self, synthetic_latents_path):
        super().__init__()
        
        self.synthetic_latents_path = os.path.join('data', '{}.pkl'.format(synthetic_latents_path))
        assert os.path.exists(self.synthetic_latents_path)
        
        self.w_list = []
        self.clip_latent_list = []
        with open(self.synthetic_latents_path, 'rb') as f:
            latent_list = pkl.load(f)
            for latent_dict in latent_list:
                self.w_list.append(torch.tensor(latent_dict['w']))
                self.clip_latent_list.append(torch.tensor(latent_dict['clip_latent']))

        self.cov = torch.cov(torch.stack(self.clip_latent_list, dim=1))


    def __len__(self):
        return len(self.w_list)

    
    def __getitem__(self, idx):
        return self.clip_latent_list[idx], self.w_list[idx]


class SyntheticNVAEDataset(Dataset):
    def __init__(self, synthetic_latents_path):
        super().__init__()
        
        self.synthetic_latents_path = os.path.join('data', '{}.pkl'.format(synthetic_latents_path))
        assert os.path.exists(self.synthetic_latents_path)
        
        self.z_list = []
        self.clip_latent_list = []
        with open(self.synthetic_latents_path, 'rb') as f:
            latent_list = pkl.load(f)
            for latent_dict in latent_list:
                self.z_list.append(torch.tensor(latent_dict['z']))
                self.clip_latent_list.append(torch.tensor(latent_dict['clip_latent']))

        self.cov = torch.cov(torch.stack(self.clip_latent_list, dim=1))


    def __len__(self):
        return len(self.z_list)

    
    def __getitem__(self, idx):
        return self.clip_latent_list[idx], self.z_list[idx]
