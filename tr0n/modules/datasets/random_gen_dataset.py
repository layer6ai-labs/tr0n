'''
This file is the dataloader to generate random sample from GAN latent space
'''
import os.path
import random
from typing import Any, Callable, Optional, Tuple, List
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
from scipy.stats import truncnorm
import torch.nn.functional as F

class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):    
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif self.dist_type == 'truncate':
            self.dim_z = kwargs['dim_z']
            self.truncation = kwargs['truncation']
        elif self.dist_type == 'uniform':
            self.min_v = kwargs['min_v']
            self.max_v = kwargs['max_v']

    
    def truncated_noise_sample(self, dim_z, truncation=1., seed=None):
        """ Create a truncated noise vector.
            Params:
                dim_z: dim of noise vector
                truncation: truncation value to use
                seed: seed for the random generator
            Output:
                array of shape (batch_size, dim)
        """
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=dim_z, random_state=state).astype(np.float32)
        return torch.Tensor(truncation * values)


    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'truncate':
            self.copy_(self.truncated_noise_sample(dim_z=self.dim_z, truncation=self.truncation))
        elif self.dist_type == 'uniform':
            self.uniform_(self.min_v, self.max_v)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)    
        else:
            raise NotImplemented
        
    
    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj


class RandomBigGANDataset(Dataset):
    def __init__(self,
        dataset_size,
        y_codebook,
        dim_z,
        dim_y,
        n_classes,
        z_thresh,
        z_dist_type, # [truncate|normal|uniform]
        y_dist_type, # [categorical|normal]
        z_truncation):
        super().__init__()
        
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.n_classes = n_classes
        self.z_thresh = z_thresh
        self.z_dist_type = z_dist_type
        self.y_dist_type = y_dist_type
        self.z_truncation = z_truncation
        self.y_codebook = y_codebook
        self.dataset_size = dataset_size
        
        self.z_, self.y_ = self.prepare_z_y()
    
    
    def prepare_z_y(self, z_var=1.0):
        
        z_ = Distribution(torch.randn(self.dim_z, requires_grad=False))
        if self.z_dist_type == 'truncate':
            z_.init_distribution('truncate', dim_z=self.dim_z, truncation=self.z_truncation)
        elif self.z_dist_type == 'normal':
            z_.init_distribution('normal', mean=0, var=z_var)
        elif self.z_dist_type == 'uniform':
            # analogous to fusedream
            z_.init_distribution('uniform', min_v=-self.z_thresh, max_v=self.z_thresh)
        else:
            raise NotImplementedError

        z_ = z_.to(torch.float32)
        if self.y_dist_type == 'normal':
            y_ = Distribution(torch.randn(self.dim_y, requires_grad=False))
            y_.init_distribution('normal', mean=0, var=z_var)
            y_ = y_.to(torch.float32)
        elif self.y_dist_type == 'categorical':
            y_ = Distribution(torch.zeros(1, requires_grad=False))
            y_.init_distribution('categorical', num_categories=self.n_classes)
            y_ = y_.to(torch.int64)
        elif self.y_dist_type == 'categorical_fixed':
            assert self.dataset_size % self.n_classes == 0
            num_per_class = self.dataset_size // self.n_classes
            self.class_sample_list = list(range(self.n_classes)) * num_per_class
            y_ = None
        else:
            raise NotImplementedError

        return z_, y_

    
    def sample(self, z_, y_, idx):
        with torch.no_grad():
            z_.sample_()
            z_ = torch.tensor(z_.numpy())
            if y_ is not None:
                y_.sample_()
                y_ = torch.tensor(y_.numpy())
            if self.y_dist_type == 'categorical':
                y_onehot_ = F.one_hot(y_, num_classes=self.n_classes).to(torch.float32)
                y_cat_ = y_onehot_ @ self.y_codebook 
                return z_, y_cat_.squeeze(0)
            elif self.y_dist_type == 'categorical_fixed':
                y_ = torch.LongTensor([self.class_sample_list[idx]])
                y_onehot_ = F.one_hot(y_, num_classes=self.n_classes).to(torch.float32)
                y_cat_ = y_onehot_ @ self.y_codebook
                return z_, y_cat_.squeeze(0)
            else:
                return z_, y_

    
    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        with torch.no_grad():
            z , y = self.sample(self.z_, self.y_, idx)
            
            z.clamp_(-self.z_thresh, self.z_thresh)

            if self.y_dist_type == 'normal':
                # if normal, we should set the boundary of y
                y.clamp_(-1.0, 1.0)

            return z, y


class RandomStyleGANDataset(Dataset):
    def __init__(self,
        dataset_size,
        w_mapper,
        dim_z,
        z_dist_type,
        z_truncation):
        super().__init__()
        
        self.dim_z = dim_z
        self.w_mapper = w_mapper
        self.dataset_size = dataset_size
        self.z_dist_type = z_dist_type
        self.z_truncation = z_truncation

        self.z_ = self.prepare_z()


    def prepare_z(self, z_var=1.0):
        
        z_ = Distribution(torch.randn(self.dim_z, requires_grad=False))
        if self.z_dist_type == 'truncate':
            z_.init_distribution('truncate', dim_z=self.dim_z, truncation=self.z_truncation)
        elif self.z_dist_type == 'normal':
            z_.init_distribution('normal', mean=0, var=z_var)
        elif self.z_dist_type == 'uniform':
            z_.init_distribution('uniform', min_v=-1., max_v=1.)
        else:
            raise NotImplementedError

        z_ = z_.to(torch.float32)
        return z_
        
    
    def sample(self, z_):
        with torch.no_grad():
            z_.sample_()
            z_ = torch.tensor(z_.numpy())
            w_ = self.w_mapper(z_.unsqueeze(0), c=None)[0, 0]
            return w_


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        with torch.no_grad():
            w = self.sample(self.z_)
            return w


class RandomStyleGANXLDataset(Dataset):
    def __init__(self,
        dataset_size,
        w_mapper,
        w_avg_codebook,
        dim_z,
        dim_y,
        n_classes,
        z_dist_type,
        z_truncation,
        y_dist_type,
        truncation_psi=1.0):
        super().__init__()
        
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.w_mapper = w_mapper
        self.w_avg_codebook = w_avg_codebook
        self.dataset_size = dataset_size
        self.z_dist_type = z_dist_type
        self.z_truncation = z_truncation
        self.y_dist_type = y_dist_type
        self.n_classes = n_classes
        self.truncation_psi = truncation_psi

        self.z_, self.y_ = self.prepare_z_y()


    def prepare_z_y(self, z_var=1.0):
        
        z_ = Distribution(torch.randn(self.dim_z, requires_grad=False))
        if self.z_dist_type == 'truncate':
            z_.init_distribution('truncate', dim_z=self.dim_z, truncation=self.z_truncation)
        elif self.z_dist_type == 'normal':
            z_.init_distribution('normal', mean=0, var=z_var)
        elif self.z_dist_type == 'uniform':
            z_.init_distribution('uniform', min_v=-1., max_v=1.)
        else:
            raise NotImplementedError

        z_ = z_.to(torch.float32)
        if self.y_dist_type == 'normal':
            y_ = Distribution(torch.randn(self.dim_y, requires_grad=False))
            y_.init_distribution('normal', mean=0, var=z_var)
            y_ = y_.to(torch.float32)
        elif self.y_dist_type == 'categorical':
            y_ = Distribution(torch.zeros(1, requires_grad=False))
            y_.init_distribution('categorical', num_categories=self.n_classes)
            y_ = y_.to(torch.int64)
        elif self.y_dist_type == 'categorical_fixed':
            assert self.dataset_size % self.n_classes == 0
            num_per_class = self.dataset_size // self.n_classes
            self.class_sample_list = list(range(self.n_classes)) * num_per_class
            y_ = None
        else:
            raise NotImplementedError

        return z_, y_
        
    
    def sample(self, z_, y_, idx):
        
        with torch.no_grad():
            # z init
            z_.sample_()
            z_ = torch.tensor(z_.numpy())

            # y init
            if self.y_dist_type == 'categorical':
                y_.sample_()
                y_ = torch.tensor(y_.numpy())
                y_onehot_ = F.one_hot(y_, num_classes=self.n_classes).to(torch.float32)
                w_avg = y_onehot_ @ self.w_avg_codebook 
            elif self.y_dist_type == 'categorical_fixed':
                y_ = torch.LongTensor([self.class_sample_list[idx]])
                y_onehot_ = F.one_hot(y_, num_classes=self.n_classes).to(torch.float32)
                w_avg = y_onehot_ @ self.w_avg_codebook

            z_ = z_.unsqueeze(0)
            w_avg = w_avg.unsqueeze(1).repeat(1, 37, 1)

            # w shape, (bs, num_ws=37, 512)
            w_ = self.w_mapper(z_, y_onehot_)

            w_ = w_avg + (w_ - w_avg) * self.truncation_psi
            w_ = w_[0, 0]
            return w_, z_, y_onehot_


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        with torch.no_grad():
            w = self.sample(self.z_, self.y_, idx)
            return w


class RandomNVAEDataset(Dataset):
    def __init__(self,
        dataset_size,
        dim_z,
        z_truncation):
        super().__init__()
        
        self.dataset_size = dataset_size

        from tr0n.modules.NVAE_utils.distributions import Normal
        self.z_ = Normal(mu=torch.zeros(dim_z), log_sigma=torch.zeros(dim_z), temp=z_truncation)

    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        with torch.no_grad():
            z, _ = self.z_.sample()
            return z
