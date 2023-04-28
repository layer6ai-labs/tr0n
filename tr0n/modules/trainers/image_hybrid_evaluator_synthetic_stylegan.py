import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from tr0n.modules.trainers.base_trainer import BaseTrainer
from tr0n.modules.utils.basic_utils import mkdir, deletedir
from tr0n.modules.optimizers.optimizer_factory import OptimizerFactory

class Evaluator(BaseTrainer):
    def __init__(self, model, train_loss, val_loss, optimizers, config, train_data_loader,
                 val_data_loader, device, lr_schedulers=None, writer=None):
        super().__init__(model, train_loss, val_loss, optimizers, config, device, lr_schedulers, writer)
        
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader


    def _val_epoch(self, epoch):
        """
        Validate at a certain epoch
        :return: A log that contains information about validation
        """
        self.model.eval()
        for p in self.model.translator.parameters():
            p.requires_grad = False
        total_val_loss = 0.0
        orig_image_list = []
        pred_image_list = []
        if self.config.num_hybrid_iters > 0:
            assert self.config.batch_size == 1 # required for hybrid
        if self.config.times_augment_pred_image > 0:
            assert self.config.val_loss.startswith('aug_cosine_latent')
        for image_orig, image in tqdm(self.val_data_loader):
            orig_image_list.extend(image_orig)
            image = image.to(self.device)

            # initialize from translator for hybrid exploration
            with torch.no_grad():
                if self.config.with_gmm:
                    target_clip_latent, _, w = self.model(x=image, x_type='image', return_after_translator=True)
                else:
                    target_clip_latent, w = self.model(x=image, x_type='image', return_after_translator=True)
            
            w.requires_grad = True 
            
            optimizer_latent = OptimizerFactory.get_optimizer(self.config.optim_latent, (w,), self.device, self.config.lr_latent, 
                    self.config.momentum, self.config.weight_decay, self.config.sgld_noise_std)

            for i in range(self.config.num_hybrid_iters):
                _, _, pred_clip_latent, _, _ = self.model(x=w, x_type='gan_latent',
                        times_augment_pred_image=self.config.times_augment_pred_image)

                if self.config.val_loss.endswith('aesthetic'):
                    _, loss = self.val_loss(target_clip_latent, pred_clip_latent)
                else:
                    loss = self.val_loss(target_clip_latent, pred_clip_latent)
                loss.backward()

                torch.nn.utils.clip_grad_norm_((w,), self.config.max_grad_norm) 
                optimizer_latent.step() 
                optimizer_latent.zero_grad()
 
            with torch.no_grad():
                _, _, pred_clip_latent, _, pred_image_raw = self.model(x=w, x_type='gan_latent',
                        times_augment_pred_image=self.config.times_augment_pred_image)

                if self.config.val_loss.endswith('aesthetic'):
                    loss, _ = self.val_loss(target_clip_latent, pred_clip_latent)
                else:
                    loss = self.val_loss(target_clip_latent, pred_clip_latent)

            total_val_loss += loss.item()
                
            pred_image_list.extend(((pred_image_raw+1.)/2.).cpu())
                
        total_val_loss = total_val_loss / len(self.val_data_loader)
        
        deletedir(self.save_image_dir)
        mkdir(self.save_image_dir)
        for i, (image_orig, image) in enumerate(zip(orig_image_list, pred_image_list)):
            image_orig = image_orig.unsqueeze(0)
            image = image.unsqueeze(0)
            image = torch.cat((image_orig, image), dim=0)
            image_grid = make_grid(image, nrow=2)
            PIL_image = ToPILImage()(image_grid)
            PIL_image.save(os.path.join(self.save_image_dir, f'image_{i}.jpg'))
        del orig_image_list
        del pred_image_list

        res = dict()
        res['loss_val'] = total_val_loss

        print(f"-----Val Epoch: {epoch}-----\n", 
                f"Loss: {res['loss_val']}")
        return res
