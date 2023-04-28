import os
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm
import clip
import open_clip
from tr0n.modules.trainers.base_trainer import BaseTrainer
from tr0n.modules.utils.basic_utils import mkdir, deletedir
from tr0n.modules.metrics.pytorch_fid import fid_score
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
        pred_image_list = []
        text_list = []
        if self.config.num_hybrid_iters > 0:
            assert self.config.batch_size == 1 # required for hybrid
        if self.config.times_augment_pred_image > 0:
            assert self.config.val_loss.startswith('aug_cosine_latent')
        for image, text in tqdm(self.val_data_loader):
            if self.config.with_open_clip:
                text_tok = open_clip.tokenize(text).to(self.device)
            else:
                text_tok = clip.tokenize(text, truncate=True).to(self.device)
            image = image.to(self.device)
            
            # initialize from translator for hybrid exploration
            with torch.no_grad():
                if self.config.with_gmm:
                    target_clip_latent, z_mixture_logits, z_means, y = self.model(x=text_tok, x_type='text', return_after_translator=True, no_sample=True)
                    pi = z_mixture_logits.unsqueeze(-1).repeat(1, 1, z_means.shape[-1]) # bs x num_mixtures x z_dim
                    z = z_means # bs x num_mixtures x z_dim
                else:
                    target_clip_latent, z, y = self.model(x=text_tok, x_type='text', return_after_translator=True)
                    pi = None
           
            z.requires_grad = True
            y.requires_grad = True
            
            if pi is None:
                optimizer = OptimizerFactory.get_optimizer(self.config.optim, (y,), self.device, self.config.lr,
                        self.config.momentum, self.config.weight_decay, self.config.sgld_noise_std)
            else:
                pi.requires_grad = True
                optimizer = OptimizerFactory.get_optimizer(self.config.optim, (y, pi), self.device, self.config.lr,
                        self.config.momentum, self.config.weight_decay, self.config.sgld_noise_std)

            optimizer_latent = OptimizerFactory.get_optimizer(self.config.optim_latent, (z,), self.device, self.config.lr_latent,
                    self.config.momentum, self.config.weight_decay, self.config.sgld_noise_std)

            for i in range(self.config.num_hybrid_iters):
                if pi is None:
                    z_prime = z
                else:
                    soft_pi = F.softmax(pi, dim=1)
                    z_prime = soft_pi * z
                    z_prime = z_prime.sum(dim=1)
                
                _, _, _, pred_clip_latent, _, _ = self.model(x=(z_prime, y), x_type='gan_latent',
                        times_augment_pred_image=self.config.times_augment_pred_image)

                if self.config.val_loss.endswith('aesthetic'):
                    _, loss = self.val_loss(target_clip_latent, pred_clip_latent)
                else:
                    loss = self.val_loss(target_clip_latent, pred_clip_latent)
                loss.backward()

                torch.nn.utils.clip_grad_norm_((z,), self.config.max_grad_norm)
                if pi is None:
                    torch.nn.utils.clip_grad_norm_((y,), self.config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_((y, pi), self.config.max_grad_norm)
                optimizer_latent.step()
                optimizer.step()
                optimizer_latent.zero_grad()
                optimizer.zero_grad()

                z.data = torch.clamp(z.data, min=-self.config.z_thresh, max=self.config.z_thresh)

            with torch.no_grad():
                if pi is None:
                    z_prime = z
                else:
                    soft_pi = F.softmax(pi, dim=1)
                    z_prime = soft_pi * z
                    z_prime = z_prime.sum(dim=1)
                
                _, _, _, pred_clip_latent, _, pred_image_raw = self.model(x=(z_prime, y), x_type='gan_latent',
                        times_augment_pred_image=self.config.times_augment_pred_image)

                if self.config.val_loss.endswith('aesthetic'):
                    loss, _ = self.val_loss(target_clip_latent, pred_clip_latent)
                else:
                    loss = self.val_loss(target_clip_latent, pred_clip_latent)

            total_val_loss += loss.item()
                
            pred_image_list.extend(((pred_image_raw+1.)/2.).cpu())
            text_list.extend(text)
                
        total_val_loss = total_val_loss / len(self.val_data_loader)
        
        deletedir(self.save_image_dir)
        mkdir(self.save_image_dir)
        for i, (image, text) in enumerate(zip(pred_image_list, text_list)):
            PIL_image = ToPILImage()(image)
            text_path = text.strip().replace('.', '').replace('/', '')
            text_path = text_path[:min(len(text_path), 200)]
            PIL_image.save(os.path.join(self.save_image_dir, f'image_{i}_{text_path}.jpg'))

            if self.writer is not None and i < 10:
                self.writer.add_image(f'image_{i}_{text}', image, self.global_step)
        del pred_image_list
        del text_list

        fid_value = fid_score.calculate_fid_given_paths(
                (self.config.val_image_dir, self.save_image_dir),
                self.config.batch_size,
                self.device,
                2048,
                self.config.biggan_resolution,
                self.config.num_workers)

        res = dict()
        res['loss_val'] = total_val_loss
        res['fid_val'] = fid_value

        print(f"-----Val Epoch: {epoch}-----\n",
                f"FID: {res['fid_val']}\n", 
                f"Loss: {res['loss_val']}")

        if self.writer is not None:
            for m in res:
                self.writer.add_scalar(f'val/{m}', res[m], self.global_step)
            self.writer.flush()
        return res
