import os
import torch
from torchvision.transforms import ToPILImage
import numpy as np
from tqdm import tqdm
import clip
import open_clip
from tr0n.modules.trainers.base_trainer import BaseTrainer
from tr0n.modules.utils.basic_utils import mkdir, deletedir

class Trainer(BaseTrainer):
    def __init__(self, model, train_loss, val_loss, optimizers, config, train_data_loader,
                 val_data_loader, device, lr_schedulers=None, writer=None):
        super().__init__(model, train_loss, val_loss, optimizers, config, device, lr_schedulers, writer)
        
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader

        self.best = float('inf')


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)

        for batch_idx, (clip_latent, target_w) in enumerate(self.train_data_loader):
            
            clip_latent = clip_latent.to(self.device)
            target_w = target_w.to(self.device)
            
            if self.config.train_loss.startswith('cosine_latent'):
                _, target_clip_latent, pred_clip_latent, _, _ = self.model(x=clip_latent, 
                        x_type='clip_latent', add_clip_noise=self.config.add_clip_noise, 
                        reparam_sample=True)
                
                if self.config.train_loss.endswith('aesthetic'):
                    _, loss = self.train_loss(target_clip_latent, pred_clip_latent)
                else:
                    loss = self.train_loss(target_clip_latent, pred_clip_latent)
            
            elif self.config.train_loss == 'mse_latent':
                _, pred_w = self.model(x=clip_latent, x_type='clip_latent', 
                    add_clip_noise=self.config.add_clip_noise, return_after_translator=True)                        
            
                loss = self.train_loss(target_w, pred_w)
            
            elif self.config.train_loss == 'nll_latent':
                assert self.config.with_gmm
                _, pred_w_gmm, _ = self.model(x=clip_latent, x_type='clip_latent',
                        add_clip_noise=self.config.add_clip_noise, return_after_translator=True)

                loss = self.train_loss(target_w, pred_w_gmm, None, None)

            else:
                raise NotImplementedError

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.translator.parameters(), self.config.max_grad_norm)
            self.optimizers['translator'].step()
            if self.lr_schedulers is not None:
                self.lr_schedulers['translator'].step()
            self.optimizers['translator'].zero_grad()

            self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar('train/loss_train', loss.detach().item(), self.global_step)
                if self.lr_schedulers is not None:
                    self.writer.add_scalar('train/lr', self.lr_schedulers['translator'].get_last_lr()[0], self.global_step)

            total_loss += loss.detach().item()

            if (batch_idx+1) % self.log_step == 0:
                print('Train Epoch: {} dl: {}/{} Loss: {:.6f}'.format(
                    epoch,
                    batch_idx+1,
                    num_steps,
                    loss.detach().item()))

        if epoch % self.config.eval_every == 0:
            val_res = self._val_epoch(epoch)
            self.model.train()

            if val_res['loss_val'] < self.best:
                self.best = val_res['loss_val']
                self._save_checkpoint(epoch, save_best=True)

            print(" Current Best Val. Loss is {}\n\n".format(self.best))

        if self.writer is not None:
            self.writer.flush()
        res = {
            'loss_train':  total_loss / num_steps
        } 
        return res


    def _val_epoch(self, epoch):
        """
        Validate at a certain epoch
        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0.0
        pred_image_list = []
        text_list = []
        assert not self.config.val_loss.endswith('aesthetic')
        if self.config.times_augment_pred_image > 0:
            assert self.config.val_loss == 'aug_cosine_latent'
        with torch.no_grad():
            for text in tqdm(self.val_data_loader):
                if self.config.with_open_clip:
                    text_tok = open_clip.tokenize(text).to(self.device)
                else:
                    text_tok = clip.tokenize(text, truncate=True).to(self.device)

                _, target_clip_latent, pred_clip_latent, _, pred_image_raw = self.model(x=text_tok, x_type='text',
                        times_augment_pred_image=self.config.times_augment_pred_image)

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

        res = dict()
        res['loss_val'] = total_val_loss

        print(f"-----Val Epoch: {epoch}-----\n", 
                f"Loss: {res['loss_val']}")

        if self.writer is not None:
            for m in res:
                self.writer.add_scalar(f'val/{m}', res[m], self.global_step)
            self.writer.flush()
        return res
