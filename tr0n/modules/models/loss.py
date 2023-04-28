import torch.nn as nn
import torch
import torch.nn.functional as F
from tr0n.modules.models.aesthetic_predictor import get_aesthetic_model

class CosineSimLatent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_latent, pred_latent):
        """
        Inputs: predicted and ground truth latents
            target_latent: bs x dim
            pred_latent: bs x dim
        """
        sims = F.cosine_similarity(pred_latent, target_latent)
        return -sims.mean()


class AugCosineSimLatent(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target_latent, pred_latent):
        """
        Inputs: predicted and ground truth latents
            target_latent: 1 x dim
            pred_latent: times_augment_pred_image x dim
        """
        assert target_latent.shape[0] == 1
        target_latent = target_latent / torch.linalg.vector_norm(target_latent, dim=-1, keepdim=True)
        pred_latent = pred_latent / torch.linalg.vector_norm(pred_latent, dim=-1, keepdim=True)
        sims = torch.mm(pred_latent, target_latent.t()).squeeze(-1)
        return -sims.mean()


class MSELatent(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, target_latent, pred_latent):
        """
        Inputs: predicted and ground truth latents
            target_latent: bs x dim
            pred_latent: bs x dim
        """
        return self.loss(pred_latent, target_latent)


class NLLLatent(nn.Module):
    def __init__(self, alpha_nll):
        super().__init__()
        self.alpha_nll = alpha_nll
        self.loss_y = nn.MSELoss()

    def forward(self, target_z, pred_z_gmm, target_y=None, pred_y=None):
        """
        Inputs: predicted and ground truth latents and GMM
            target_z: bs x z_dim
            pred_z_gmm: torch.distributions.Distribution
            target_y: bs x y_dim or None
            pred_y: bs x y_dim or None
        """
        loss = -self.alpha_nll*pred_z_gmm.log_prob(target_z).mean()

        if target_y is not None and pred_y is not None:
            loss += self.loss_y(pred_y, target_y)
        
        return loss


class Aesthetic(nn.Module):
    def __init__(self, clip_arch):
        super().__init__()
        self.aesthetic_predictor = get_aesthetic_model(clip_arch)
        self.aesthetic_predictor.eval()
        for p in self.aesthetic_predictor.parameters():
            p.requires_grad = False

    def forward(self, clip_latent):
        """
        Inputs: clip image latents
            clip_latent: bs x dim 
        """
        clip_latent = clip_latent / torch.linalg.vector_norm(clip_latent, dim=-1, keepdim=True)
        loss = -self.aesthetic_predictor(clip_latent).mean()
        return loss


class CosineSimLatentAndAesthetic(nn.Module):
    def __init__(self, clip_arch, alpha_aesthetic):
        super().__init__()
        self.alpha_aesthetic = alpha_aesthetic
        self.cosine_latent_loss = CosineSimLatent()
        self.aesthetic_loss = Aesthetic(clip_arch)

    def forward(self, target_latent, pred_latent):
        """
        Inputs: predicted and ground truth latents
            target_latent: bs x dim
            pred_latent: bs x dim
        """
        cosine_latent_loss = self.cosine_latent_loss(target_latent, pred_latent) 
        aesthetic_loss = self.aesthetic_loss(pred_latent)
        loss = cosine_latent_loss + self.alpha_aesthetic * aesthetic_loss
        return cosine_latent_loss, loss


class AugCosineSimLatentAndAesthetic(nn.Module):
    def __init__(self, clip_arch, alpha_aesthetic):
        super().__init__()
        self.alpha_aesthetic = alpha_aesthetic
        self.aug_cosine_latent_loss = AugCosineSimLatent()
        self.aesthetic_loss = Aesthetic(clip_arch)

    def forward(self, target_latent, pred_latent):
        """
        Inputs: predicted and ground truth latents
            target_latent: bs x dim
            pred_latent: bs x dim
        """
        aug_cosine_latent_loss = self.aug_cosine_latent_loss(target_latent, pred_latent) 
        aesthetic_loss = self.aesthetic_loss(pred_latent)
        loss = aug_cosine_latent_loss + self.alpha_aesthetic * aesthetic_loss
        return aug_cosine_latent_loss, loss


class LossFactory:
    @staticmethod
    def get_loss(loss_name, config):
        if loss_name == 'cosine_latent':
            return CosineSimLatent()
        elif loss_name == 'aug_cosine_latent':
            return AugCosineSimLatent()
        elif loss_name == 'mse_latent':
            return MSELatent()
        elif loss_name == 'nll_latent':
            return NLLLatent(config.alpha_nll)
        elif loss_name == 'cosine_latent+aesthetic':
            return CosineSimLatentAndAesthetic(config.clip_arch, config.alpha_aesthetic)
        elif loss_name == 'aug_cosine_latent+aesthetic':
            return AugCosineSimLatentAndAesthetic(config.clip_arch, config.alpha_aesthetic)
        else:
            raise NotImplementedError
