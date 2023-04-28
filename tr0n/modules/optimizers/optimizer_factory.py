from modules.optimizers.sgld import SGLD
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class OptimizerFactory:
    @staticmethod
    def get_optimizer(optim, params, device, lr, momentum=None, weight_decay=None, sgld_noise_std=None):
        if optim == "sgd":
            return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optim == "adam":
            return Adam(params, lr=lr, weight_decay=weight_decay)
        elif optim == "sgld":
            return SGLD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, noise_std=sgld_noise_std, device=device)
        else:
            raise NotImplementedError

class SchedulerFactory:
    @staticmethod
    def get_scheduler(scheduler, optimizer, num_training_steps):
        if scheduler is None:
            return None
        elif scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, num_training_steps)
        else:
            raise NotImplementedError
