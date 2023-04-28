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
from torch.utils.tensorboard.writer import SummaryWriter
from tr0n.modules.datasets.data_factory import DataFactory
from tr0n.modules.models.model_factory import ModelFactory
from tr0n.modules.models.loss import LossFactory
from tr0n.modules.optimizers.optimizer_factory import OptimizerFactory, SchedulerFactory
from tr0n.modules.trainers.trainer_factory import TrainerFactory 

def main():
    config = parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    train_data_loader = DataFactory.get_loader(config, train=True)
    val_data_loader  = DataFactory.get_loader(config, train=False)
    
    if hasattr(train_data_loader.dataset, 'cov'):
        clip_noise_cov = train_data_loader.dataset.cov
    else:
        clip_noise_cov = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelFactory.get_model(config, device, clip_noise_cov)
     
    params_translator = list(model.translator.parameters())
    optimizer_translator = OptimizerFactory.get_optimizer(config.optim, params_translator, device, config.lr, config.momentum, config.weight_decay, config.sgld_noise_std)
    num_training_steps = len(train_data_loader) * config.num_epochs
    scheduler_translator = SchedulerFactory.get_scheduler(config.scheduler, optimizer_translator, num_training_steps)
        
    optimizers={'translator': optimizer_translator}
    schedulers={'translator': scheduler_translator}

    train_loss = LossFactory.get_loss(config.train_loss, config)
    val_loss = LossFactory.get_loss(config.val_loss, config)
    
    trainer = TrainerFactory.get_trainer(model, train_loss, val_loss, optimizers,
                      dataset_name=config.dataset_name,
                      dec_type=config.dec_type,
                      config=config,
                      train_data_loader=train_data_loader,
                      val_data_loader=val_data_loader,
                      lr_schedulers=schedulers,
                      device=device,
                      writer=writer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")

    trainer.train()


if __name__ == '__main__':
    main()
