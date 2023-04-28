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
from tr0n.modules.models.loss import LossFactory
from tr0n.modules.trainers.trainer_factory import TrainerFactory

def main():
    config = parse_args(keep_prev=True)
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
     
    val_data_loader = DataFactory.get_loader(config, train=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelFactory.get_model(config, device)

    val_loss = LossFactory.get_loss(config.val_loss, config)

    trainer = TrainerFactory.get_trainer(model, None, val_loss, None,
                      dataset_name=config.dataset_name,
                      dec_type=config.dec_type,
                      num_interps=config.num_interps,
                      num_hybrid_iters=config.num_hybrid_iters,
                      config=config,
                      train_data_loader=None,
                      val_data_loader=val_data_loader,
                      lr_schedulers=None,
                      device=device,
                      writer=None)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")    
    
    trainer.evaluate()


if __name__ == '__main__':
    main()
