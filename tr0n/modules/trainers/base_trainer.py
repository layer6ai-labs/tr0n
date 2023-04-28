import torch
import os
from abc import abstractmethod

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, train_loss, val_loss, optimizers, config, device, lr_schedulers=None, writer=None):
        self.config = config
        self.device = device
        self.model = model.to(self.device)

        self.train_loss = train_loss.to(self.device) if train_loss is not None else None
        self.val_loss = val_loss.to(self.device)
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.start_epoch = 1
        self.global_step = 0

        self.num_epochs = config.num_epochs
        self.writer = writer
        self.save_checkpoint_dir = config.model_path
        self.load_checkpoint_dir = config.load_path
        self.save_image_dir = config.save_image_path

        self.log_step = config.log_step

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _val_epoch(self, epoch):
        """
        Validation logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError


    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            result = self._train_epoch(epoch)
            if epoch % self.config.save_every == 0:
                    self._save_checkpoint(epoch, save_best=False)
        self.writer.close()

    def evaluate(self):
        self._val_epoch(0)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, save checkpoint to 'model_best.pth'
        """

        state = {
            'translator_state_dict': self.model.translator.state_dict(),
        }
        
        if save_best:
            best_path = os.path.join(self.save_checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            filename = os.path.join(self.save_checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))


    def load_checkpoint(self, model_name):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        load_path = os.path.join(self.load_checkpoint_dir, model_name)
        print("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)

        translator_state_dict = checkpoint['translator_state_dict']
        self.model.translator.load_state_dict(translator_state_dict)

        print("Checkpoint loaded")
