import abc
from typing import Iterator
import torch
import engine.checkpoint.functional as checkpoint_f
import logging


class BaseModel(abc.ABC):
    """
    1. Must be Create model and optimizer in constructer
    2. loss.backward() in run_step
    3. return dict of loss item in run_step
    4. Provide inference interface named generator
    5. provide disable/enable train function named enable_train and disable_train
    6. Provide model status acquisition and setting interface named get_state_dict and load_state_dict
    7. Provide optimizer status acquisition and setting interface named get_addition_state_dict and load_addition_state_dict
    8. may be create scheduler in constructer
    9. run_step and schedule in __call__
    """
    def __init__(self, cfg):
        """
        eg:
            from engine.slover import build_optimizer_with_gradient_clipping, build_lr_scheduler

            self.model = ResNet()
            self.optimizer = build_optimizer_with_gradient_clipping(cfg, torch.optim.SGD)(
                self.model.parameters(),
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY
            )
            self.scheduler = build_lr_scheduler(cfg, self.optimizer)
        """
        self.model = self.create_model(cfg).to(cfg.MODEL.DEVICE)
        self.optimizer = self.create_optimizer(cfg, self.model.parameters())
        self.scheduler = self.create_scheduler(cfg, self.optimizer)
        self.default_log_name = cfg.OUTPUT_LOG_NAME
        return

    @abc.abstractmethod
    def create_optimizer(self, cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        raise NotImplemented('the create_optimizer must be implement')

    @abc.abstractmethod
    def create_scheduler(self, cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        raise NotImplemented('the create_scheduler must be implement')

    @abc.abstractmethod
    def create_model(self, cfg) -> torch.nn.Module:
        raise NotImplemented('the create_model must be implement')

    @abc.abstractmethod
    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        raise NotImplemented('the run_step must be implement')

    @abc.abstractmethod
    def generator(self, data):
        """
        :param data: type is dict
        :return:
        """
        raise NotImplemented('the generator must be implement')

    def __call__(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        eg:
                loss_dict = self.run_step(data, **kwargs)
                self.scheduler.step(epoch)
        """
        loss_dict = self.run_step(data=data, epoch=epoch, **kwargs)
        self.scheduler.step(epoch)

        loss_dict['learning_rate'] = '*'.join([str(lr) for lr in self.scheduler.get_last_lr()])
        return loss_dict

    def enable_train(self):
        self.model.train()
        return

    def disable_train(self):
        self.model.eval()
        return

    def get_state_dict(self):
        state_dict = dict()
        state_dict['model'] = checkpoint_f.get_model_state_dict(self.model)
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_model_state_dict(self.model, state_dict['model'], log_name=self.default_log_name)
        return

    def get_addition_state_dict(self):
        state_dict = dict()
        state_dict['optimizer'] = checkpoint_f.get_model_state_dict(self.optimizer)
        state_dict['scheduler'] = checkpoint_f.get_model_state_dict(self.scheduler)
        return state_dict

    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_checkpoint_state_dict(self.optimizer, state_dict['optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.scheduler, state_dict['scheduler'])
        return

    def enable_distribute(self, cfg):
        """
        :param cfg:
        :return:

        eg:
            if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
                logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
            elif cfg.MODEL.TRAINER.TYPE == 0:
                logging.getLogger(__name__).info('launch model by parallel')
                model = torch.nn.parallel.DataParallel(model)
            else:
                logging.getLogger(__name__).info('launch model by singal machine')

        """
        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(self.default_log_name).info('launch model by parallel')
            self.model = torch.nn.parallel.DataParallel(self.model)
        else:
            logging.getLogger(self.default_log_name).info('launch model by stand alone machine')
        return