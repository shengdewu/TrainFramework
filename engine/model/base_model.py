import abc
from typing import Iterator
import torch
from engine.slover import build_optimizer_with_gradient_clipping, build_lr_scheduler
import engine.checkpoint.functional as checkpoint_f
import logging
from .import_optimizer import import_optimizer


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
            self.g_optimizer = build_optimizer_with_gradient_clipping(cfg, torch.optim.SGD)(
                self.model.parameters(),
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY
            )
            self.g_scheduler = build_lr_scheduler(cfg, self.g_optimizer)
        """
        self.g_model = self.create_g_model(cfg).to(cfg.MODEL.DEVICE)
        self.g_optimizer = self.create_g_optimizer(cfg, self.g_model.parameters())
        self.g_scheduler = self.create_g_scheduler(cfg, self.g_optimizer)
        self.default_log_name = cfg.OUTPUT_LOG_NAME
        self.device = cfg.MODEL.DEVICE
        logging.getLogger(self.default_log_name).info(f'create {self.__class__.__name__}')
        return

    def create_g_optimizer(self, cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        op_type = cfg.SOLVER.OPTIMIZER.GENERATOR.TYPE
        params = dict()
        for key, param in cfg.SOLVER.OPTIMIZER.GENERATOR.PARAMS.items():
            params[key.lower()] = param

        cls = import_optimizer(op_type)

        return build_optimizer_with_gradient_clipping(cfg, cls)(parameters, **params)

    def create_g_scheduler(self, cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        return build_lr_scheduler(cfg, optimizer)

    @abc.abstractmethod
    def create_g_model(self, cfg) -> torch.nn.Module:
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
                self.g_scheduler.step(epoch)
        """
        loss_dict = self.run_step(data=data, epoch=epoch, **kwargs)
        self.g_scheduler.step(epoch)

        loss_dict['learning_rate'] = '*'.join([str(lr) for lr in self.g_scheduler.get_last_lr()])
        return loss_dict

    def enable_train(self):
        self.g_model.train()
        return

    def disable_train(self):
        self.g_model.eval()
        return

    def get_state_dict(self):
        state_dict = dict()
        state_dict['g_model'] = checkpoint_f.get_model_state_dict(self.g_model)
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_model_state_dict(self.g_model, state_dict['g_model'], log_name=self.default_log_name)
        return

    def get_addition_state_dict(self):
        state_dict = dict()
        state_dict['g_optimizer'] = checkpoint_f.get_model_state_dict(self.g_optimizer)
        state_dict['g_scheduler'] = checkpoint_f.get_model_state_dict(self.g_scheduler)
        return state_dict

    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_checkpoint_state_dict(self.g_optimizer, state_dict['g_optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.g_scheduler, state_dict['g_scheduler'])
        return

    def enable_distribute(self, cfg):
        """
        :param cfg:
        :return:

        eg:
            if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
                logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.g_model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
            elif cfg.MODEL.TRAINER.TYPE == 0:
                logging.getLogger(__name__).info('launch model by parallel')
                model = torch.nn.parallel.DataParallel(model)
            else:
                logging.getLogger(__name__).info('launch model by singal machine')

        """
        if cfg.MODEL.TRAINER.TYPE == 1 and cfg.MODEL.TRAINER.GPU_ID >= 0:
            logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.MODEL.TRAINER.GPU_ID))
            self.g_model = torch.nn.parallel.DistributedDataParallel(self.g_model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(self.default_log_name).info('launch model by parallel')
            self.g_model = torch.nn.parallel.DataParallel(self.g_model)
        else:
            logging.getLogger(self.default_log_name).info('launch model by stand alone machine')
        return