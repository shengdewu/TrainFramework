import abc
from engine.model.base_model import BaseModel
from engine.slover import build_optimizer_with_gradient_clipping
from engine.slover.lr_scheduler import LRMultiplierScheduler, EmptyLRScheduler
from typing import Iterator
import torch
import engine.checkpoint.functional as checkpoint_f
import logging
from .import_optimizer import import_optimizer
from .import_scheduler import import_scheduler


class BaseGanModel(BaseModel):
    def __init__(self, cfg):
        super(BaseGanModel, self).__init__(cfg)

        self.d_model = self.create_d_model(cfg).to(cfg.MODEL.DEVICE)
        self.d_optimizer = self.create_d_optimizer(cfg, self.d_model.parameters())
        self.d_scheduler = self.create_g_scheduler(cfg, self.d_optimizer)
        return

    def create_d_optimizer(self, cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        op_type = cfg.SOLVER.OPTIMIZER.DISCRIMINATOR.TYPE
        params = dict()
        for key, param in cfg.SOLVER.OPTIMIZER.DISCRIMINATOR.PARAMS.items():
            params[key.lower()] = param

        cls = import_optimizer(op_type)

        return build_optimizer_with_gradient_clipping(cfg, cls)(parameters, **params)

    def create_d_scheduler(self, cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

        if not cfg.SOLVER.LR_SCHEDULER.DISCRIMINATOR.ENABLED:
            return EmptyLRScheduler(optimizer)

        op_type = cfg.SOLVER.LR_SCHEDULER.DISCRIMINATOR.TYPE
        params = dict()
        for key, param in cfg.SOLVER.LR_SCHEDULER.DISCRIMINATOR.PARAMS.items():
            if isinstance(param, dict):
                sub_param = dict()
                for sk, sp in param.items():
                    sub_param[sk.lower()] = sp
                params[key.lower()] = sub_param
            else:
                params[key.lower()] = param

        if op_type == 'LRMultiplierScheduler':
            params['max_iter'] = cfg.SOLVER.MAX_ITER
            cls = LRMultiplierScheduler
        else:
            cls = import_scheduler(op_type)
        return cls(optimizer, **params)

    @abc.abstractmethod
    def create_d_model(self, cfg) -> torch.nn.Module:
        raise NotImplemented('the create_d_model must be implement')

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
        self.g_scheduler.step(epoch)
        self.d_scheduler.step(epoch)

        loss_dict['g_learning_rate'] = '*'.join([str(lr) for lr in self.g_scheduler.get_last_lr()])
        loss_dict['d_learning_rate'] = '*'.join([str(lr) for lr in self.d_scheduler.get_last_lr()])
        return loss_dict

    def enable_train(self):
        self.g_model.train()
        self.d_model.train()
        return

    def disable_train(self):
        self.g_model.eval()
        self.d_model.eval()
        return

    def get_addition_state_dict(self):
        state_dict = dict()
        state_dict['g_optimizer'] = checkpoint_f.get_model_state_dict(self.g_optimizer)
        state_dict['g_scheduler'] = checkpoint_f.get_model_state_dict(self.g_scheduler)
        state_dict['d_optimizer'] = checkpoint_f.get_model_state_dict(self.d_optimizer)
        state_dict['d_scheduler'] = checkpoint_f.get_model_state_dict(self.d_scheduler)
        state_dict['d_model'] = checkpoint_f.get_model_state_dict(self.d_model)
        state_dict['cfg'] = self.cfg.dump()
        return state_dict

    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_checkpoint_state_dict(self.g_optimizer, state_dict['g_optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.g_scheduler, state_dict['g_scheduler'])
        checkpoint_f.load_checkpoint_state_dict(self.d_optimizer, state_dict['d_optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.d_scheduler, state_dict['d_scheduler'])
        checkpoint_f.load_model_state_dict(self.d_model, state_dict['d_model'], log_name=self.default_log_name)
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
            self.g_model = torch.nn.parallel.DistributedDataParallel(self.g_model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
            self.d_model = torch.nn.parallel.DistributedDataParallel(self.d_model, device_ids=[cfg.MODEL.TRAINER.GPU_ID])
        elif cfg.MODEL.TRAINER.TYPE == 0:
            logging.getLogger(self.default_log_name).info('launch model by parallel')
            self.g_model = torch.nn.parallel.DataParallel(self.g_model)
            self.d_model = torch.nn.parallel.DataParallel(self.d_model)
        else:
            logging.getLogger(self.default_log_name).info('launch model by stand alone machine')
        return
