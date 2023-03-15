import abc
from typing import Iterator
from yacs.config import CfgNode
import torch
from engine.slover import build_optimizer_with_gradient_clipping, LRMultiplierScheduler, EmptyLRScheduler
import engine.checkpoint.functional as checkpoint_f
import logging
from .import_optimizer import import_optimizer
from .import_scheduler import import_scheduler
from .build import BUILD_MODEL_REGISTRY


@BUILD_MODEL_REGISTRY.register()
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
        assert isinstance(cfg, CfgNode)
        self.default_log_name = cfg.OUTPUT_LOG_NAME
        self.cfg = cfg.clone()
        self.device = cfg.TRAINER.DEVICE
        self.max_iter = cfg.SOLVER.MAX_ITER

        self.g_model = self.create_model(params=cfg.TRAINER.MODEL.get('GENERATOR', dict())).to(self.device)
        self.g_optimizer = self.create_optimizer(cfg.SOLVER.GENERATOR.OPTIMIZER, self.g_model.parameters())
        self.g_scheduler = self.create_scheduler(cfg.SOLVER.GENERATOR.LR_SCHEDULER, self.g_optimizer)

        logging.getLogger(self.default_log_name).info('create model {} with {}'.format(self.__class__.__name__, self.g_model))
        return

    def create_optimizer(self, optimizer_cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        op_type = optimizer_cfg.TYPE
        params = dict()
        for key, param in optimizer_cfg.PARAMS.items():
            params[key.lower()] = param

        cls = import_optimizer(op_type)

        return build_optimizer_with_gradient_clipping(optimizer_cfg.CLIP_GRADIENTS, cls)(parameters, **params)

    def create_scheduler(self, scheduler_cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

        if not scheduler_cfg.ENABLED:
            return EmptyLRScheduler(optimizer)

        op_type = scheduler_cfg.TYPE
        params = dict()
        for key, param in scheduler_cfg.PARAMS.items():
            if isinstance(param, dict):
                sub_param = dict()
                for sk, sp in param.items():
                    sub_param[sk.lower()] = sp
                params[key.lower()] = sub_param
            else:
                params[key.lower()] = param

        if op_type == 'LRMultiplierScheduler':
            params['max_iter'] = self.max_iter
            cls = LRMultiplierScheduler
        else:
            cls = import_scheduler(op_type)
        return cls(optimizer, **params)

    @abc.abstractmethod
    def create_model(self, params) -> torch.nn.Module:
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
        state_dict['cfg'] = self.cfg.dump()
        return state_dict

    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_checkpoint_state_dict(self.g_optimizer, state_dict['g_optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.g_scheduler, state_dict['g_scheduler'])
        return

    def enable_dirstribute_ddp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.TRAINER.PARADIGM.GPU_ID))
        self.g_model = torch.nn.parallel.DistributedDataParallel(self.g_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
        return

    def enable_distribute_dp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by parallel')
        self.g_model = torch.nn.parallel.DataParallel(self.g_model)
        return

    def enable_distribute(self, cfg):
        """
        :param cfg:
        :return:
        """
        if cfg.TRAINER.PARADIGM.TYPE == 'DDP' and cfg.TRAINER.PARADIGM.GPU_ID is not None:
            self.enable_dirstribute_ddp(cfg)
        elif cfg.TRAINER.PARADIGM.TYPE == 'DP':
            self.enable_distribute_dp(cfg)
        else:
            logging.getLogger(self.default_log_name).info('launch model by stand alone machine')
        return
