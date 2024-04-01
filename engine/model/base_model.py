import abc
from typing import Iterator
from yacs.config import CfgNode
import torch
from engine.slover import build_optimizer_with_gradient_clipping, EmptyLRScheduler
import engine.checkpoint.functional as checkpoint_f
import logging
import engine.slover.lr_scheduler as engine_scheduler
from .import_optimizer import import_optimizer
from .import_scheduler import import_scheduler
from .build import BUILD_MODEL_REGISTRY
from .ema import EMA
from .build import build_network


def lower(k):
    new_k = k.lower()
    if new_k.upper() != k:
        new_k = k
    return new_k


def ema_wrapper(func):
    def ema_func(cls, data):
        if hasattr(cls, cls.EMA_KEY):
            getattr(cls, cls.EMA_KEY).store(cls.g_model.parameters())
            getattr(cls, cls.EMA_KEY).copy(cls.g_model.parameters())
            result = func(cls, data)
            getattr(cls, cls.EMA_KEY).copy_back(cls.g_model.parameters())
        else:
            result = func(cls, data)
        return result

    return ema_func


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

    EMA_KEY = 'ema'

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
        self.g_optimizer = self.create_optimizer(cfg.SOLVER.GENERATOR.OPTIMIZER, cfg.SOLVER.GENERATOR.CLIP_GRADIENTS, self.g_model.parameters())
        self.g_scheduler = self.create_scheduler(cfg.SOLVER.GENERATOR.LR_SCHEDULER, self.g_optimizer)
        self.create_ema(cfg.SOLVER.get('EMA', dict()))

        logging.getLogger(self.default_log_name).info('create model {} with {}'.format(self.__class__.__name__, self.g_model))
        return

    def create_optimizer(self, optimizer_cfg, clip_gradients_cfg, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        op_type = optimizer_cfg.TYPE
        params = dict()
        for key, param in optimizer_cfg.PARAMS.items():
            params[lower(key)] = param

        cls = import_optimizer(op_type)

        return build_optimizer_with_gradient_clipping(clip_gradients_cfg, cls)(parameters, **params)

    def create_scheduler(self, scheduler_cfg, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

        if not scheduler_cfg.ENABLED:
            return EmptyLRScheduler(optimizer)

        op_type = scheduler_cfg.TYPE
        params = dict()
        for key, param in scheduler_cfg.PARAMS.items():
            if isinstance(param, dict):
                sub_param = dict()
                for sk, sp in param.items():
                    sub_param[lower(sk)] = sp
                params[lower(key)] = sub_param
            else:
                params[lower(key)] = param

        if hasattr(engine_scheduler, op_type):
            cls = getattr(engine_scheduler, op_type)
        else:
            cls = import_scheduler(op_type)
        return cls(optimizer, **params)

    def create_model(self, params) -> torch.nn.Module:
        return build_network(params)

    def create_ema(self, params):
        if not params.get('ENABLED', False):
            return
        assert hasattr(self, 'g_model')

        kwargs = dict()
        for k, v in params.items():
            if k.lower() == 'enabled':
                continue
            kwargs[k.lower()] = v

        ema = EMA(self.g_model.parameters(), **kwargs)
        ema.to(self.device)
        setattr(self, self.EMA_KEY, ema)
        return

    @abc.abstractmethod
    def run_step(self, data, *, epoch=None, **kwargs):
        """
        :param data: type is dict
        :param epoch:
        :param kwargs: another args
        :return:
        """
        raise NotImplemented('the run_step must be implement')

    @ema_wrapper
    def generator_imp(self, data):
        return self.generator(data)

    @abc.abstractmethod
    def generator(self, data):
        """
        :param data: type is dict
        :return:
        """
        raise NotImplemented('the generator must be implement')

    def __normal_step(self, loss: torch.Tensor, epoch: int):
        self.g_optimizer.zero_grad()
        loss.backward()
        self.g_optimizer.step()

        if hasattr(self, self.EMA_KEY):
            getattr(self, self.EMA_KEY).update(self.g_model.parameters())

        self.g_scheduler.step(epoch)
        return

    def __accumulation_step(self, loss: torch.Tensor, epoch: int, accumulation_epoch: int, data_eopch: int):
        loss = loss / accumulation_epoch
        loss.backward()  # pytorch 的 backward 做的是梯度累加
        if 0 == (data_eopch + 1) % accumulation_epoch:
            self.g_optimizer.step()
            self.g_optimizer.zero_grad()

            if hasattr(self, self.EMA_KEY):
                getattr(self, self.EMA_KEY).update(self.g_model.parameters())

            self.g_scheduler.step(epoch)
        return

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
        if self.g_model.training:
            loss = self.run_step(data=data, epoch=epoch, **kwargs)

            accumulation_epoch = kwargs.get('accumulation_epoch', -1)

            if accumulation_epoch > 1:
                data_epoch = kwargs.get('data_epoch', 0)
                self.__accumulation_step(loss, epoch, accumulation_epoch, data_epoch)
            else:
                self.__normal_step(loss, epoch)

            lr = '*'.join([str(lr) for lr in self.g_scheduler.get_last_lr()])
            return {'total_loss': loss.detach().item(), 'lr': lr}
        else:
            return self.generator_imp(data)

    def enable_train(self):
        self.g_model.train()
        return

    def disable_train(self):
        self.g_model.eval()
        return

    def get_state_dict(self):
        state_dict = dict()
        state_dict['g_model'] = checkpoint_f.get_model_state_dict(self.g_model)
        if hasattr(self, self.EMA_KEY):
            state_dict[self.EMA_KEY] = getattr(self, self.EMA_KEY).state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_model_state_dict(self.g_model, state_dict['g_model'], log_name=self.default_log_name)
        if hasattr(self, self.EMA_KEY) and self.EMA_KEY in state_dict.keys():
            getattr(self, self.EMA_KEY).load_state_dict(state_dict[self.EMA_KEY])
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

    def sync_batch_norm(self):
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.g_model)

    def enable_dirstribute_ddp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.TRAINER.PARADIGM.GPU_ID))
        model = self.sync_batch_norm()
        self.g_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
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
