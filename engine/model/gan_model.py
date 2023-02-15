import abc
from engine.model.base_model import BaseModel
import torch
import engine.checkpoint.functional as checkpoint_f
import logging
from yacs.config import CfgNode


class BaseGanModel(BaseModel):
    def __init__(self, cfg):
        super(BaseGanModel, self).__init__(cfg)

        d_model_cfg = cfg.TRAINER.MODEL.DISCRIMINATOR
        if isinstance(d_model_cfg, list):
            d_model = dict()
            for d_cfg in d_model_cfg:
                assert isinstance(d_cfg, dict)
                assert len(d_cfg) == 2
                d_name = d_cfg['NAME']
                d_model[d_name] = self.create_d_model(params=d_cfg['PARAMS']).to(self.device)
        else:
            d_model = self.create_d_model(params=cfg.TRAINER.MODEL.DISCRIMINATOR).to(self.device)

        self.d_model = d_model

        d_solver_cfg = cfg.SOLVER.DISCRIMINATOR
        if isinstance(self.d_model, dict):
            assert isinstance(d_solver_cfg, list)
            d_optimizer = dict()
            d_scheduler = dict()
            for d_cfg in d_solver_cfg:
                assert isinstance(d_cfg, dict)
                assert len(d_cfg) == 2
                d_name = d_cfg['NAME']
                d_optimizer[d_name] = self.create_optimizer(CfgNode(init_dict=d_cfg['PARAMS']['OPTIMIZER']), self.d_model[d_name].parameters())
                d_scheduler[d_name] = self.create_scheduler(CfgNode(init_dict=d_cfg['PARAMS']['LR_SCHEDULER']), d_optimizer[d_name])

        else:
            assert isinstance(cfg.SOLVER.DISCRIMINATOR, dict)
            d_optimizer = self.create_optimizer(CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['OPTIMIZER']), self.d_model.parameters())
            d_scheduler = self.create_scheduler(CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['LR_SCHEDULER']), d_optimizer)

        self.d_optimizer = d_optimizer
        self.d_scheduler = d_scheduler
        return

    @abc.abstractmethod
    def create_d_model(self, params) -> torch.nn.Module:
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
        if isinstance(self.d_scheduler, dict):
            for k, d_schedler in self.d_scheduler.items():
                d_schedler.step(epoch)
        else:
            self.d_scheduler.step(epoch)

        loss_dict['g_learning_rate'] = '*'.join([str(lr) for lr in self.g_scheduler.get_last_lr()])
        if isinstance(self.d_scheduler, dict):
            loss_dict['d_learning_rate'] = list()
            for k, d_schedler in self.d_scheduler.items():
                loss_dict['d_learning_rate'].append({k: '*'.join([str(lr) for lr in d_schedler.get_last_lr()])})
        else:
            loss_dict['d_learning_rate'] = '*'.join([str(lr) for lr in self.d_scheduler.get_last_lr()])
        return loss_dict

    def enable_train(self):
        self.g_model.train()
        if isinstance(self.d_model, dict):
            for k, d_model in self.d_model.items():
                d_model.train()
        else:
            self.d_model.train()
        return

    def disable_train(self):
        self.g_model.eval()
        if isinstance(self.d_model, dict):
            for k, d_model in self.d_model.items():
                d_model.eval()
        else:
            self.d_model.eval()
        return

    def get_addition_state_dict(self):
        state_dict = dict()
        state_dict['g_optimizer'] = checkpoint_f.get_model_state_dict(self.g_optimizer)
        state_dict['g_scheduler'] = checkpoint_f.get_model_state_dict(self.g_scheduler)
        if isinstance(self.d_model, dict):
            for k, d_model in self.d_model.items():
                state_dict['d_model_{}'.format(k)] = checkpoint_f.get_model_state_dict(d_model)
                state_dict['d_optimizer_{}'] = checkpoint_f.get_model_state_dict(self.d_optimizer[k])
                state_dict['d_scheduler_{}'] = checkpoint_f.get_model_state_dict(self.d_scheduler[k])
        else:
            state_dict['d_model'] = checkpoint_f.get_model_state_dict(self.d_model)
            state_dict['d_optimizer'] = checkpoint_f.get_model_state_dict(self.d_optimizer)
            state_dict['d_scheduler'] = checkpoint_f.get_model_state_dict(self.d_scheduler)
        state_dict['cfg'] = self.cfg.dump()
        return state_dict

    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_checkpoint_state_dict(self.g_optimizer, state_dict['g_optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.g_scheduler, state_dict['g_scheduler'])

        if isinstance(self.d_model, dict):
            for k, d_model in self.d_model.items():
                checkpoint_f.load_model_state_dict(d_model,  state_dict['d_model_{}'.format(k)], log_name=self.default_log_name)
                checkpoint_f.load_model_state_dict(self.d_optimizer[k], state_dict['d_optimizer_{}'.format(k)], log_name=self.default_log_name)
                checkpoint_f.load_model_state_dict(self.d_scheduler[k], state_dict['d_scheduler_{}'.format(k)], log_name=self.default_log_name)
        else:
            checkpoint_f.load_model_state_dict(self.d_model, state_dict['d_model'], log_name=self.default_log_name)
            checkpoint_f.load_checkpoint_state_dict(self.d_optimizer, state_dict['d_optimizer'])
            checkpoint_f.load_checkpoint_state_dict(self.d_scheduler, state_dict['d_scheduler'])
        return

    def enable_distribute(self, cfg):
        """
        :param cfg:
        :return:

        eg:
            if cfg.TRAINER.PARADIGM.TYPE == 1 and cfg.TRAINER.PARADIGM.GPU_ID >= 0:
                logging.getLogger(__name__).info('launch model by distribute in gpu_id {}'.format(cfg.TRAINER.PARADIGM.GPU_ID))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
            elif cfg.TRAINER.PARADIGM.TYPE == 0:
                logging.getLogger(__name__).info('launch model by parallel')
                model = torch.nn.parallel.DataParallel(model)
            else:
                logging.getLogger(__name__).info('launch model by singal machine')

        """
        if cfg.TRAINER.PARADIGM.TYPE == 'DDP' and cfg.TRAINER.PARADIGM.GPU_ID >= 0:
            logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.TRAINER.PARADIGM.GPU_ID))
            self.g_model = torch.nn.parallel.DistributedDataParallel(self.g_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
            if isinstance(self.d_model, dict):
                new_d_model = dict()
                for k, d_model in self.d_model.items():
                    new_d_model[k] = torch.nn.parallel.DistributedDataParallel(d_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
                self.d_model = new_d_model
            else:
                self.d_model = torch.nn.parallel.DistributedDataParallel(self.d_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
        elif cfg.TRAINER.PARADIGM.TYPE == 0:
            logging.getLogger(self.default_log_name).info('launch model by parallel')
            self.g_model = torch.nn.parallel.DataParallel(self.g_model)
            if isinstance(self.d_model, dict):
                new_d_model = dict()
                for k, d_model in self.d_model.items():
                    new_d_model[k] = torch.nn.parallel.DataParallel(d_model)
                self.d_model = new_d_model
            else:
                self.d_model = torch.nn.parallel.DataParallel(self.d_model)
        else:
            logging.getLogger(self.default_log_name).info('launch model by stand alone machine')
        return
