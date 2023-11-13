import abc
from typing import Union, Dict, Callable
from engine.model.base_model import BaseModel
import engine.checkpoint.functional as checkpoint_f
from .build import build_network
import torch
import logging
from yacs.config import CfgNode


class BaseGanModel(BaseModel, abc.ABC):
    DEFAULT_DMODEL_NAME = 'default'

    def __init__(self, cfg):
        super(BaseGanModel, self).__init__(cfg)

        self.d_model = dict()
        d_model_cfg = cfg.TRAINER.MODEL.get('DISCRIMINATOR', dict())
        if isinstance(d_model_cfg, list):
            for d_cfg in d_model_cfg:
                assert isinstance(d_cfg, dict)
                assert len(d_cfg) >= 1
                d_name = d_cfg['NAME']
                self.d_model[d_name] = self.create_d_model(params=d_cfg.get('PARAMS', dict())).to(self.device)
        else:
            self.d_model[self.DEFAULT_DMODEL_NAME] = self.create_d_model(params=d_model_cfg).to(self.device)

        self.d_optimizer = dict()
        self.d_scheduler = dict()
        d_solver_cfg = cfg.SOLVER.DISCRIMINATOR
        if isinstance(d_solver_cfg, list):
            for d_cfg in d_solver_cfg:
                assert isinstance(d_cfg, dict)
                assert len(d_cfg) == 2
                d_name = d_cfg['NAME']
                self.d_optimizer[d_name] = self.create_optimizer(CfgNode(init_dict=d_cfg['PARAMS']['OPTIMIZER']), self.d_model[d_name].parameters())
                self.d_scheduler[d_name] = self.create_scheduler(CfgNode(init_dict=d_cfg['PARAMS']['LR_SCHEDULER']), self.d_optimizer[d_name])

        else:
            assert isinstance(cfg.SOLVER.DISCRIMINATOR, dict)
            self.d_optimizer[self.DEFAULT_DMODEL_NAME] = self.create_optimizer(CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['OPTIMIZER']),
                                                                               self.d_model[self.DEFAULT_DMODEL_NAME].parameters())
            self.d_scheduler[self.DEFAULT_DMODEL_NAME] = self.create_scheduler(CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['LR_SCHEDULER']),
                                                                               self.d_optimizer[self.DEFAULT_DMODEL_NAME])

        assert len(set(self.d_model.keys()).intersection(self.d_optimizer.keys())) == len(self.d_model.keys()), f'the d_model not match the d_optimizer/d_scheduler name'
        return

    def create_d_model(self, params) -> torch.nn.Module:
        return build_network(params)

    def _dmodel_forward(self, d_model_input: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        assert len(set(d_model_input.keys()).intersection(self.d_model.keys())) == len(self.d_model.keys()), f'the name of input != the name of self.d_model'

        output = dict()
        for key, param in d_model_input.items():
            output[key] = self.d_model[key](**param)

        return output

    def dmodel_forward(self, d_model_input: Union[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        :param d_model_input: dict(d_mode_key=dict())
        :return:
        """
        if isinstance(d_model_input[list(d_model_input.keys())[0]], torch.Tensor):
            new_input = {self.DEFAULT_DMODEL_NAME: d_model_input}
            output = self._dmodel_forward(new_input)
            output = output[self.DEFAULT_DMODEL_NAME]
        else:
            output = self._dmodel_forward(d_model_input)
        return output

    def _dmodel_backward(self, d_model_loss: Dict[str, torch.Tensor], retain_graph=False):
        """
        :param retain_graph:
        :param d_model_loss: dict(d_model_key=torch.Tensor)
        :return:
        """
        assert len(d_model_loss) == len(self.d_optimizer), 'the d_model_loss not match d_optimizer number'
        for key, loss_tensor in d_model_loss.items():
            assert isinstance(loss_tensor, torch.Tensor), 'the type of value of d_model_loss must be torch.Tensor'
            self.d_optimizer[key].zero_grad()
            loss_tensor.backward(retain_graph=retain_graph)
            self.d_optimizer[key].step()
        return

    def dmodel_backward(self, d_model_loss: Union[Dict[str, torch.Tensor], torch.Tensor], retain_graph=False):
        """
        :param retain_graph:
        :param d_model_loss: dict(d_model_key=torch.Tensor)
        :return:
        """
        if isinstance(d_model_loss, torch.Tensor):
            new_input = {self.DEFAULT_DMODEL_NAME: d_model_loss}
            self._dmodel_backward(new_input, retain_graph)
        else:
            self._dmodel_backward(d_model_loss, retain_graph)
        return

    def gmodel_backward(self, loss_tensor: torch.Tensor, retain_graph=False):
        self.g_optimizer.zero_grad()
        loss_tensor.backward(retain_graph=retain_graph)
        self.g_optimizer.step()
        return

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
        for k, d_schedler in self.d_scheduler.items():
            d_schedler.step(epoch)

        loss_dict['g_learning_rate'] = '*'.join([str(lr) for lr in self.g_scheduler.get_last_lr()])
        loss_dict['d_learning_rate'] = list()
        for k, d_schedler in self.d_scheduler.items():
            loss_dict['d_learning_rate'].append({k: '*'.join([str(lr) for lr in d_schedler.get_last_lr()])})

        return loss_dict

    def enable_train(self):
        self.g_model.train()
        for k, d_model in self.d_model.items():
            d_model.train()
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
                state_dict['d_optimizer_{}'.format(k)] = checkpoint_f.get_model_state_dict(self.d_optimizer[k])
                state_dict['d_scheduler_{}'.format(k)] = checkpoint_f.get_model_state_dict(self.d_scheduler[k])
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
                checkpoint_f.load_model_state_dict(d_model, state_dict['d_model_{}'.format(k)], log_name=self.default_log_name)
                checkpoint_f.load_checkpoint_state_dict(self.d_optimizer[k], state_dict['d_optimizer_{}'.format(k)])
                checkpoint_f.load_checkpoint_state_dict(self.d_scheduler[k], state_dict['d_scheduler_{}'.format(k)])
        else:
            checkpoint_f.load_model_state_dict(self.d_model, state_dict['d_model'], log_name=self.default_log_name)
            checkpoint_f.load_checkpoint_state_dict(self.d_optimizer, state_dict['d_optimizer'])
            checkpoint_f.load_checkpoint_state_dict(self.d_scheduler, state_dict['d_scheduler'])
        return

    def enable_dirstribute_ddp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.TRAINER.PARADIGM.GPU_ID))
        self.g_model = torch.nn.parallel.DistributedDataParallel(self.g_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
        if isinstance(self.d_model, dict):
            new_d_model = dict()
            for k, d_model in self.d_model.items():
                new_d_model[k] = torch.nn.parallel.DistributedDataParallel(d_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
            self.d_model = new_d_model
        else:
            self.d_model = torch.nn.parallel.DistributedDataParallel(self.d_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
        return

    def enable_distribute_dp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by parallel')
        self.g_model = torch.nn.parallel.DataParallel(self.g_model)
        if isinstance(self.d_model, dict):
            new_d_model = dict()
            for k, d_model in self.d_model.items():
                new_d_model[k] = torch.nn.parallel.DataParallel(d_model)
            self.d_model = new_d_model
        else:
            self.d_model = torch.nn.parallel.DataParallel(self.d_model)
        return
