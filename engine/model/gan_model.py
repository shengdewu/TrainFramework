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
                self.d_optimizer[d_name] = self.create_optimizer(CfgNode(init_dict=d_cfg['PARAMS']['OPTIMIZER']),
                                                                 CfgNode(init_dict=d_cfg['PARAMS']['CLIP_GRADIENTS']),
                                                                 self.d_model[d_name].parameters())
                self.d_scheduler[d_name] = self.create_scheduler(CfgNode(init_dict=d_cfg['PARAMS']['LR_SCHEDULER']), self.d_optimizer[d_name])

        else:
            assert isinstance(cfg.SOLVER.DISCRIMINATOR, dict)
            self.d_optimizer[self.DEFAULT_DMODEL_NAME] = self.create_optimizer(CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['OPTIMIZER']),
                                                                               CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['CLIP_GRADIENTS']),
                                                                               self.d_model[self.DEFAULT_DMODEL_NAME].parameters())
            self.d_scheduler[self.DEFAULT_DMODEL_NAME] = self.create_scheduler(CfgNode(init_dict=cfg.SOLVER.DISCRIMINATOR['LR_SCHEDULER']),
                                                                               self.d_optimizer[self.DEFAULT_DMODEL_NAME])

        assert len(set(self.d_model.keys()).intersection(self.d_optimizer.keys())) == len(self.d_model.keys()), f'the d_model not match the d_optimizer/d_scheduler name'
        return

    def create_d_model(self, params) -> torch.nn.Module:
        return build_network(params)

    def dmodel_forward(self, d_model_input: Union[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        :param d_model_input:
                1. d_model 的配置如下：
                model['discriminator'] = [
                        dict(name='d_model1', params=dict(name='Discriminator')),
                        dict(name='d_model2', params=dict(name='Discriminator'))
                        ]

                则：
                    self.d_model = dict(d_model1=Discriminator()),
                                        d_model2=Discriminator())

                所以
                    d_model_input = dict(d_model1=dict(input_name=Tensor), 内层 dict 表示模型 Discriminator 输入参数
                                         d_model2=dict(input_name=Tensor))

                2. d_model 的配置如下：
                model['discriminator'] = dict(name='Discriminator')

                则：
                    self.d_model = dict(default=Discriminator())) #这种情况下 使用缺省 key[default]

                所以
                    d_model_input = dict(input_name=Tensor)  # 表示模型 Discriminator 输入参数
                    或者
                    d_model_input = dict(default=dict(input_name=Tensor)) # 内层 dict 表示模型 Discriminator 输入参数
        :return:
                dict(d_model1=Tensor, d_model2=Tensor)
                或者
                dict(default=Tensor)
        """
        if isinstance(d_model_input[list(d_model_input.keys())[0]], torch.Tensor):
            d_model_input = {self.DEFAULT_DMODEL_NAME: d_model_input}

        assert len(set(d_model_input.keys()).intersection(self.d_model.keys())) == len(self.d_model.keys()), f'the name of input != the name of self.d_model'

        output = dict()
        for key, param in d_model_input.items():
            output[key] = self.d_model[key](**param)
        return output

    def _dmodel_backward(self, d_model_loss: Union[Dict[str, torch.Tensor], torch.Tensor], retain_graph=False):
        """
        :param retain_graph:
        :param d_model_loss:
                参见方法 _dmodel_forward
                d_model_loss=dict(d_model1=torch.Tensor),
                                  d_model2=torch.Tensor)
                或者
                d_model_input = torch.Tensor  #使用缺省 key[default] 只有一个d_model且用户未指定d_model的key
        :return:
        """
        if isinstance(d_model_loss, torch.Tensor):
            d_model_loss = {self.DEFAULT_DMODEL_NAME: d_model_loss}

        assert len(d_model_loss) == len(self.d_optimizer), 'the d_model_loss not match d_optimizer number'
        for key, loss_tensor in d_model_loss.items():
            assert isinstance(loss_tensor, torch.Tensor), 'the type of value of d_model_loss must be torch.Tensor'
            self.d_optimizer[key].zero_grad()
            loss_tensor.backward(retain_graph=retain_graph)
            self.d_optimizer[key].step()
        return

    def _gmodel_backward(self, loss_tensor: torch.Tensor, retain_graph=False):
        self.g_optimizer.zero_grad()
        loss_tensor.backward(retain_graph=retain_graph)
        self.g_optimizer.step()
        return

    @abc.abstractmethod
    def gmodel_step(self, data, epoch=None, **kwargs) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def dmodel_step(self, data, epoch=None, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def run_step(self, data, *, epoch=None, **kwargs):
        step_info = dict()

        d_model_loss = self.dmodel_step(data, epoch, **kwargs)
        self._dmodel_backward(d_model_loss, retain_graph=False)

        if epoch % self.cfg.TRAINER.MODEL.get('G_STEP', 1) == 0:
            g_model_loss = self.gmodel_step(data, epoch, **kwargs)
            self._gmodel_backward(g_model_loss, retain_graph=False)
            self.g_scheduler.step(epoch)
            step_info['g_model_loss'] = g_model_loss.detach().item()

        for k, d_schedler in self.d_scheduler.items():
            d_schedler.step(epoch)

        for name, item in d_model_loss.items():
            step_info[f'd_model_loss_{name}'] = item.detach().item()
        return step_info

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
        for k, d_model in self.d_model.items():
            d_model.train()
        return

    def get_addition_state_dict(self):
        state_dict = dict()
        state_dict['g_optimizer'] = checkpoint_f.get_model_state_dict(self.g_optimizer)
        state_dict['g_scheduler'] = checkpoint_f.get_model_state_dict(self.g_scheduler)

        for k, d_model in self.d_model.items():
            state_dict['d_model_{}'.format(k)] = checkpoint_f.get_model_state_dict(d_model)
            state_dict['d_optimizer_{}'.format(k)] = checkpoint_f.get_model_state_dict(self.d_optimizer[k])
            state_dict['d_scheduler_{}'.format(k)] = checkpoint_f.get_model_state_dict(self.d_scheduler[k])

        state_dict['cfg'] = self.cfg.dump()
        return state_dict

    def load_addition_state_dict(self, state_dict: dict):
        """
        :param state_dict: type is dict
        :return: 
        """""
        checkpoint_f.load_checkpoint_state_dict(self.g_optimizer, state_dict['g_optimizer'])
        checkpoint_f.load_checkpoint_state_dict(self.g_scheduler, state_dict['g_scheduler'])

        for k, d_model in self.d_model.items():
            checkpoint_f.load_model_state_dict(d_model, state_dict['d_model_{}'.format(k)], log_name=self.default_log_name)
            checkpoint_f.load_checkpoint_state_dict(self.d_optimizer[k], state_dict['d_optimizer_{}'.format(k)])
            checkpoint_f.load_checkpoint_state_dict(self.d_scheduler[k], state_dict['d_scheduler_{}'.format(k)])
        return

    def enable_dirstribute_ddp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by distribute in gpu_id {}'.format(cfg.TRAINER.PARADIGM.GPU_ID))
        self.g_model = torch.nn.parallel.DistributedDataParallel(self.g_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
        new_d_model = dict()
        for k, d_model in self.d_model.items():
            new_d_model[k] = torch.nn.parallel.DistributedDataParallel(d_model, device_ids=[cfg.TRAINER.PARADIGM.GPU_ID])
        self.d_model = new_d_model
        return

    def enable_distribute_dp(self, cfg):
        logging.getLogger(self.default_log_name).info('launch model by parallel')
        self.g_model = torch.nn.parallel.DataParallel(self.g_model)
        new_d_model = dict()
        for k, d_model in self.d_model.items():
            new_d_model[k] = torch.nn.parallel.DataParallel(d_model)
        self.d_model = new_d_model
        return
