from typing import Dict, List, Tuple, Callable, Union
from torch import Tensor
from .build import build_loss


def dict2lower(item: dict):
    new_item = dict()
    for k, v in item.items():
        new_item[k.lower()] = v
    return new_item


class LossCompose:
    def __init__(self, loss_cfgs: list):
        self.losses = list()

        for loss in loss_cfgs:
            if callable(loss):
                self.losses.append(loss)
            elif isinstance(loss, dict):
                kwargs = dict()
                arch_name = ''
                for k, v in loss.items():
                    if k.lower() == 'name':
                        arch_name = v
                        continue
                    kwargs[k.lower()] = v
                self.losses.append(build_loss(arch_name, **kwargs))
            else:
                raise TypeError('loss must be callable or a dict')
        return

    def __call__(self, loss_input: Union[List[Tensor], Tuple[Tensor]]):
        score = self.losses[0](*loss_input)
        for i in range(1, len(self.losses)):
            score += self.losses[i](*loss_input)
        return score

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.losses:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class LossFunc:
    def __init__(self, func: Callable, param):
        self.func = func
        self.param = param
        return

    def __call__(self, loss_input: Union[Dict, Tuple, List]):
        if isinstance(loss_input, Dict):
            assert list(loss_input.keys()) == self.param, f'the loss input key {loss_input.keys()} must be == the loss_functions key {self.param}'
            return self.func(**loss_input)
        elif isinstance(loss_input, Tuple) or isinstance(loss_input, List):
            assert self.param == '', f'the loss input key must be empty {self.param}'
            return self.func(*loss_input)
        raise NotImplemented('the loss input is not implemented')

    def __repr__(self):
        return str(self.func)


class LossKeyCompose:
    LOSS_NAME = 'name'
    LOSS_PARAM = 'param'
    LOSS_INPUT_NAME = 'input_name'

    def get_values(self, item: dict, key: str, must_be: bool = True):
        if key.lower() in item.keys():
            return item[key.lower()]
        if key.upper() in item.keys():
            return item[key.upper()]

        if must_be:
            raise KeyError(f'{key.lower()} or {key.upper()} not in {item.keys()}')
        return ''

    def __init__(self, loss_cfgs: dict, device=None):
        '''
        :param loss_cfgs:
        loss_cfgs = dict(
            loss1=[
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
            ],
            loss2=[
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
            ],
        )
        '''

        self.losses = dict()

        for key, losses in loss_cfgs.items():
            key = key.lower()
            if self.losses.get(key) is not None:
                raise NameError('the {} be repeated definition'.format(key))

            self.losses[key] = list()
            for loss in losses:
                if callable(loss):
                    self.losses[key] = loss
                elif isinstance(loss, dict):
                    arch_name = self.get_values(loss, self.LOSS_NAME)
                    kwargs = dict2lower(self.get_values(loss, self.LOSS_PARAM))
                    input_name = self.get_values(loss, self.LOSS_INPUT_NAME, False)
                    loss_func = build_loss(arch_name, **kwargs)
                    if device in ['cpu', 'cuda']:
                        loss_func = loss_func.to(device)
                    self.losses[key].append(LossFunc(func=loss_func, param=input_name))
                else:
                    raise TypeError('loss must be callable or a dict')
        return

    def __loss_fn__(self, loss_inputs: Union[List, Tuple], loss_functions: List[LossFunc]):
        """
        :param loss_inputs: [{input1: tensor, input2:tensor}, {...}]} or [[tensor, tensor], [..]] or (tensor, tensor)
        :param loss_functions:
        :return:
        """
        if len(loss_inputs) == len(loss_functions):
            score = loss_functions[0](loss_inputs[0])
            for i in range(1, len(loss_functions)):
                score += loss_functions[i](**(loss_inputs[i]))
        else:
            score = loss_functions[0](loss_inputs)
            for i in range(1, len(loss_functions)):
                score += loss_functions[i](loss_inputs)
        return score

    def __call__(self, losses: dict):
        '''
        :param losses:
                    {key: [{input1: tensor, input2:tensor}, {...}]}
                    {key: [[tensor, tensor], [..]]}
                    {key: (tensor, tensor)}
        :return:
        '''
        assert len(losses) == len(self.losses)
        score = None
        for key, loss_input in losses.items():
            if key not in self.losses.keys():
                raise NotImplemented('the {} not implemented'.format(key))

            if score is None:
                score = self.__loss_fn__(loss_input, self.losses[key])
                continue

            score += self.__loss_fn__(loss_input, self.losses[key])

        return score

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for k, losses in self.losses.items():
            format_string += f'\n\t{k}: \n'
            for loss in losses:
                format_string += f'\t\t{loss}'
            format_string += '\n'
        format_string += '\n)'
        return format_string
