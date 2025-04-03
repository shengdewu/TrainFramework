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


class LossContainer:
    def __init__(self):
        self.funcs = list()
        return

    def add(self, func: Callable, params: Union[List, Tuple, str]):
        if isinstance(params, str):
            if params == '':
                params = []
            else:
                params = [params]
        else:
            params = params
        self.funcs.append((func, params))
        return

    def __call__(self, loss_input: Union[Dict, Tuple, List]):
        if isinstance(loss_input, Dict):
            score = None
            for func, param in self.funcs:
                assert list(loss_input.keys()) == param, f'the loss input key {loss_input.keys()} must be == the loss_functions key {param}'
                if score is None:
                    score = func(**loss_input)
                else:
                    score += func(**loss_input)
            return score
        elif isinstance(loss_input, Tuple) or isinstance(loss_input, List):
            score = None
            for func, param in self.funcs:
                assert 0 == len(param), f'the loss input key must be empty, but {param}'
                if score is None:
                    score = func(*loss_input)
                else:
                    score += func(*loss_input)
            return score
        raise NotImplementedError(f'the loss input {type(loss_input)} is not implemented, the loss input type must be Dict, Tuple, List')

    def __repr__(self):
        format_string = ''
        for i, (func, param) in enumerate(self.funcs):
            if 0 == len(format_string):
                format_string = f'{i}: {str(func)}'
            else:
                format_string = f'{format_string}, {i}: {str(func)}'
        return format_string


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

    def create_loss_func(self, loss: dict, device=None):
        arch_name = self.get_values(loss, self.LOSS_NAME)
        kwargs = dict2lower(self.get_values(loss, self.LOSS_PARAM))
        input_names = self.get_values(loss, self.LOSS_INPUT_NAME, False)
        loss_func = build_loss(arch_name, **kwargs)
        if device in ['cpu', 'cuda']:
            loss_func = loss_func.to(device)
        return loss_func, input_names

    def __init__(self, loss_cfgs: dict, device=None):
        """
        :param loss_cfgs:
        loss_cfgs = dict(
            loss1=[
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
            ],
            loss2=[
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
            ],
            loss3=[
                (
                    dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2']),
                    dict(name='VGGLoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
                ),
                dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
            ]
        )
        """

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
                    loss_func, input_names = self.create_loss_func(loss, device)
                    loss_container = LossContainer()
                    loss_container.add(loss_func, input_names)
                    self.losses[key].append(loss_container)
                elif (isinstance(loss, list) or isinstance(loss, tuple)) and isinstance(loss[0], dict):
                    loss_container = LossContainer()
                    for l in loss:
                        loss_func, input_names = self.create_loss_func(l, device)
                        loss_container.add(loss_func, input_names)
                    self.losses[key].append(loss_container)
                else:
                    raise TypeError('loss must be callable or a dict or dict[list]')
        return

    def __loss_fn__(self, loss_inputs: Union[List[List], List[Tuple]], loss_functions: List[LossContainer]):
        """
        :param loss_inputs: [{input1: tensor, input2:tensor}, {...}]} or [[tensor, tensor], [..]] or [(tensor, tensor)]
        :param loss_functions:
        :return:
        """
        assert (loss_inputs[0], List) or isinstance(loss_inputs[0], Tuple), 'loss_input must be List[Tuple] or List[List]'

        if len(loss_inputs) == len(loss_functions):  # 表示loss_inputs中每个成员和loss_functions中的成员一一对应
            score = loss_functions[0](loss_inputs[0])
            for i in range(1, len(loss_functions)):
                score += loss_functions[i](loss_inputs[i])
        else:
            # 多个输入共享一个loss
            if 1 == len(loss_functions):
                score = loss_functions[0](loss_inputs[0])
                for i in range(1, len(loss_inputs)):
                    score += loss_functions[0](loss_inputs[i])
            # 一个输入共享多个loss
            else:
                assert 1 == len(loss_inputs), 'loss_input must be equal 1 when len(loss_functions) > 1 and len(loss_inputs) != len(loss_functions)'
                score = loss_functions[0](loss_inputs[0])
                for i in range(1, len(loss_functions)):
                    score += loss_functions[i](loss_inputs[0])

        return score

    def __call__(self, losses: dict):
        """
        :param losses:
                    {key: [{input1: tensor, input2:tensor}, {...}]}
                    {key: [[tensor, tensor], [..], (...,)]}
        :return:
        """
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


class LossItem:
    def __init__(self, func: Callable, params: List):
        self.func = func
        self.params = params
        return

    def __call__(self, loss_input: Dict):
        assert loss_input.keys() == self.params, f'the loss input key {loss_input.keys()} must be == the loss_functions key {self.params}'
        return self.func(**loss_input)

    def __repr__(self):
        return '{str(self.func)}'


class LossKeyCompose2:
    LOSS_NAME = 'name'
    LOSS_PARAM = 'param'
    LOSS_INPUT_NAME = 'input_name'

    @staticmethod
    def get_values(item: dict, key: str, must_be: bool = True):
        if key.lower() in item.keys():
            return item[key.lower()]
        if key.upper() in item.keys():
            return item[key.upper()]

        if must_be:
            raise KeyError(f'{key.lower()} or {key.upper()} not in {item.keys()}')
        return ''

    def create_loss_func(self, loss: dict, device=None) -> (Callable, List):
        arch_name = self.get_values(loss, self.LOSS_NAME)
        kwargs = dict2lower(self.get_values(loss, self.LOSS_PARAM))
        input_names = self.get_values(loss, self.LOSS_INPUT_NAME, False)
        loss_func = build_loss(arch_name, **kwargs)
        if device in ['cpu', 'cuda']:
            loss_func = loss_func.to(device)
        return loss_func, input_names

    def __init__(self, loss_cfgs: List[Dict], device=None):
        """
        :param loss_cfgs:
        loss_cfgs = [
            dict(name='MSELoss', param=dict(param1=1.0), input_name=['input1', 'input2']),
            dict(name='VGGLoss', param=dict(param1=1.0), input_name=['input1', 'input2'])
        ]
        """

        self.losses: List[LossItem] = list()

        for loss in loss_cfgs:
            assert isinstance(loss, dict)
            loss_func, input_names = self.create_loss_func(loss, device)
            loss_item = LossItem(loss_func, input_names)
            self.losses.append(loss_item)
        return

    def __call__(self, losses: dict):
        """
        :param losses:
                {input1: tensor, input2: tensor, input3:tensor}
        :return:
        """
        assert len(losses) == len(self.losses)
        score = None
        for loss in self.losses:
            in_params_name = loss.params
            in_kwargs = dict()
            for name in in_params_name:
                if name not in losses.keys():
                    raise NotImplemented(f'{name} not in losses {losses.keys()}')

                in_kwargs[name] = losses[name]

            if score is None:
                score = loss(in_kwargs)
            else:
                score += loss(in_kwargs)

        return score

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for loss in self.losses:
            format_string += f'\n\t{loss}'
        format_string += '\n)'
        return format_string
