from .build import build_loss


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

    def __call__(self, x, gt):
        score = self.losses[0](x, gt)
        for i in range(1, len(self.losses)):
            score += self.losses[i](x, gt)
        return score

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.losses:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


class LossKeyCompose:
    def __init__(self, loss_cfgs: dict, device=None):
        '''
        :param loss_cfgs: {NAME: [NAME: MSE, PARAM: 1.0], }
        '''
        self.losses = dict()

        for key, loss in loss_cfgs.items():
            key = key.lower()
            if self.losses.get(key) is not None:
                raise NameError('the {} be repeated definition'.format(key))

            if callable(loss):
                self.losses[key] = loss
            elif isinstance(loss, dict):
                kwargs = dict()
                arch_name = ''
                for k, v in loss.items():
                    if k.lower() == 'name':
                        arch_name = v
                        continue
                    kwargs[k.lower()] = v
                self.losses[key] = build_loss(arch_name, **kwargs)
                if device in ['cpu', 'cuda']:
                    self.losses[key] = self.losses[key].to(device)
            else:
                raise TypeError('loss must be callable or a dict')
        return

    def __loss_fn__(self, loss_fn, pair):
        if isinstance(pair, tuple) or isinstance(pair, list):
            score = loss_fn(*pair)
        else:
            score = loss_fn(pair)
        return score

    def __call__(self, key_pair: dict):
        assert len(key_pair) == len(self.losses)
        score = None
        for key, pair in key_pair.items():
            if key not in self.losses.keys():
                raise NotImplemented('the {} not implemented'.format(key))

            if score is None:
                score = self.__loss_fn__(self.losses[key], pair)
                continue

            score += self.__loss_fn__(self.losses[key], pair)

        return score

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for k, t in self.losses.items():
            format_string += '\n'
            format_string += f'{k}: {t}'
        format_string += '\n)'
        return format_string