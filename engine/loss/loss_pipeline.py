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
