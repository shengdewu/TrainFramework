import importlib


def import_optimizer(name):
    optimizer_module = 'torch.optim'
    module = importlib.import_module(optimizer_module)
    cls = getattr(module, name)
    return cls
