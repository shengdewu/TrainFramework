import importlib


def import_scheduler(name):
    scheduler_module = 'torch.optim.lr_scheduler'
    module = importlib.import_module(scheduler_module)
    cls = getattr(module, name)
    return cls
