from fvcore.common.registry import Registry

BUILD_MODEL_REGISTRY = Registry('MODEL')
BUILD_MODEL_REGISTRY.__doc__ = """
BUILD_MODEL_REGISTRY
"""


def build_model(cfg):
    return BUILD_MODEL_REGISTRY.get(cfg.TRAINER.MODEL.NAME)(cfg)


BUILD_NETWORK_REGISTRY = Registry('NETWORK')
BUILD_NETWORK_REGISTRY.__doc__ = """
BUILD_NETWORK_REGISTRY
"""


def build_network(cfg: dict):
    kwargs = dict()
    arch_name = ''
    for k, v in cfg.items():
        if k.lower() == 'name':
            arch_name = v
            continue
        if isinstance(v, dict):
            kwargs[k.lower()] = dict()
            for kk, vv in v.items():
                kwargs[k.lower()][kk.lower()] = vv
        else:
            kwargs[k.lower()] = v
    model = BUILD_NETWORK_REGISTRY.get(arch_name)(**kwargs)
    return model
