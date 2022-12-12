from fvcore.common.registry import Registry  # for backward compatibility.


BUILD_DATASET_REGISTRY = Registry("DATA")
BUILD_DATASET_REGISTRY.__doc__ = """

Registry for data

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_dataset(cfg):
    kwargs = dict()
    arch_name = ''
    for k, v in cfg.items():
        if k.lower() == 'name':
            arch_name = v
            continue
        kwargs[k.lower()] = v
    return BUILD_DATASET_REGISTRY.get(arch_name)(**kwargs)
