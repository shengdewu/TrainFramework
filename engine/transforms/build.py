from fvcore.common.registry import Registry

BUILD_TRANSFORMER_REGISTRY = Registry('TRANSFORMER')
BUILD_TRANSFORMER_REGISTRY.__doc__ = """
BUILD_TRANSFORMER_REGISTRY
"""


def build_transformer(name, **kwargs):
    return BUILD_TRANSFORMER_REGISTRY.get(name)(**kwargs)