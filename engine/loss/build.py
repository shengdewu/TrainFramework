from fvcore.common.registry import Registry  # for backward compatibility.


LOSS_ARCH_REGISTRY = Registry("LOSS")
LOSS_ARCH_REGISTRY.__doc__ = """

Registry for model

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_loss(name, **kwargs):
    return LOSS_ARCH_REGISTRY.get(name)(**kwargs)

