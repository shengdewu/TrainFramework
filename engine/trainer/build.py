from fvcore.common.registry import Registry  # for backward compatibility.
from .trainer import BaseTrainer

TRAINER_ARCH_REGISTRY = Registry("TRAINER")
TRAINER_ARCH_REGISTRY.__doc__ = """

Registry for trainer

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_trainer(cfg) -> BaseTrainer:
    trainer = TRAINER_ARCH_REGISTRY.get(cfg.MODEL.TRAINER.NAME)(cfg)
    return trainer
