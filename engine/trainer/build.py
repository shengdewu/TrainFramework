from fvcore.common.registry import Registry

BUILD_TRAINER_REGISTRY = Registry('TRAINER')
BUILD_TRAINER_REGISTRY.__doc__ = """
BUILD_TRAINER_REGISTRY
"""


def build_trainer(cfg):
    return BUILD_TRAINER_REGISTRY.get(cfg.MODEL.TRAINER.TRAINER)(cfg)