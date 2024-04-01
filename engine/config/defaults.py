from fvcore.common.config import CfgNode

_C = CfgNode(new_allowed=True)

_C.TRAINER = CfgNode(new_allowed=True)
_C.TRAINER.NAME = 'BaseTrainer'
_C.TRAINER.ENABLE_EPOCH_METHOD = False
_C.TRAINER.WEIGHTS = None
_C.TRAINER.DEVICE = 'cuda'
_C.TRAINER.PARADIGM = CfgNode(new_allowed=True)
_C.TRAINER.MODEL = CfgNode(new_allowed=True)
_C.TRAINER.MODEL.NAME = 'BaseModel'
_C.TRAINER.MODEL.GENERATOR = CfgNode(new_allowed=True)
_C.TRAINER.MODEL.DISCRIMINATOR = CfgNode(new_allowed=True)
"""
GENERATOR一样
或者List

    - NAME： ''  # 和模型名没有关系，主要起到一个标志作用
      PARAMS： CfgNode(new_allowed=True)
    - NAME： ''  # 和模型名没有关系，主要起到一个标志作用
      PARAMS： CfgNode(new_allowed=True)      

"""

_C.SOLVER = CfgNode(new_allowed=False)
_C.SOLVER.GRADIENT_ACCUMULATION_BATCH = -1  # 是否启用梯度累加 > 1 启用
_C.SOLVER.TRAIN_PER_BATCH = 16  # 当 gradient_accumulation_batch > 1 时，真实的 train_per_batch = train_per_batch // gradient_accumulation_batch
_C.SOLVER.TEST_PER_BATCH = 8
_C.SOLVER.MAX_ITER = 90000
_C.SOLVER.MAX_KEEP = 20
_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.EMA = CfgNode(init_dict={"ENABLED": False, "DECAY_RATE": 0.995}, new_allowed=True)

_C.SOLVER.GENERATOR = CfgNode(new_allowed=True)
_C.SOLVER.GENERATOR.LR_SCHEDULER = CfgNode(init_dict={"ENABLED": False}, new_allowed=True)
_C.SOLVER.GENERATOR.LR_SCHEDULER.TYPE = 'LRMultiplierScheduler'
_C.SOLVER.GENERATOR.LR_SCHEDULER.PARAMS = CfgNode(new_allowed=True)
_C.SOLVER.GENERATOR.OPTIMIZER = CfgNode(new_allowed=True)
_C.SOLVER.GENERATOR.OPTIMIZER.TYPE = 'Adam'
_C.SOLVER.GENERATOR.OPTIMIZER.PARAMS = CfgNode(new_allowed=True)
_C.SOLVER.GENERATOR.OPTIMIZER.PARAMS.LR = 0.0001
_C.SOLVER.GENERATOR.OPTIMIZER.PARAMS.BETAS = (0.9, 0.999)
_C.SOLVER.GENERATOR.CLIP_GRADIENTS = CfgNode({"ENABLED": False})
_C.SOLVER.GENERATOR.CLIP_GRADIENTS.CLIP_TYPE = "value"
_C.SOLVER.GENERATOR.CLIP_GRADIENTS.CLIP_VALUE = 1.0
_C.SOLVER.GENERATOR.CLIP_GRADIENTS.NORM_TYPE = 2.0
_C.SOLVER.GENERATOR.CLIP_GRADIENTS.GROUP = False

_C.SOLVER.DISCRIMINATOR = CfgNode(new_allowed=True)

"""
GENERATOR一样

或者List

    - NAME： ''  # 和模型名没有关系，主要起到一个标志作用
      PARAMS： CfgNode(new_allowed=True)
    - NAME： ''  # 和模型名没有关系，主要起到一个标志作用
      PARAMS： CfgNode(new_allowed=True)      

    eg:
    
    solver['discriminator'] = [
        dict(name='gen1', params=dict(lr_scheduler=lr_scheduler, optimizer=optimizer, clip_gradients=clip_gradients)),
        dict(name='gen2', params=dict(lr_scheduler=lr_scheduler, optimizer=optimizer, clip_gradients=clip_gradients))]
            
必须和 _C.TRAINER.MODEL.DISCRIMINATOR 保持一致
"""

# ---------------------------------------------------------------------------- #
# data loader config
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CfgNode(new_allowed=True)
_C.DATALOADER.NUM_WORKERS = 4

_C.OUTPUT_DIR = ''
_C.OUTPUT_LOG_NAME = __name__


def get_defaults():
    return _C.clone()
