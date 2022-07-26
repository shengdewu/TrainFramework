from fvcore.common.config import CfgNode

_C = CfgNode(new_allowed=True)

_C.MODEL = CfgNode(new_allowed=True)
_C.MODEL.WEIGHTS = ""
_C.MODEL.TRAINER = CfgNode(new_allowed=True)
_C.MODEL.DEVICE = 'cuda'


_C.SOLVER = CfgNode(new_allowed=False)
_C.SOLVER.TRAIN_PER_BATCH = 16
_C.SOLVER.TEST_PER_BATCH = 8
_C.SOLVER.MAX_ITER = 90000
_C.SOLVER.MAX_KEEP = 20
_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.LR_SCHEDULER = CfgNode({"ENABLED": False})
_C.SOLVER.LR_SCHEDULER.WARMUP_FACTOR = 0.01
_C.SOLVER.LR_SCHEDULER.WARMUP_ITERS = 1000
_C.SOLVER.LR_SCHEDULER.STEPS = (60000, 80000)
_C.SOLVER.LR_SCHEDULER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
_C.SOLVER.LR_SCHEDULER.GAMMA = 0.1
_C.SOLVER.LR_SCHEDULER.WARMUP_METHOD = "linear"
_C.SOLVER.OPTIMIZER = CfgNode(new_allowed=True)
_C.SOLVER.OPTIMIZER.MOMENTUM = 0.9
_C.SOLVER.OPTIMIZER.NESTEROV = False
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY = 0.0001
_C.SOLVER.OPTIMIZER.DAMPENING = 0.0
_C.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS = _C.SOLVER.OPTIMIZER.WEIGHT_DECAY
_C.SOLVER.OPTIMIZER.BASE_LR = 0.01
_C.SOLVER.CLIP_GRADIENTS = CfgNode({"ENABLED": False})
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
_C.SOLVER.CLIP_GRADIENTS.GROUP = False
_C.SOLVER.LOSS = CfgNode(new_allowed=True)


# ---------------------------------------------------------------------------- #
# data loader config
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CfgNode(new_allowed=True)
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode(new_allowed=True)

_C.OUTPUT_DIR = ''
_C.OUTPUT_LOG_NAME = __name__


def get_defaults():
    return _C.clone()