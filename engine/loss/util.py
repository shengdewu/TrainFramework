from enum import Enum


class LossReduction(Enum):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"
