# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from bisect import bisect_right
from typing import List
import torch
from fvcore.common.param_scheduler import CosineParamScheduler, MultiStepParamScheduler

from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    LinearParamScheduler,
    ParamScheduler,
)

logger = logging.getLogger(__name__)


class WarmupParamScheduler(CompositeParamScheduler):
    """
    Add an initial warmup stage to another scheduler.
    """

    def __init__(
            self,
            scheduler: ParamScheduler,
            warmup_factor: float,
            warmup_length: float,
            warmup_method: str = "linear",
    ):
        """
        Args:
            scheduler: warmup will be added at the beginning of this scheduler
            warmup_factor: the factor w.r.t the initial value of ``scheduler``, e.g. 0.001
            warmup_length: the relative length (in [0, 1]) of warmup steps w.r.t the entire
                training, e.g. 0.01
            warmup_method: one of "linear" or "constant"
        """
        end_value = scheduler(warmup_length)  # the value to reach when warmup ends
        start_value = warmup_factor * scheduler(0.0)
        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_value)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))
        super().__init__(
            [warmup, scheduler],
            interval_scaling=["rescaled", "fixed"],
            lengths=[warmup_length, 1 - warmup_length],
        )


class LRMultiplierScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    A LRScheduler which uses fvcore :class:`ParamScheduler` to multiply the
    learning rate of each param in the optimizer.
    Every step, the learning rate of each parameter becomes its initial value
    multiplied by the output of the given :class:`ParamScheduler`.

    The absolute learning rate value of each parameter can be different.
    This scheduler can be used as long as the relative scale among them do
    not change during training.

    Examples:
    ::
        LRMultiplier(
            opt,
            WarmupParamScheduler(
                MultiStepParamScheduler(
                    [1, 0.1, 0.01],
                    milestones=[60000, 80000],
                    num_updates=90000,
                ), 0.001, 100 / 90000
            ),
            max_iter=90000
        )
    """

    # NOTES: in the most general case, every LR can use its own scheduler.
    # Supporting this requires interaction with the optimizer when its parameter
    # group is initialized. For example, classyvision implements its own optimizer
    # that allows different schedulers for every parameter group.
    # To avoid this complexity, we use this class to support the most common cases
    # where the relative scale among all LRs stay unchanged during training.  In this
    # case we only need a total of one scheduler that defines the relative LR multiplier.

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_factor,
            warmup_iter,
            lr_scheduler_param: dict,
            max_iter: int,
            last_iter: int = -1,
            warmup_method='linear'
    ):
        """
        Args:
            optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                ``last_iter`` is the same as ``last_epoch``.
            max_iter: the total number of training iterations
        """
        self.max_iter = max_iter

        name = lr_scheduler_param['name']

        if name == "WarmupMultiStepLR":
            steps = [x for x in lr_scheduler_param['steps'] if x <= max_iter]
            if len(steps) != len(lr_scheduler_param['steps']):
                logging.getLogger(__name__).warning(
                    "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                    "These values will be ignored."
                )
            scheduler = MultiStepParamScheduler(
                values=[lr_scheduler_param.get('gamma', 0.1) ** k for k in range(len(steps) + 1)],
                milestones=steps,
                num_updates=max_iter,
            )
        elif name == "WarmupCosineLR":
            start_value = lr_scheduler_param.get('start_value', 1.0)
            end_value = lr_scheduler_param.get('end_value', 0.)
            scheduler = CosineParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown LR scheduler: {}".format(name))

        self._multiplier = WarmupParamScheduler(
            scheduler,
            warmup_factor,
            min(warmup_iter / max_iter, 1.0),
            warmup_method,
        )

        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        # fvcore schedulers are stateless. Only keep pytorch scheduler states
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self.max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]


"""
Content below is no longer needed!
"""


# NOTE: PyTorch's LR scheduler interface uses names that assume the LR changes
# only on epoch boundaries. We typically use iteration based schedules instead.
# As a result, "epoch" (e.g., as in self.last_epoch) should be understood to mean
# "iteration" instead.

# FIXME: ideally this would be achieved with a CombinedLRScheduler, separating
# MultiStepLR with WarmupLR but the current LRScheduler design doesn't allow it.


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_iter: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        logger.warning(
            "WarmupMultiStepLR is deprecated! Use LRMultipilier with fvcore ParamScheduler instead!"
        )
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iter = warmup_iter
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iter, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iter: int,
            warmup_factor: float = 0.001,
            warmup_iter: int = 1000,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        logger.warning(
            "WarmupCosineLR is deprecated! Use LRMultipilier with fvcore ParamScheduler instead!"
        )
        self.max_iter = max_iter
        self.warmup_factor = warmup_factor
        self.warmup_iter = warmup_iter
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iter, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iter
        # instead of at 0. In the case that warmup_iter << max_iter the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iter))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class EmptyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        super(EmptyLRScheduler, self).__init__(optimizer)
        return

    def step(self, epoch=None):
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Poly learning rate schedule used to train DeepLab.
    Paper: DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
        Atrous Convolution, and Fully Connected CRFs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iter: int,
        warmup_factor: float = 0.001,
        warmup_iter: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
        power: float = 0.9,
        constant_ending: float = 0.0,
    ):
        self.max_iter = max_iter
        self.warmup_factor = warmup_factor
        self.warmup_iter = warmup_iter
        self.warmup_method = warmup_method
        self.power = power
        self.constant_ending = constant_ending
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iter, self.warmup_factor
        )
        if self.constant_ending > 0 and warmup_factor == 1.0:
            # Constant ending lr.
            if (
                math.pow((1.0 - self.last_epoch / self.max_iter), self.power)
                < self.constant_ending
            ):
                return [base_lr * self.constant_ending for base_lr in self.base_lrs]
        return [
            base_lr * warmup_factor * math.pow((1.0 - self.last_epoch / self.max_iter), self.power)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_iter(
        method: str, iter_num: int, warmup_iter: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter_num (int): iteration at which to calculate the warmup factor.
        warmup_iter (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter_num >= warmup_iter:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter_num / warmup_iter
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

