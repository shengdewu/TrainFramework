from abc import ABC
import torch.utils.data
from engine.transforms.pipe import TransformCompose
from typing import List, Dict


class EngineDataSet(torch.utils.data.Dataset, ABC):
    def __init__(self, transformers: List):
        super(EngineDataSet, self).__init__()
        self.transformers = TransformCompose(transformers)

    def transformate(self, params: Dict):
        return self.transformers(params)

    def __repr__(self):
        format_string = f'transformer={self.transformers})'
        return format_string


