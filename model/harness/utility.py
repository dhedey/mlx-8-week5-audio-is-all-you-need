import os
from typing import Self

import pydantic
import torch
import wandb

class PersistableData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        use_enum_values=True,
        # See note here: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.use_enum_values
        validate_default=True,
    )

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        return cls(**d)

_selected_device = None

def select_device():
    global _selected_device
    if _selected_device is None:
        DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
        device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')

        print(f'Selected device: {device}')
        _selected_device = device

    return _selected_device


class ModuleConfig(PersistableData):
    pass
