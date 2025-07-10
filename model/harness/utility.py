import os
from typing import Self

import pydantic
import torch
import wandb

def datasets_cache_folder():
    os.path.join(os.path.dirname(__file__), "..", "datasets")

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
_selected_device_no_mps = None

def _select_device_string():
    if (torch.cuda.is_available()):
        return "cuda"
    elif (torch.backends.mps.is_available()):
        # MPS has too many issues and is often slower.
        # Let's just use CPU locally
        return "cpu"
    else:
        return "cpu"
    
def _ensure_device_selected():
    global _selected_device, _selected_device_no_mps
    if _selected_device is None:
        device = torch.device(_select_device_string())
        _selected_device = device
        if device.type == 'mps':
            _selected_device_no_mps = torch.device('cpu')
            print(f'Selected device: {_selected_device} (fallback {_selected_device_no_mps})')
        else:
            _selected_device_no_mps = device
            print(f'Selected device: {_selected_device}')

def select_device():
    _ensure_device_selected()
    return _selected_device

def select_device_no_mps():
    _ensure_device_selected()
    return _selected_device_no_mps

class ModuleConfig(PersistableData):
    pass


def upload_model_artifact(
    model_name: str,
    file_path: str,
    artifact_name: str,
    metadata: dict = None,
    description: str = None
):
    """
    Upload a model as a wandb artifact.

    Args:
        model_name: Name of the model. Used for the file name inside the artifact.
        model_path: Path to the saved model file
        artifact_name: Optional custom artifact name (defaults to model_name)
        metadata: Optional metadata dictionary to include with the artifact
        description: Optional description for the artifact
    """
    if not os.path.exists(file_path):
        print(f"⚠️ Model file not found: {file_path}")
        return None

    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description or f"Trained model: {model_name}",
        metadata=metadata or {}
    )

    # Add the model file
    artifact.add_file(file_path, name=f"{model_name}.pt")

    # Log the artifact
    wandb.log_artifact(artifact)
    print(f"📦 Uploaded model artifact: {artifact_name}")

    return artifact
