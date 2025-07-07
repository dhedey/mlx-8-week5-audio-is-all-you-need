from dataclasses import dataclass
import PIL.Image
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import os
import statistics
import transformers
import random
import einops
import pandas as pd
import math
import PIL
from typing import Optional, Self, Any, Union, Generator
from peft import LoraConfig, TaskType, get_peft_model

from sympy.stats.rv import probability

from .modules.bert_custom_multi_modal_model import BertMultiModalModelConfig, BertMultiModalModel
from .modules.qwen_multi_modal_model import QwenMultiModalModelConfig, QwenMultiModalModel
from .modules.multi_modal_model import Section, CaptionSectionResult, CaptionSection, MultiModalModelResult, \
    SpecialTokenIds
from .common import ModelBase, ModuleConfig, TrainingConfig, Field

class UrbanSoundClassifierModel(ModelBase):
    def __init__(self, model_name: str, config: ModuleConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

    def collate(self, dataset_batch) -> dict:
        spectrograms = torch.tensor([
            item["spectrogram"] for item in dataset_batch
        ], dtype=torch.float)
        labels = torch.tensor([
            item["class_id"] for item in dataset_batch
        ], dtype=torch.long)

        return {
            "spectrograms": spectrograms,
            "labels": labels,
        }

    def forward(self, collated_batch) -> torch.Tensor:
        # Takes MelSpectrograms and returns logits for each class
        raise NotImplementedError("Implement in subclass")

class ConvolutionalUrbanSoundClassifierModelConfig(ModuleConfig):
    pass

class ConvolutionalUrbanSoundClassifierModel(UrbanSoundClassifierModel):
    def __init__(self, model_name: str, config: ConvolutionalUrbanSoundClassifierModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        num_classes = 10

        self.convolutional_layers = nn.Sequential(*[
            # 2d convolution over time and frequency
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, n_mels, T]
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),  # [B, 32, n_mels//2, T//2]

            # 2d convolution over time and frequency
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),  # [B, 64, n_mels//4, T//4]
        ])
        self.classifier = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        ])

    def forward(self, collated_batch) -> torch.Tensor:
        # Takes MelSpectrograms and returns logits for each class
        # (batch, channels=1, freq=num_mels, time)
        hidden_state = self.convolutional_layers(collated_batch["spectrograms"])
        return self.classifier(hidden_state)

if __name__ == "__main__":
   print("Run default_models instead of this file")