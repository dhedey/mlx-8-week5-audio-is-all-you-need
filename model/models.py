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
            # [Batch, Channels=1, Freq=n_mels, Time=T]
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

class TransformerUrbanSoundClassifierModelConfig(ModuleConfig):
    def __init__(self,
                 embed_dim=128,
                 patch_mels=16,
                 patch_time=16,
                 num_heads=4,
                 num_layers=2,
                 mlp_dim=256,
                 dropout=0.1):
        self.embed_dim = embed_dim
        self.patch_mels = patch_mels
        self.patch_time = patch_time
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout = dropout

class TransformerUrbanSoundClassifierModel(UrbanSoundClassifierModel):
    def __init__(self, model_name: str, config: TransformerUrbanSoundClassifierModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        # Input: batch of spectrograms.shape = [batch, 1, n_mels, time]
        # Input for transformer: [batch, seq_len, feature_dim]
        num_classes = 10

        self.n_mels = 64
        self.time_frames = 128
        self.patch_mels = config.patch_mels
        self.patch_time = config.patch_time
        self.embed_dim = config.embed_dim

        self.num_patches_mel = self.n_mels // self.patch_mels
        self.num_patches_time = self.time_frames // self.patch_time
        self.num_patches = self.num_patches_mel * self.num_patches_time

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_mels, self.patch_time),
            stride=(self.patch_mels, self.patch_time)
        )  # [B, embed_dim, num_patches_mel, num_patches_time]
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, self.embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.classifier = nn.Linear(self.embed_dim, num_classes)


    def forward(self, collated_batch) -> torch.Tensor:
        # Takes MelSpectrograms and returns logits for each class
        # (batch, channels=1, freq=num_mels, time)

        # Patchify input collated_batch [B,1,64,128] -> [B, embed_dim, 4, 8]
        patches = self.patch_embedding(collated_batch["spectrograms"])

        # Flatten and permute to [batch, seq_len=32, embed_dim]
        patch_embeddings = patches.flatten(2).transpose(1,2)

        B, N, D = patch_embeddings.shape

        # Create the [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1) # [B, 1, D]

        # Prepend the [CLS] token to the sequence
        x = torch.cat([cls_token, patch_embeddings], dim=1)  # [B, 1 + N, D]

        # Add positional encoding
        x = x + self.pos_embedding[:, :N+1]

        # Pass through encoder
        hidden_state = self.encoder(x)

        # Grab only the [CLS] token, as it contains aggregated global information from all patches 
        cls_out = hidden_state[:,0]

        return self.classifier(cls_out)
    
if __name__ == "__main__":
   print("Run default_models instead of this file")