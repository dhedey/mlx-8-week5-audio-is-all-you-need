from dataclasses import dataclass, field
import datasets
import einops
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import statistics
import random
import wandb
import math
import os
from typing import Optional
import time

from .common import TrainingState, TrainingOverrides, ModelTrainerBase, ModelBase, TrainingConfig, BatchResults, ValidationResults
from .models import CaptionSection, CaptionSectionResult, UrbanSoundClassifierModel, UrbanSoundClassifierModel
from .prepared_datasets import generate_urban_classifier_dataset, noop_collate

class UrbanSoundClassifierModelTrainingConfig(TrainingConfig):
    pass

class UrbanSoundClassifierModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: UrbanSoundClassifierModel,
            config: UrbanSoundClassifierModelTrainingConfig,
            overrides: Optional[TrainingOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

        validation_fold = 1
        print(f"Preparing datasets with validation_fold = {validation_fold}...")
        train_dataset, eval_dataset = generate_urban_classifier_dataset(validation_fold)

        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(eval_dataset)}")

        self.train_loader = self.create_dataloader(train_dataset, self.config.batch_size, 2, model.collate)
        self.test_loader = self.create_dataloader(eval_dataset, self.config.batch_size, 2, model.collate)

        print()

    def create_dataloader(self, dataset, batch_size, num_workers, collate_fn):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.model.get_device() == 'cuda', 
            collate_fn=collate_fn
        )

    @property
    def train_data_loader(self):
        return self.train_loader

    @property
    def validation_data_loader(self):
        return self.test_loader

    def process_batch(self, collated_batch) -> BatchResults:
        device = self.model.get_device()

        self.model(collated_batch)

        return BatchResults(
            total_loss=loss,
            num_samples=total_non_padding_predictions,
            intermediates={
                # ...
            }
        )
    
    def custom_validation(self) -> Optional[dict]:
        return None
