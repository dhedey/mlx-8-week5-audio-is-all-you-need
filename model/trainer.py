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
        num_mels = model.total_mels()
        total_time_frames = model.total_time_frames()

        train_dataset, eval_dataset = generate_urban_classifier_dataset(validation_fold, num_mels, total_time_frames)

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
        num_samples = len(collated_batch["labels"])
        logits = self.model(collated_batch)      # (Batch, Classes) => Logits
        label_indices = collated_batch["labels"] # (Batch) => Class index
        criterion = nn.CrossEntropyLoss()

        loss = criterion(logits, label_indices)

        return BatchResults(
            total_loss=loss,
            num_samples=num_samples,
            intermediates={
                "logits": logits,
                "label_indices": label_indices,
            }
        )

    def start_custom_validation(self) -> dict:
        return {
            "correct_predictions": 0,
        }

    def custom_validate_batch(self, custom_validation_metrics: dict, batch_results: BatchResults, batch_num: int, total_batches: int):
        best_guesses: torch.Tensor = batch_results.intermediates["logits"].argmax(axis = -1)         # (Batch) => Index
        correct_guesses: torch.Tensor = best_guesses == batch_results.intermediates["label_indices"] # (Batch) => True/False
        custom_validation_metrics["correct_predictions"] += correct_guesses.sum().item()

    def finalize_custom_validation(self, total_samples: int, total_batches: int, custom_validation_metrics: dict) -> dict:
        prediction_accuracy = custom_validation_metrics["correct_predictions"] / total_samples
        return {
            "prediction_accuracy": prediction_accuracy,
            "objective": prediction_accuracy
        }
