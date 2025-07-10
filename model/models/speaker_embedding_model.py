import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from model.harness import ModelBase, ModuleConfig, BatchResults, ModelTrainerBase, TrainingConfig, TrainingState, TrainingOverrides, select_device_no_mps
from .prepared_datasets import generate_speaker_tagged_dataset
from typing import Optional, Any
from .prepared_datasets import soundfile_to_whisper_embedding

class SpeakerEmbeddingTwoTowersConfig(ModuleConfig):
    total_speakers: int # 109 in badayvedat/VCTK
    target_embedding_dimension: int
    whisper_embedding_dimension: int
    inner_model_name: str = "speaker-embedding"

class SpeakerEmbeddingTwoTowers(ModelBase):
    def __init__(self, model_name: str, config: SpeakerEmbeddingTwoTowersConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config

        self.speaker_embedding_model = SpeakerEmbeddingModel(
            model_name=self.config.inner_model_name,
            config=SpeakerEmbeddingModelConfig(
                whisper_embedding_dimension=self.config.whisper_embedding_dimension,
                target_embedding_dimension=self.config.target_embedding_dimension,
            )
        )
        self.known_speaker_embedding = nn.Embedding(config.total_speakers, config.target_embedding_dimension)

    def collate(self, dataset_batch) -> dict:
        whisper_embeddings = torch.cat([
            torch.tensor(item["whisper_embedding"], dtype=torch.float) for item in dataset_batch
        ])
        speaker_indices = torch.tensor([
            item["speaker_index"] for item in dataset_batch
        ], dtype=torch.long)

        return {
            "whisper_embeddings": whisper_embeddings,
            "speaker_indices": speaker_indices,
        }

    def mean_embedding_over_time_interval(self, start_ms, end_ms) -> torch.Tensor:
        raise NotImplementedError("")

    def forward(self, collated_batch) -> tuple[torch.Tensor, torch.Tensor]:
        model_embeddings = self.speaker_embedding_model(collated_batch["whisper_embeddings"])
        known_speaker_embeddings = self.known_speaker_embedding(collated_batch["speaker_indices"])

        return model_embeddings, known_speaker_embeddings

    def average_speaker_embedding(self) -> torch.Tensor:
        return self.known_speaker_embedding.weight.mean(dim=0)


class SpeakerEmbeddingModelConfig(ModuleConfig):
    whisper_embedding_dimension: int
    target_embedding_dimension: int

class SpeakerEmbeddingModel(ModelBase):
    def __init__(self, model_name, config: SpeakerEmbeddingModelConfig):
        super().__init__(model_name=model_name, config=config)

        self.config = config
        d_in  = config.whisper_embedding_dimension
        d_mid = config.target_embedding_dimension * 2
        d_out = config.target_embedding_dimension

        # first block: linear → ReLU
        self.fc1   = nn.Linear(d_in, d_mid)
        self.act1  = nn.ReLU(inplace=True)

        # batch‐norm over the d_mid channels
        self.bn    = nn.BatchNorm1d(d_mid)

        # second block: dropout → linear
        self.drop2 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(d_mid, d_out)

    def process(self, soundfiles: list[tuple[np.ndarray, int]]) -> torch.Tensor:
        return torch.cat(
            [soundfile_to_whisper_embedding(soundfile) for soundfile in soundfiles],
            dim = 0 # Batch dimension
        )

    def forward(self, whisper_embeddings: torch.Tensor) -> torch.Tensor:
        # [B, T, d_in] → [B, T, d_mid]
        x = self.act1(self.fc1(whisper_embeddings))

        # permute so BN sees channels in dim=1
        # [B, T, d_mid] → [B, d_mid, T]
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        # back to [B, T, d_mid]
        x = x.permute(0, 2, 1)

        # finish MLP
        x = self.fc2(self.drop2(x))

        # normalize per time-step
        x = F.normalize(x, p=2, dim=-1)
        return x
    

class SpeakerEmbeddingModelTrainer(ModelTrainerBase):
    def __init__(
            self,
            model: SpeakerEmbeddingTwoTowers,
            config: TrainingConfig,
            overrides: Optional[TrainingOverrides] = None,
            continuation: Optional[TrainingState] = None,
        ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)

        # These are already stored in the base class. But setting them again helps the IDE understand their type.
        self.model = model
        self.config = config

        print(f"Preparing datasets")

        train_dataset, eval_dataset, dataset_total_speakers = generate_speaker_tagged_dataset()

        print(f"The dataset has {dataset_total_speakers} total speakers")

        assert dataset_total_speakers <= self.model.config.total_speakers, f"The model assumes {self.model.config.total_speakers} total speakers, but the dataset has more ({dataset_total_speakers})"

        print(f"Training set size: {len(train_dataset)}")
        print(f"Test set size: {len(eval_dataset)}")

        self.train_loader = self.create_dataloader(train_dataset, self.config.batch_size, 2, model.collate)
        self.test_loader = self.create_dataloader(eval_dataset, self.config.batch_size, 2, model.collate)

        print()

    def create_dataloader(self, dataset, batch_size, num_workers, collate_fn):
        device = self.model.get_device()
        # Disable multiprocessing workers only for MPS (Apple Silicon GPU)
        num_workers = 0 if device.type == 'mps' else num_workers

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
        # model_embeddings: [Batch, Time=1500, OurEmbedding=8]
        # known_speaker_embeddings: [Batch, OurEmbedding=8]

        model_device = self.model.get_device()
        collated_batch = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in collated_batch.items()
        }

        model_embeddings, known_speaker_embeddings = self.model(collated_batch)
        batch_size = len(model_embeddings)

        # TODO: Tweak to only take the mean over the time-interval from the original data clip

        # Average over the time dimension
        mean_model_embeddings = model_embeddings.mean(dim=1)
        assert mean_model_embeddings.shape == (batch_size, self.model.config.target_embedding_dimension)

        ### ?! The model embeddings are almost the same for almost all audios
        # These appear to be 0 on Cuda???
        # print(mean_model_embeddings)

        margin = 0.85

        # TODO: Possibly tweak to use proper triplet loss??

        # Does the model's embedding align with the actual speaker embedding?
        positive_contribution = torch.nn.CosineSimilarity(dim=-1)(mean_model_embeddings, known_speaker_embeddings)

        # Does the model's embedding align with all speakers?
        batch_size = len(positive_contribution)
        average_speaker_embedding = self.model.average_speaker_embedding().repeat(batch_size, 1)

        negative_contribution = torch.nn.CosineSimilarity(dim=-1)(mean_model_embeddings, average_speaker_embedding)

        total_loss = torch.max(torch.tensor(0), margin - positive_contribution + negative_contribution).sum(dim=0)

        return BatchResults(
            total_loss=total_loss,
            num_samples=len(model_embeddings),
            intermediates={}
        )
    
if __name__ == "__main__":
   print("Run default_models instead of this file")