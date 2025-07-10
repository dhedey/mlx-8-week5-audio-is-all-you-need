import torch.nn as nn
import torch

from model.harness import ModelBase, ModuleConfig, BatchResults, ModelTrainerBase, TrainingConfig, TrainingState, TrainingOverrides
from .prepared_datasets import generate_speaker_tagged_dataset
from typing import Optional


class SpeakerEmbeddingTwoTowersConfig(ModuleConfig):
    total_speakers: int # 109 in badayvedat/VCTK
    target_embedding_dimension: int
    whisper_embedding_dimension: int

class SpeakerEmbeddingTwoTowers(ModelBase):
    def __init__(self, model_name: str, config: SpeakerEmbeddingTwoTowersConfig):
        super().__init__(model_name=f"{model_name}-two-towers", config=config)
        self.config = config

        self.speaker_embedding_model = SpeakerEmbeddingModel(
            model_name=model_name,
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
    def __init__(self, model_name: str, config: SpeakerEmbeddingModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        self.fc1 = nn.Linear(config.whisper_embedding_dimension, config.target_embedding_dimension, bias=False)

    def forward(self, whisper_embeddings: torch.Tensor) -> torch.Tensor:
        return self.fc1(whisper_embeddings)


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
        print(mean_model_embeddings)

        margin = 0.7

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