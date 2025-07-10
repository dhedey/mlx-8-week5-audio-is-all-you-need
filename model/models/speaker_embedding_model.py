import torch.nn as nn
import torch
import numpy as np
import einops

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
        start_offsets_ms = torch.tensor([
            item["start_offset_ms"] for item in dataset_batch
        ], dtype=torch.long)
        end_offsets_ms = torch.tensor([
            item["end_offset_ms"] for item in dataset_batch
        ], dtype=torch.long)
        speaker_indices = torch.tensor([
            item["speaker_index"] for item in dataset_batch
        ], dtype=torch.long)


        return {
            "whisper_embeddings": whisper_embeddings,
            "speaker_indices": speaker_indices,
            "start_offsets_ms": start_offsets_ms,
            "end_offsets_ms": end_offsets_ms,
        }
    
    def single_mean_embedding_over_time_interval(self, whisper_embedding, length_seconds):
        return self.mean_embedding_over_time_interval(
            whisper_embeddings=whisper_embedding.unsqueeze(0),  # Add batch dimension
            start_offsets_ms=torch.tensor([0], dtype=torch.long),
            end_offsets_ms=torch.tensor([(length_seconds * 1000)//1], dtype=torch.long),
        ).squeeze(0)

    def mean_embedding_over_time_interval(self, whisper_embeddings, start_offsets_ms, end_offsets_ms) -> torch.Tensor:
        model_embeddings = self.speaker_embedding_model(whisper_embeddings)

        batch_size, time_size, embed_size = model_embeddings.shape

        mask = torch.zeros(batch_size, time_size, dtype=torch.long, device=model_embeddings.device)
        
        for row_mask, start_offset_ms, end_offset_ms in zip(mask, start_offsets_ms, end_offsets_ms):
            frame_rate = 50 # From whisper encodings
            start_offset = (start_offset_ms * frame_rate) // 1000
            end_offset = (end_offset_ms * frame_rate) // 1000
            row_mask[start_offset:end_offset] = 1

        mask_repeated = einops.repeat(mask, 'batch time -> batch time embed', embed = embed_size)
        # (Batch, Time, Embedding)
        return (mask_repeated * model_embeddings).sum(dim=1) / mask_repeated.sum(dim=1)

    def forward(self, collated_batch) -> tuple[torch.Tensor, torch.Tensor]:
        mean_embedding = self.mean_embedding_over_time_interval(
            collated_batch["whisper_embeddings"],
            collated_batch["start_offsets_ms"],
            collated_batch["end_offsets_ms"],
        )
        known_speaker_embeddings = self.known_speaker_embedding(collated_batch["speaker_indices"])

        return mean_embedding, known_speaker_embeddings


class SpeakerEmbeddingModelConfig(ModuleConfig):
    whisper_embedding_dimension: int
    target_embedding_dimension: int

class SpeakerEmbeddingModel(ModelBase):
    def __init__(self, model_name: str, config: SpeakerEmbeddingModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        self.fc1 = nn.Linear(config.whisper_embedding_dimension, config.target_embedding_dimension, bias=False)

    def process(self, soundfiles: list[tuple[np.ndarray, int]]) -> torch.Tensor:
        return torch.cat(
            [soundfile_to_whisper_embedding(soundfile) for soundfile in soundfiles],
            dim = 0 # Batch dimension
        )

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

        mean_model_embeddings, known_speaker_embeddings = self.model(collated_batch)
        batch_size = len(mean_model_embeddings)

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

        # Now look across each speaker embedding we have and try to move away from it
        negative_contributions = []
        for speaker_embedding in self.model.known_speaker_embedding.weight:
            repeated_speaker_embedding = speaker_embedding.repeat(batch_size, 1)
            negative_contributions.append(
                torch.nn.CosineSimilarity(dim=-1)(mean_model_embeddings, repeated_speaker_embedding)
            )
        negative_contributions = torch.stack(negative_contributions) # (EachSpeaker, Batch)
        negative_contribution = negative_contributions.mean(dim=0)   # (Batch)

        total_loss = torch.max(torch.tensor(0), margin - positive_contribution + negative_contribution).sum(dim=0)

        return BatchResults(
            total_loss=total_loss,
            num_samples=batch_size,
            intermediates={}
        )
    
if __name__ == "__main__":
   print("Run default_models instead of this file")