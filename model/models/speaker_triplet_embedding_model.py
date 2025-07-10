import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.harness import (
    ModelBase,
    ModuleConfig,
    BatchResults,
    ModelTrainerBase,
    TrainingConfig,
    TrainingState,
    TrainingOverrides,
    select_device_no_mps,
)
from .prepared_datasets import generate_speaker_tagged_dataset, soundfile_to_whisper_embedding
from typing import Optional, Any, List, Tuple


class SpeakerTripletEmbeddingTwoTowersConfig(ModuleConfig):
    total_speakers: int             # e.g. 109 for VCTK
    target_embedding_dimension: int # output dim (e.g. 256)
    whisper_embedding_dimension: int# input dim from Whisper (e.g. 768)
    inner_model_name: str = "speaker-embedding"


class SpeakerTripletEmbeddingTwoTowers(ModelBase):
    def __init__(self, model_name: str, config: SpeakerTripletEmbeddingTwoTowersConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        # two-tower: one tower transforms whisper embeddings to speaker space
        self.speaker_embedding_model = SpeakerTripletEmbeddingModel(
            model_name=self.config.inner_model_name,
            config=SpeakerTripletEmbeddingModelConfig(
                whisper_embedding_dimension=self.config.whisper_embedding_dimension,
                target_embedding_dimension=self.config.target_embedding_dimension,
            ),
        )
        # second tower: embedding lookup for known speaker IDs
        self.known_speaker_embedding = nn.Embedding(
            num_embeddings=self.config.total_speakers,
            embedding_dim=self.config.target_embedding_dimension,
        )

    def collate(self, dataset_batch: List[Any]) -> dict:
        # dataset_batch items: {"soundfile": (np.ndarray, int), "speaker_index": int}
        # convert to whisper embeddings then collate speaker indices
        whisper_embs = []
        speaker_idxs = []
        for item in dataset_batch:
            emb = item["whisper_embedding"]
            if isinstance(emb, list):
              emb = torch.tensor(emb)
            elif isinstance(emb, np.ndarray):
              emb = torch.from_numpy(emb)

            # If emb has shape [1, T, D], squeeze the batch dimension
            if emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb.squeeze(0)

            whisper_embs.append(emb)
            speaker_idxs.append(item["speaker_index"])

        # pad to same T length
        whisper_embeddings = nn.utils.rnn.pad_sequence(
            whisper_embs, batch_first=True
        )  # [B, T_max, D_whisper]
        speaker_indices = torch.tensor(speaker_idxs, dtype=torch.long)
        return {"whisper_embeddings": whisper_embeddings, "speaker_indices": speaker_indices}

    def forward(self, collated_batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        # model tower output: [B, T, D]
        model_embs = self.speaker_embedding_model(collated_batch["whisper_embeddings"])
        # lookup known tower: [B, D]
        known_embs = self.known_speaker_embedding(collated_batch["speaker_indices"])
        return model_embs, known_embs

    def average_speaker_embedding(self) -> torch.Tensor:
        # global average of learned speaker embeddings for negative sampling
        return self.known_speaker_embedding.weight.mean(dim=0)


class SpeakerTripletEmbeddingModelConfig(ModuleConfig):
    whisper_embedding_dimension: int
    target_embedding_dimension: int


class SpeakerTripletEmbeddingModel(ModelBase):
    def __init__(self, model_name: str, config: SpeakerTripletEmbeddingModelConfig):
        super().__init__(model_name=model_name, config=config)
        self.config = config
        d_in = config.whisper_embedding_dimension
        d_mid = config.target_embedding_dimension * 2
        d_out = config.target_embedding_dimension
        # MLP: linear -> ReLU -> BN -> dropout -> linear
        self.fc1 = nn.Linear(d_in, d_mid)
        self.act1 = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(d_mid)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(d_mid, d_out)

    def forward(self, whisper_embeddings: torch.Tensor) -> torch.Tensor:
        # whisper_embeddings: [B, T, d_in]
        x = self.fc1(whisper_embeddings)           # [B, T, d_mid]
        x = self.act1(x)
        # BN over channels: reshape to [B*T, d_mid]
        B, T, C = x.shape
        x = x.view(B * T, C)
        x = self.bn(x)
        x = x.view(B, T, C)
        x = self.drop2(x)
        x = self.fc2(x)                           # [B, T, d_out]
        # L2 normalize per time-step
        x = F.normalize(x, p=2, dim=-1)
        return x


class SpeakerTripletEmbeddingModelTrainer(ModelTrainerBase):
    def __init__(
        self,
        model: SpeakerTripletEmbeddingTwoTowers,
        config: TrainingConfig,
        overrides: Optional[TrainingOverrides] = None,
        continuation: Optional[TrainingState] = None,
    ):
        super().__init__(model=model, config=config, overrides=overrides, continuation=continuation)
        # prepare data
        train_ds, eval_ds, total_speakers = generate_speaker_tagged_dataset()
        assert total_speakers <= model.config.total_speakers, (
            f"Dataset has {total_speakers} speakers, but model allows {model.config.total_speakers}."
        )
        device = select_device_no_mps()
        self.train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            collate_fn=model.collate,
        )
        self.test_loader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            collate_fn=model.collate,
        )
        # loss and optimizer
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.learning_rate
        )

    @property
    def train_data_loader(self):
        return self.train_loader

    @property
    def validation_data_loader(self):
        return self.test_loader

    def process_batch(self, collated_batch: dict) -> BatchResults:
        device = self.model.get_device()
        # send to device
        whisper_embs = collated_batch["whisper_embeddings"]
        speaker_idxs = collated_batch["speaker_indices"]
        # forward towers
        model_embs_time, known_embs_pos = self.model(
            {"whisper_embeddings": whisper_embs, "speaker_indices": speaker_idxs}
        )
        # [B, T, D] -> average over time
        mean_model_embs = model_embs_time.mean(dim=1)  # [B, D]
        # Construct negatives: for each anchor, choose a different speaker's embedding
        # Here, we sample random negatives within batch
        batch_size = mean_model_embs.size(0)
        all_indices = torch.arange(batch_size, device=device)
        neg_indices = (all_indices + torch.randint(1, batch_size, (batch_size,), device=device)) % batch_size
        known_embs_neg = known_embs_pos[neg_indices]
        # compute triplet loss: anchor=mean_model_embs, pos=known_embs_pos, neg=known_embs_neg
        loss = self.triplet_loss(mean_model_embs, known_embs_pos, known_embs_neg)
        return BatchResults(total_loss=loss, num_samples=batch_size, intermediates={})

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        return self.process_batch(batch)

    def validation_step(self, batch, batch_idx):
        return self.process_batch(batch)
