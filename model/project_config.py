from .harness import PersistableData, ModelBase, ModuleConfig, TrainingConfig, ModelTrainerBase, WandbDefaults
import model.models as models
import torch
from typing import Optional, Any, Callable
from dataclasses import dataclass

WANDB_DEFAULTS = WandbDefaults(
    # default_entity="david-edey-machine-learning-institute",
    default_entity="mlx-bb",
    default_project="week5-audio-is-all-you-need",
)

has_cuda = torch.cuda.is_available()

@dataclass
class SweepDefinition:
    config: dict
    """
    Equivalent to wandb_sweep.yaml but in Python - https://docs.wandb.ai/guides/sweeps/sweep-config-keys/
    ```python
    {
        'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
            'name': 'validation_objective',
            'goal': 'maximise'
        },
        'parameters': {
            'embedding_size': {
                'values': [32, 64, 128]
            },
            'learning_rate': {
                'min': 0.00001,
                'max': 0.001,
                'distribution': 'log_uniform_values'
            },
            'epochs': {
                'value': 20
            },
        }
    }
    ```
    """
    model_config_mapper: Callable[[dict], ModuleConfig]
    """
    Maps an instance of the sweep config to a module config and training config.
    Read fields with e.g. config.epochs
    """
    training_config_mapper: Callable[[dict], TrainingConfig]
    """
    Maps an instance of the sweep config to a module config and training config.
    Read fields with e.g. config.epochs
    """

class ModelDefinition(PersistableData):
    model: type[ModelBase]
    config: ModuleConfig
    trainer: type[ModelTrainerBase]
    training_config: TrainingConfig
    sweep_configs: Optional[dict[str, SweepDefinition]] = None

# These are used by start_train
# * The first in this list is the default model
# * Others can be run with --model <model_name>
DEFINED_MODELS: dict[str, ModelDefinition] = {
    "speaker-embedding-david-two-towers": ModelDefinition(
        model=models.SpeakerEmbeddingTwoTowers,
        config=models.SpeakerEmbeddingTwoTowersConfig(
            total_speakers=319,
            target_embedding_dimension=8,
            whisper_embedding_dimension=384,
            embedding_model=models.LayeredSpeakerEmbeddingConfig(),
        ),
        trainer=models.SpeakerEmbeddingModelTrainer,
        training_config=models.SpeakerEmbeddingTrainingConfig(
            batch_size=32,
            epochs=5,
            recalculate_running_loss_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
            loss_kind="david",
        ),
        sweep_configs={
            "default": SweepDefinition(
                config={
                    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
                    'metric': {
                        'name': 'validation_objective',
                        'goal': 'maximize'
                    },
                    'parameters': {
                        'embedding_dimension': {
                            'values': [8, 16]
                        },
                        'learning_rate': {
                            'min': 0.00001,
                            'max': 0.001,
                            'distribution': 'log_uniform_values'
                        },
                        'loss_kind': {
                            'values': ["david", "triplet"]
                        },
                        'model_kind': {
                            'values': ["layered", "linear"]
                        },
                    }
                },
                model_config_mapper=lambda config: models.SpeakerEmbeddingTwoTowersConfig(
                    total_speakers=319,
                    target_embedding_dimension=config.embedding_dimension,
                    whisper_embedding_dimension=384,
                    embedding_model=models.LayeredSpeakerEmbeddingConfig() if config.model_kind == "layered" else models.LinearSpeakerEmbeddingConfig(),
                ),
                training_config_mapper=lambda config: models.SpeakerEmbeddingTrainingConfig(
                    batch_size=32,
                    epochs=3,
                    recalculate_running_loss_after_batches=1,
                    learning_rate=config.learning_rate,
                    optimizer="adamw",
                    loss_kind=config.loss_kind,
                ),
            )
        }
    ),
    "speaker-embedding-triplet-two-towers": ModelDefinition(
        model=models.SpeakerEmbeddingTwoTowers,
        config=models.SpeakerEmbeddingTwoTowersConfig(
            total_speakers=319,
            target_embedding_dimension=8,
            whisper_embedding_dimension=384,
            embedding_model=models.LayeredSpeakerEmbeddingConfig(),
        ),
        trainer=models.SpeakerEmbeddingModelTrainer,
        training_config=models.SpeakerEmbeddingTrainingConfig(
            batch_size=32,
            epochs=5,
            recalculate_running_loss_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
            loss_kind="triplet",
        ),
    ),
    "speaker-embedding-basic-two-towers": ModelDefinition(
        model=models.SpeakerEmbeddingTwoTowers,
        config=models.SpeakerEmbeddingTwoTowersConfig(
            total_speakers=319,
            target_embedding_dimension=8,
            whisper_embedding_dimension=384,
            embedding_model=models.LinearSpeakerEmbeddingConfig()
        ),
        trainer=models.SpeakerEmbeddingModelTrainer,
        training_config=models.SpeakerEmbeddingTrainingConfig(
            batch_size=32,
            epochs=5,
            recalculate_running_loss_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
            loss_kind="david",
        ),
    ),
    "urban-sound-classifier-patch-transformer-v1": ModelDefinition(
        model=models.PatchTransformerUrbanSoundClassifierModel,
        config=models.PatchTransformerUrbanSoundClassifierModelConfig(
            embed_dim=128,
            patch_size_mels=16,
            patch_size_time=3, # 80 / 16 = 5 patches per second
            num_mels=64,
            time_frames=321, # Across 4 seconds, so 12.5ms / frame
            num_heads=4,
            num_layers=2,
            mlp_dim=256,
            dropout=0.1
        ),
        trainer=models.UrbanSoundClassifierModelTrainer,
        training_config=models.UrbanSoundClassifierModelTrainingConfig(
            batch_size=128,
            epochs=5,
            recalculate_running_loss_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
        )
    ),
    "urban-sound-classifier-cnn-v1": ModelDefinition(
        model=models.ConvolutionalUrbanSoundClassifierModel,
        config=models.ConvolutionalUrbanSoundClassifierModelConfig(
            dropout=0.1
        ),
        trainer=models.UrbanSoundClassifierModelTrainer,
        training_config=models.UrbanSoundClassifierModelTrainingConfig(
            batch_size=128,
            epochs=5,
            recalculate_running_loss_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
        ),
    ),
}

DEFAULT_MODEL_NAME=list(DEFINED_MODELS.keys())[0]

if __name__ == "__main__":
    from model.harness import ModelBase, TrainingOverrides, ModelTrainerBase, TrainingState, PersistableData, TrainingConfig

    for model_name, parameters in DEFINED_MODELS.items():
        best_version = f"{model_name}-best"
        lookup_locations = [
            (best_version, "trained"),
            (model_name, "trained"),
            (best_version, "snapshots"),
            (model_name, "snapshots"),
        ]

        for name, location in lookup_locations:
            if ModelBase.exists(model_name=name, location=location):
                print(f"Loading Model: {location}/{name}")
                model, training_state_dict, training_config_dict = ModelBase.load_advanced(model_name=best_version)
                break
        else:
            print(f"Model {model_name} or {best_version} could not be found in the trained or snapshots folder. Skipping.")
            continue

        model = model.eval()
        model.print_detailed_parameter_counts()

        trainer = ModelTrainerBase.load(
            model,
            training_config_dict,
            state=TrainingState.from_dict(training_state_dict),
            overrides=TrainingOverrides(
                override_batch_limit=1, # Speed up the validation pass below
            )
        )

        print(f"Latest validation metrics: {trainer.latest_validation_results}")

        print(f"Running model to check it's working...")
        trainer.validate()
