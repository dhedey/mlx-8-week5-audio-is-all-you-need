from model.modules.bert_custom_multi_modal_model import BertMultiModalModelConfig
from model.modules.qwen_multi_modal_model import QwenMultiModalModelConfig, SpecialTokensStrategy, \
    EmbeddingLearningStrategy
import model.models as models
from .trainer import UrbanSoundClassifierModelTrainer, UrbanSoundClassifierModelTrainingConfig
import torch

has_cuda = torch.cuda.is_available()

DEFAULT_MODEL_PARAMETERS = {
    "urban-sound-classifier-patch-transformer-v1": {
        "model_class": models.PatchTransformerUrbanSoundClassifierModel,
        "model": models.PatchTransformerUrbanSoundClassifierModelConfig(
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
        "model_trainer": UrbanSoundClassifierModelTrainer,
        "training": UrbanSoundClassifierModelTrainingConfig(
            batch_size=128,
            epochs=5,
            print_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
        ),
    },
    "urban-sound-classifier-cnn-v1": {
        "model_class": models.ConvolutionalUrbanSoundClassifierModel,
        "model": models.ConvolutionalUrbanSoundClassifierModelConfig(
            dropout=0.1,
        ),
        "model_trainer": UrbanSoundClassifierModelTrainer,
        "training": UrbanSoundClassifierModelTrainingConfig(
            batch_size=128,
            epochs=5,
            print_after_batches=1,
            learning_rate=0.0005,
            optimizer="adamw",
        ),
    },
}

DEFAULT_MODEL_NAME=list(DEFAULT_MODEL_PARAMETERS.keys())[0]

if __name__ == "__main__":
    import os
    from PIL import Image
    from .common import ModelBase, ModelTrainerBase, TrainingOverrides

    for model_name, parameters in DEFAULT_MODEL_PARAMETERS.items():
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
                model, training_state, training_config = ModelBase.load_advanced(model_name=best_version)
                break
        else:
            print(f"Model {model_name} or {best_version} could not be found in the trained or snapshots folder. Skipping.")
            continue

        model = model.eval()
        model.print_detailed_parameter_counts()

        trainer = ModelTrainerBase.load(
            model,
            training_config,
            training_state,
            overrides=TrainingOverrides(
                override_batch_limit=1, # Speed up the validation pass below
            )
        )

        print(f"Latest validation metrics: {trainer.latest_validation_results}")

        print(f"Running model to check it's working...")
        trainer.validate()
