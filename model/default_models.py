from model.modules.bert_custom_multi_modal_model import BertMultiModalModelConfig
from model.modules.qwen_multi_modal_model import QwenMultiModalModelConfig, SpecialTokensStrategy, \
    EmbeddingLearningStrategy
import model.models as models
from .trainer import UrbanSoundClassifierModelTrainer, UrbanSoundClassifierModelTrainingConfig
import torch

has_cuda = torch.cuda.is_available()

DEFAULT_MODEL_PARAMETERS = {
    "urban-sound-classifier-cnn-v1": {
        "model_class": models.ConvolutionalUrbanSoundClassifierModel,
        "model": models.ConvolutionalUrbanSoundClassifierModelConfig(),
        "model_trainer": UrbanSoundClassifierModelTrainer,
        "training": UrbanSoundClassifierModelTrainingConfig(
            # Note: This is a very small batch size for Qwen models, but it is necessary to fit in memory.
            batch_size=4,
            epochs=1,
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

        if not ModelBase.exists(model_name=best_version):
            print(f"Model {best_version} does not exist locally. Skipping.")
            continue

        print(f"Loading Model: {best_version}")

        model, training_state, training_config = ModelBase.load_advanced(model_name=best_version)
        model = model.eval()

        model.print_detailed_parameter_counts()

        trainer = ModelTrainerBase.load(model, training_config, training_state)

        print(f"Latest validation metrics: {trainer.latest_validation_results}")

        print(f"Running model to check it's working...")
        trainer.additional_custom_validation()
