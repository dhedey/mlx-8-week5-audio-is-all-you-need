# Run as uv run -m model.start_train

import argparse
from model.harness import ModelBase, TrainingOverrides, select_device, WandbHelper
from .project_config import WANDB_ARGUMENTS_HELPER, DEFINED_MODELS, DEFAULT_MODEL_NAME
import wandb
import os

# Typical training process:
#
# TRAINING:
# - Run on GPU with --wandb parameter to save to the wandb project
# - Use model.continue_train if necessary with tweaks to learning-rate / end-epoch
#
# PRODUCTIONISING:
# - Use model.continue_train locally to download the model from wandb and copy it from
#   /artifacts to /trained (nb - we should have a dedicated script for this TBH)

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        '--end-epoch',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size for training (default: None, uses model default)'
    )
    parser.add_argument(
        '--batch-limit',
        type=int,
        default=None,
        help='Override the batch count per epoch/validation (default: None, uses full batch)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Override the default learning rate for training (default: None, uses model default)'
    )
    parser.add_argument(
        '--ignore-dataset-cache',
        action='store_true',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Set a random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        '--print-after-batches',
        type=int,
        default=None,
        help="Print a log message after N batches (default: None, uses model default)",
    )
    parser.add_argument(
        '--validate-after-epochs',
        type=int,
        default=1,
        help="Run validation after every N epochs (default: 1)",
    )
    parser.add_argument(
        '--immediate-validation',
        action='store_true',
        help='Run validation immediately after loading the model'
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping during training'
    )
    WANDB_ARGUMENTS_HELPER.add_arg_parser_arguments(parser)
    
    args = parser.parse_args()

    model_name = args.model

    if model_name not in DEFINED_MODELS:
        raise ValueError(f"Model '{model_name}' is not defined. The choices are: {list(DEFINED_MODELS.keys())}")

    model_definition = DEFINED_MODELS[model_name]

    wandb_helper = WANDB_ARGUMENTS_HELPER.handle_arguments(args)

    model = model_definition.model(
        model_name=model_name,
        config=model_definition.config,
    ).to(device)

    overrides = TrainingOverrides(
        print_detailed_parameter_counts=True,
        override_to_epoch=args.end_epoch,
        override_batch_size=args.batch_size,
        override_batch_limit=args.batch_limit,
        override_learning_rate=args.learning_rate,
        validate_after_epochs=args.validate_after_epochs,
        print_after_batches=args.print_after_batches,
        seed=args.seed,
        use_dataset_cache=not args.ignore_dataset_cache,
    )
    trainer = model_definition.trainer(
        model=model,
        config=model_definition.training_config,
        overrides=overrides,
    )

    wandb_helper.start_new_run(trainer=trainer)

    if args.immediate_validation:
        print("Immediate validation enabled, running validation before training:")
        trainer.validate()

    results = trainer.train()

    wandb_helper.upload_latest_and_best_model_snapshots(model_name=model_name)
    wandb_helper.finish()



        
