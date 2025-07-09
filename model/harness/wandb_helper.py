from .base_trainer import TrainingState, ModelTrainerBase
from .base_model import ModelBase
import os
import wandb

class WandbHelper:
    def __init__(
        self,
        enabled: bool,
        wandb_entity: str,
        wandb_project: str,
        artifact_upload_enabled: bool,
    ):
        self.enabled = enabled
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.should_upload_artifacts = artifact_upload_enabled

    def start_new_run(self, trainer: ModelTrainerBase):
        if not self.enabled:
            return

        run_id = wandb.util.generate_id()
        model_name = trainer.model.model_name
        model_class = trainer.model.__class__.__name__
        model_config = trainer.model.config.to_dict()
        training_config = trainer.config.to_dict()

        wandb.init(
            id=run_id,
            entity=self.wandb_entity,
            project=self.wandb_project,
            name=f"train-{model_name}-{run_id}",
            config={
                "model_name": model_name,
                "model_class": model_class,
                "model_config": model_config,
                "training_config": training_config,
                "from_epoch": trainer.epoch,
            },
        )

    @staticmethod
    def is_running() -> bool:
        return wandb.run is not None

    def finish(self) -> None:
        if self.is_running():
            wandb.finish()

    def upload_latest_and_best_model_snapshots(self, model_name: str):
        if not self.is_running():
            print("Wandb is not enabled, so skipping of model snapshots")
            return

        if not self.should_upload_artifacts:
            print("Skipping uploading of model snapshots, due to config.should_upload_artifacts = False")
            return

        self.upload_model_snapshot_if_exists(
            model_name=model_name, # The latest snapshot
            artifact_name=f"{model_name}-final",
            artifact_description=f"Final model: {model_name}"
        )
        self.upload_model_snapshot_if_exists(
            model_name=f"{model_name}-best", # The best snapshot
            artifact_name=f"{model_name}-best",
            artifact_description=f"Best validation model: {model_name}"
        )
        print()

    def upload_model_snapshot_if_exists(
        self,
        model_name: str,
        artifact_name: str,
        artifact_description: str,
    ):
        model_path = ModelBase.resolve_path(model_name)

        if ModelBase.exists(model_path=model_path):
            print(f"Uploading {model_name} from {model_path}")
            self.upload_model_snapshot(
                model_path=model_path,
                artifact_name=artifact_name,
                artifact_description=artifact_description
            )
        else:
            print(f"Could not find {model_name} at {model_path}, skipping upload")

    def upload_model_snapshot(
        self,
        model_path,
        artifact_name,
        artifact_description,
    ):
        model, training_state_dict, training_config_dict = ModelBase.load_advanced(model_path=model_path)

        training_state = TrainingState.from_dict(training_state_dict)
        latest_train_results = training_state.latest_training_results
        latest_validation_results = training_state.latest_validation_results

        artifact_metadata = {
            "model_name": model.model_name,
            "model_class": model.__class__.__name__,
            "model_config": model.config.to_dict(),
            "training_config": training_config_dict,
            "training_state": training_state_dict,
            "validation_objective": latest_validation_results.objective,
            "validation_loss": latest_validation_results.train_comparable_loss,
            "train_loss": latest_train_results.average_loss,
            "total_epochs": latest_train_results.total_epochs,
        }

        WandbHelper.upload_model_artifact(
            file_path=model_path,
            artifact_name=artifact_name,
            metadata=artifact_metadata,
            description=artifact_description
        )


    @staticmethod
    def upload_model_artifact(
        file_path: str,
        artifact_name: str,
        description: str,
        metadata: dict = None,
        artifact_file_name: str = "model.pt",
    ):
        """
        Upload a model as a wandb artifact.

        Args:
            file_path: Path to the saved model file
            artifact_name: Optional custom artifact name (defaults to model_name)
            description: Description for the artifact
            metadata: Optional metadata dictionary to include with the artifact
            artifact_file_name: Optional file name for the model inside the artifact. (default: model.pt)
        """
        if not os.path.exists(file_path):
            print(f"⚠️ Model file not found: {file_path}")
            return None

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=description,
            metadata=metadata or {}
        )

        # Add the model file
        artifact.add_file(file_path, name=artifact_file_name)

        # Log the artifact
        wandb.log_artifact(artifact)
        print(f"📦 Uploaded model artifact: {artifact_name} containing {artifact_file_name}")

        return artifact

class WandbArgumentsHelper:
    def __init__(self, default_enable: bool, uploading: bool, downloading: bool):
        self.default_enable = default_enable
        self.uploading = uploading
        self.downloading = downloading

    def handle_arguments(self, args, uploading: bool, downloading: bool) -> WandbHelper:
        other_args_present = args.wandb_entity is not None or args.wandb_project is not None
        if uploading:
            other_args_present = other_args_present or args.wandb_no_upload is not None
        if downloading:
            other_args_present = other_args_present or args.wandb_run or args.wandb_artifact_name is not None or args.wandb_artifact_type is not None or args.wandb_artifact_version is not None

        if self.default_enable:
            if args.wandb_enable:
                if other_args_present:
                    raise ValueError("You cannot both provide --disable-wandb and other wandb parameters")
                enabled = False
            else:
                enabled = True
        else:
            enabled = args.wandb_disable or other_args_present

        if enabled:
            # Force a login prompt early in the process
            # This means that even if creation of a run is delayed (by e.g. initializing datasets) we will already
            # be logged in
            wandb.login(verify = True)

        return WandbHelper(
            enabled=enabled,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            artifact_upload_enabled=not args.wandb_no_upload,
        )

class WandbDefaults:
    def __init__(self, default_entity, default_project):
        self.default_entity = default_entity
        self.default_project = default_project
        self.default_enable = os.environ.get('WANDB_DEFAULT_ENABLED', False)

    def add_arg_parser_arguments(self, parser, uploading: bool, downloading: bool) -> WandbArgumentsHelper:
        if self.default_enable:
            parser.add_argument(
                '--wandb-disable',
                action='store_true',
                help='Disable using wandb'
            )
        else:
            parser.add_argument(
                '--wandb-enable',
                # Aliases
                '--wandb',
                action='store_true',
                help='Enable wandb logging'
            )
        parser.add_argument(
            '--wandb-entity',
            default=self.default_entity,
            help='W&B entity name (used if --from-wandb is set)'
        )
        parser.add_argument(
            '--wandb-project',
            default=self.default_project,
            help=f'W&B project name (default: {self.default_project})'
        )
        if uploading:
            parser.add_argument(
                '--wandb-no-upload',
                action='store_true',
                help='Disable artifact uploading to W&B (if wandb is enabled)'
            )
        if downloading:
            parser.add_argument(
                '--model-source',
                choices=["default", "local", "wandb"],
                default="default",
                help='Where to load the model from (defaults to wandb if it exists)'
            )
            parser.add_argument(
                '--wandb-run',
                # Aliases
                '--wandb-continue-run',
                type=str,
                default=None,
                help='Continue from the specified wandb run ID.'
            )
            parser.add_argument(
                '--wandb-artifact-name',
                type=str,
                default=None,
                help='Continue by reading a specific artifact name. Not needed if --wandb-continue-run is provided.'
            )
            parser.add_argument(
                '--wandb-artifact-type',
                type=str,
                choices=["best", "final"],
                default="best",
                help='Which model type to download. Not used if --wandb-artifact-name is specified.'
            )
            parser.add_argument(
                '--wandb-artifact-version',
                type=str,
                default="latest",
                help='The version of the artifact to load. Not used if --wandb-artifact-name is specified.'
            )

        return WandbArgumentsHelper(
            default_enable=self.default_enable,
            uploading=uploading,
            downloading=downloading
        )
