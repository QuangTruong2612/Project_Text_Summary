import os
from src.entity.config_entity import (
    DataProcessedConfig,
    TrainingModelConfig,
    EvaluationModelConfig
)
from src.utils import read_yaml, create_directories
from pathlib import Path
from src.constants import *
import os 


class ConfigurationManager:
    def __init__(self, config_path=Path(CONFIG_FILE_PATH), params_path=Path(PARAMS_FILE_PATH)):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])

        self.repo_name = os.getenv('MLFLOW_TRACKING_REPO')
        self.repo_owner = os.getenv('MLFLOW_TRACKING_USERNAME')

    def get_data_processed_config(self) -> DataProcessedConfig:
        config = self.config.data_processed
        create_directories([config.root_dir])
        return DataProcessedConfig(
            root_dir=Path(config.root_dir),
            data_file=Path(config.data_file),
            train_data_file=Path(config.train_data_file),
            test_data_file=Path(config.test_data_file),
            # params
            columns_not_use=self.params.COLUMNS_NOT_USE,
            columns_combined=self.params.COMBINED_COLUMNS,
            split_data=self.params.SPLIT_DATA
        )
    
    def get_training_model_config(self) -> TrainingModelConfig:
        config = self.config.training_model
        create_directories([config.root_dir])
        return TrainingModelConfig(
            root_dir=Path(config.root_dir),
            data_file= Path(config.data_file),
            save_model_path= Path(config.save_model_path),
            save_tokenizer_path=Path(config.save_tokenizer_path),
            # params
            model_name=self.params.MODEL_NAME,
            model_checkpoint=self.params.MODEL_CHECKPOINT,
            combined_columns=self.params.COMBINED_COLUMNS,
            split_data=self.params.SPLIT_DATA,
            learning_rate=self.params.LEARNING_RATE,
            weight_decay=self.params.WEIGHT_DECAY,
            train_batch=self.params.TRAIN_BATCH,
            eval_batch=self.params.EVAL_BATCH,
            epochs=self.params.EPOCHS,
            device=self.params.DEVICE,
            # mlflow
            repo_name=self.repo_name,
            repo_owner=self.repo_owner,
        )

    def get_evaluation_model_config(self) -> EvaluationModelConfig:
        config = self.config.evaluation_model
        create_directories([config.root_dir])
        return EvaluationModelConfig(
            root_dir=Path(config.root_dir),
            test_file=Path(config.test_file),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            model_name=self.params.MODEL_NAME,
            device=self.params.DEVICE,
            repo_name=self.repo_name,
            repo_owner=self.repo_owner
        )
