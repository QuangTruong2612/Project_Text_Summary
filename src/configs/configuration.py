import os
from src.entity.config_entity import DataProcessedConfig
from src.utils import read_yaml, create_directories
from pathlib import Path
from src.constants import *

class ConfigurationManager:
    def __init__(self, config_path=Path(CONFIG_FILE_PATH), params_path=Path(PARAMS_FILE_PATH)):
        self.config = self.read_yaml(config_path)
        self.params = self.read_yaml(params_path)
        create_directories([self.config.artifacts_root])

    def get_data_processed_config(self) -> DataProcessedConfig:
        config = self.config.data_processed
        create_directories([config.root_dir])
        return DataProcessedConfig(
            root_dir=Path(config.root_dir),
            data_file=Path(config.data_file),
            tprocessed_data_file=Path(config.processed_data_file),
            columns_not_use=self.params.COLUMNS_NOT_USE,
        )