from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataProcessedConfig:
    root_dir: Path
    data_file: Path
    train_data_file: Path
    test_data_file: Path
    columns_not_use: list[str]
    split_data: list[float]

@dataclass(frozen=True)
class TrainingModelConfig:
    root_dir: Path
    data_file: Path
    save_path: Path

    model_name: str
    model_checkpoint: str
    combined_columns: list[str]
    split_data: list[float]
    learning_rate: float
    weight_decay: float
    train_batch: int
    eval_batch: int
    epochs: int    
    device: str

    repo_name: str
    repo_owner: str
    