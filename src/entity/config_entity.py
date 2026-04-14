from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataProcessedConfig:
    root_dir: Path
    data_file: Path
    processed_data_file:Path
    columns_not_use: list[str]
