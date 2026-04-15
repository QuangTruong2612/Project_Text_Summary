from src.configs.configuration import ConfigurationManager
from src.components.processed_data import ProcessedData
from src import logger

STAGE_NAME = "Data Processed Stage"

class DataProcessedPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_processed_config = config.get_data_processed_config()
        data_processed = ProcessedData(data_processed_config)
        data = data_processed.processed()
        data_processed.save_data(data)

