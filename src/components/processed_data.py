import re
import pandas as pd
from src import logger
from src.entity.config_entity import DataProcessedConfig
from pathlib import Path

class ProcessedData:
    def __init__(self, config: DataProcessedConfig):
        self.config = config

    def load_data(self)->pd.DataFrame:
        try:
            data = pd.read_csv(self.config.data_file)
            logger.info(f'Load data for training model successed! File data: {self.config.data_file}')
            return data 
        except Exception as e:
            logger.debug(f"Error: {e}")
            raise e
    
    def processed_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def processed(self) -> pd.DataFrame:
        data = self.load_data()

        # processed null
        data = data.dropna(ignore_index=True)

        # drop columns not use
        data = data.drop(columns=self.config.columns_not_use, inplace=True)

        # get columns use
        columns = list(data.columns)
        for column in columns:
            data[column] = data[column].apply(self.processed_text)
            
        return data 

    @staticmethod
    def save_data(self, data: pd.DataFrame):
        try:
            train_size, test_size = self.config.split_data
            
            data_size = len(data)
            split_index = int(train_size * data_size)
            train_data = data[:split_index]
            test_data = data[split_index:]

            train_data.to_csv(self.config.train_data_file, index=False)
            logger.info(f"Save file train data completed! Name file: {self.config.train_data_file}")

            test_data.to_csv(self.config.test_data_file, index=False)
            logger.info(f"Save file test data completed! Name file: {self.config.test_data_file}")

        except Exception as e:
            logger.debug(f"Error: {e}")
            raise e