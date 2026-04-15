from src.configs.configuration import ConfigurationManager
from src.components.training_model import TrainingModel
from src import logger

STAGE_NAME = "Training Model Stage"

class TrainingModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_model_config()
        training_model = TrainingModel(training_config)
        training_model.train()
        

