from src.configs.configuration import ConfigurationManager
from src.components.evaluation_model import EvaluationModel
from src import logger

STAGE_NAME = "Evaluation Model Stage"

class EvaluationModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_model_config()
        evaluation_model = EvaluationModel(evaluation_config)
        evaluation_model.evaluation()
        

