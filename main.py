from src import logger
from src.pipeline.processed_data import DataProcessedPipeline
from src.pipeline.training_model import TrainingModelPipeline
from src.pipeline.evaluation_model import EvaluationModelPipeline

logger.info("Starting Training Model Pipeline...")

STAGE_NAME = 'LOADING AND PROCESSED DATA'
try:
    logger.info(f"========== Stage {STAGE_NAME} started =========")
    obj = DataProcessedPipeline()
    obj.main()
    logger.info(f"========== Stage {STAGE_NAME} completed =========\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'TRAINING MODEL'
try:
    logger.info(f"========== Stage {STAGE_NAME} started =========")
    obj = TrainingModelPipeline()
    obj.main()
    logger.info(f"========== Stage {STAGE_NAME} completed =========\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'EVALUATION MODEL'
try:
    logger.info(f"========== Stage {STAGE_NAME} started =========")
    obj = EvaluationModelPipeline()
    obj.main()
    logger.info(f"========== Stage {STAGE_NAME} completed =========\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e