from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import pandas as pd
import os

from torch.utils.data import random_split

from src.entity.config_entity import TrainingModelConfig
from src import logger

import dagshub
import mlflow
from metrics import compute_bleu_score, compute_rouge_score
from .pre_dataset import PreDataset

class TrainingModel:
    def __init__(self, config: TrainingModelConfig):
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint, use_fast=False)
        self.compute_metrics = self._compute_multiple_metrics

    def _compute_multiple_metrics(self, eval_pred):
        rouge_results = compute_rouge_score(self.tokenizer, eval_pred)
        bleu_results = compute_bleu_score(self.tokenizer, eval_pred)

        return {
            "rouge1": round(rouge_results["rouge1"], 4),
            "rouge2": round(rouge_results["rouge2"], 4),
            "rougeL": round(rouge_results["rougeL"], 4),
            "bleu": round(bleu_results["score"], 4)
        }

    def load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.config.data_file)
            logger.info(f"Load data for training model succeeded! File: {self.config.data_file}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")  # FIX: logger.debug → logger.error
            raise e

    def pre_data(self):
        dataset = PreDataset(self.load_data(), self.tokenizer)
        data_train, data_eval = random_split(dataset, self.config.split_data)
        return data_train, data_eval

    def train(self):

        if "MLFLOW_TRACKING_URI" not in os.environ:
            logger.info("Running locally, initializing DagsHub...")
            dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name, mlflow=True)
        else:
            logger.info("Running in CI/CD, using existing environment variables.")

        exp_name = "Project_Tracking_Model_Text_Summary"
        logger.info(f"Setting MLflow experiment to: {exp_name}")
        mlflow.set_experiment(exp_name)

        with mlflow.start_run():
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
            data_train, data_eval = self.pre_data()

            training_args = Seq2SeqTrainingArguments(
                output_dir=f"{self.config.root_dir}/results_summary",
                eval_strategy="epoch",
                learning_rate=float(self.config.learning_rate),
                logging_strategy="steps",
                logging_steps=10,
                per_device_train_batch_size=self.config.train_batch,
                per_device_eval_batch_size=self.config.eval_batch,
                weight_decay=self.config.weight_decay,
                num_train_epochs=self.config.epochs,
                predict_with_generate=True,
                fp16=True,
                save_strategy="epoch", 
                save_total_limit=1,
                report_to="mlflow",
            )

            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=data_train,
                eval_dataset=data_eval,
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

            logger.info("========= START TRAINING MODEL =========")
            trainer.train()

            logger.info("Evaluating on eval data...")
            eval_results = trainer.evaluate()
            logger.info(f"Final eval results: {eval_results}")

            trainer.save_model(self.config.save_model_path)
            self.tokenizer.save_pretrained(self.config.save_tokenizer_path)
            # FIX: self.config.save_path không tồn tại → dùng save_model_path
            logger.info(f"Model saved locally to: {self.config.save_model_path}")

            logger.info("Uploading model to MLflow Registry...")
            components = {
                "model": trainer.model,
                "tokenizer": self.tokenizer,
            }

            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path="models",
                registered_model_name=self.config.model_name,
                task="summarization",
                pip_requirements="requirements.txt",
            )

            logger.info("Model registered to MLflow Registry successfully!")