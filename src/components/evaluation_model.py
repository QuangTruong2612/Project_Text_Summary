from src.entity.config_entity import EvaluationModelConfig
from src import logger

import dagshub
import mlflow
from mlflow.tracking import MlflowClient  # FIX: missing import

from metrics import compute_bleu_score, compute_rouge_score

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)

import torch
import pandas as pd
from .pre_dataset import PreDataset


class EvaluationModel:
    def __init__(self, config: EvaluationModelConfig):
        self.config = config
        self.device = config.device

        self.model_new = AutoModelForSeq2SeqLM.from_pretrained(config.model_path).to(self.device)
        self.tokenizer_new = AutoTokenizer.from_pretrained(config.tokenizer_path)

        self.client = MlflowClient()

    def load_data(self, tokenizer):
        try:
            data = pd.read_csv(self.config.test_file)
            logger.info(f"Load data for testing model succeeded! File: {self.config.test_file}")
            dataset = PreDataset(data, tokenizer)
            return dataset
        except Exception as e:
            logger.error(f"Error loading data: {e}") 
            raise e

    def get_latest_model_version_number(self, model_name: str) -> str | None:
        """
        Chỉ dùng để lấy SỐ VERSION (metadata) để promote,
        không dùng để load weights.
        """
        results = self.client.search_model_versions(f"name='{model_name}'")
        if not results:
            logger.warning(f"Chưa tìm thấy model '{model_name}' trên Registry.")
            return None

        latest_version = max(int(v.version) for v in results)
        return str(latest_version)

    def get_champion_model(self, model_name: str):
        try:
            model_uri = f"models:/{model_name}@champion" 
            components = mlflow.transformers.load_model(
                model_uri=model_uri,
                return_type="components"
            )
            logger.info("Đã load thành công phiên bản @champion!")
            tokenizer = components["tokenizer"]
            model = components["model"]
            return model, tokenizer
        except Exception as e:
            logger.info(f"--> Chưa có Champion nào ({e}). Model hiện tại sẽ là Champion đầu tiên.")
            return None

    def promote_champion(self,
                         model_name: str,
                         version: str | None,
                         current_metrics: dict,
                         old_metrics: dict | None):
        if version is None:
            logger.warning("Không tìm thấy version trên Registry để promote.")
            return

        if old_metrics is None:
            promote = True
        else:
            promote = (
                current_metrics['bleu']   > old_metrics['bleu']   and
                current_metrics['rouge1'] > old_metrics['rouge1'] and
                current_metrics['rouge2'] > old_metrics['rouge2'] and
                current_metrics['rougeL'] > old_metrics['rougeL']
            )

        if promote:
            logger.info(f"--> CHÚC MỪNG! Version {version} đang được thăng cấp lên @champion")
            self.client.set_registered_model_alias(model_name, "champion", version)
        else:
            logger.info(f"--> Rất tiếc. Version {version} không vượt qua được Champion hiện tại.")

    def _compute_multiple_metrics(self, eval_pred):
        rouge_result = compute_rouge_score(self.tokenizer_new, eval_pred)
        bleu_result  = compute_bleu_score(self.tokenizer_new, eval_pred)

        return {
            "rouge1": round(rouge_result["rouge1"], 4),
            "rouge2": round(rouge_result["rouge2"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
            "bleu":   round(bleu_result["score"],   4),
        }

    def _evaluate_single_model(self, model, tokenizer) -> dict:
        dataset = self.load_data(tokenizer)

        eval_args = Seq2SeqTrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            report_to="none",       # Tắt log để tránh rác vào MLflow Tracking
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=eval_args,
            eval_dataset=dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_multiple_metrics,
        )

        metrics = trainer.evaluate()

        return {
            'rouge1': metrics.get('eval_rouge1', 0),
            'rouge2': metrics.get('eval_rouge2', 0),
            'rougeL': metrics.get('eval_rougeL', 0),
            'bleu':   metrics.get('eval_bleu',   0),
        }


    def evaluation(self):
        logger.info("========= ĐÁNH GIÁ MÔ HÌNH MỚI =========")
        new_metrics = self._evaluate_single_model(self.model_new, self.tokenizer_new)
        logger.info(f"New Model Metrics: {new_metrics}")

        logger.info("========= TÌM KIẾM CHAMPION =========")
        champion_data = self.get_champion_model(self.config.model_name)

        old_metrics = None
        if champion_data is not None:
            champion_model, champion_tokenizer = champion_data
            logger.info("========= ĐÁNH GIÁ CHAMPION =========")
            old_metrics = self._evaluate_single_model(champion_model, champion_tokenizer)
            logger.info(f"Champion Metrics: {old_metrics}")

            import torch
            del champion_model
            torch.cuda.empty_cache()

        latest_version = self.get_latest_model_version_number(self.config.model_name)
        self.promote_champion(self.config.model_name, latest_version, new_metrics, old_metrics)