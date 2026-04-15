from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from pyvi import ViTokenizer
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

from src.entity.config_entity import TrainingModelConfig
from src import logger

import dagshub
import mlflow
import mlflow.pytorch

from metrics import compute_bleu_score, compute_rouge_score

class PreDataset(Dataset):
    def __init__(self, df, tokenizer, max_source_len=512, max_target_len=128):
        self.text_list = df['Combined'].apply(ViTokenizer.tokenize).tolist()
        self.summary = df['Summary'].apply(ViTokenizer.tokenize).tolist()
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = str(self.text_list[idx])
        summary = str(self.summary[idx])

        encoding_text = self.tokenizer(
            text,
            add_special_tokens=True, 
            max_length=self.max_source_len,
            padding='max_length',    
            truncation=True,         
            return_attention_mask=True, 
            return_tensors='pt',
        )

        encoding_summary = self.tokenizer(
            summary,
            add_special_tokens=True, 
            max_length=self.max_target_len,
            padding='max_length',    
            truncation=True,         
            return_attention_mask=True, 
            return_tensors='pt',
        )

        labels = encoding_summary['input_ids'].flatten()

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            # .flatten() để làm phẳng tensor từ kích thước [1, max_len] thành [max_len]
            'input_ids': encoding_text['input_ids'].flatten(),
            'attention_mask': encoding_text['attention_mask'].flatten(),
            'labels': labels,
        }

class TrainingModel:
    def __init__(self, config: TrainingModelConfig):
        self.config = config
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
        self.compute_metrics = self._compute_multiple_metrics
    
    def _compute_multiple_metrics(self, eval_pred):
        rouge_result = compute_rouge_score(self.tokenizer, eval_pred)
        bleu_result = compute_bleu_score(self.tokenizer, eval_pred)

        return {
            "rouge1": round(rouge_results["rouge1"], 4),
            "rouge2": round(rouge_results["rouge2"], 4),
            "rougeL": round(rouge_results["rougeL"], 4),
            "bleu": round(bleu_results["score"], 4)
        }

    def load_data(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.config.data_file)
            logger.info(f'Load data for training model successed! File data: {self.config.data_file}')
            return data 
        except Exception as e:
            logger.debug(f"Error: {e}")
            raise e
    
    def pre_data(self):
        dataset = PreDataset(self.load_data(), self.tokenizer)
        data_train, data_eval = random_split(dataset, self.config.split_data)
        return data_train, data_eval
    
    def train(self):

        if "MLFLOW_TRACKING_URI" not in os.environ:
            print("Running locally, initializing DagsHub...")
            dagshub.init(repo_owner=self.config.repo_owner, repo_name=self.config.repo_name, mlflow=True)
        else:
            print("Running in CI/CD, using existing environment variables.")
        
        exp_name = "Project_Tracking_Model_Text_Summary"
        print(f"Setting MLflow experiment to: {exp_name}")
        mlflow.set_experiment(exp_name)

        with mlflow.start_run():
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
            data_train, data_eval = self.pre_data()

            training_args = Seq2SeqTrainingArguments(
                output_dir="./results_summary",
                eval_strategy="epoch",
                learning_rate=self.config.learning_rate,
                logging_strategy="steps",
                logging_steps=10,
                per_device_train_batch_size=self.config.train_batch,   
                per_device_eval_batch_size=self.config.eval_batch,     
                weight_decay=self.config.weight_decay,
                num_train_epochs=self.config.epochs,               
                predict_with_generate=True,      
                fp16=True,
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

            logger.info("Evaluation one more test data")
            eval_results = trainer.evaluate()
            logger.info(f"Final result: {eval_results}")

            trainer.save_model(self.config.save_path)
            self.tokenizer.save_pretrained(self.config.save_path)
            print(f"Save model completed in local: {self.config.save_path}")

            print("Uploading model to MLFlow storage...")
            components = {
                "model": trainer.model,
                "tokenizer": self.tokenizer,
            }

            # Sử dụng mlflow.transformers thay vì mlflow.pytorch
            mlflow.transformers.log_model(
                transformers_model=components,
                artifact_path="model",                     # Tên thư mục chứa model trong Run
                registered_model_name=self.config.model_name, # Tên Model trong Registry 
                task="summarization",                      # Khai báo luôn task để sau này load bằng pipeline cực dễ
                pip_requirements="requirements.txt"        # Giữ nguyên ý tưởng của bạn để tiện deploy
            )

            print("Model registered to MLflow Registry successfully!")