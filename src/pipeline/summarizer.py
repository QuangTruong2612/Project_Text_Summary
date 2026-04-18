import mlflow
from pyvi import ViTokenizer
import torch
import re

class SummarizerPipeline:
    def __init__(self, model_name: str, device: str = None):
        self.device = device

        model_uri = f"models:/{model_name}@champion"
        print(f"Đang tải Champion Model từ {model_uri}...")

        try:
            components = mlflow.transformers.load_model(
                model_uri=model_uri,
                return_type="components"
            )
            self.tokenizer = components["tokenizer"]
            self.model = components["model"].to(self.device)

            self.model.eval() 
            print("Tải model thành công!")
            
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            raise e
    
    def predict(self, text: str, max_length: int = 128) -> str:
        if not text or len(text.strip()) == 0:
            return ""

        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        segmented_text = ViTokenizer.tokenize(text)

        inputs = self.tokenizer(
            segmented_text,
            return_tensors = 'pt',
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,          # Dùng Beam Search để câu văn mượt hơn
                length_penalty=2.0,   # Khuyến khích model tạo câu dài hơn một chút
                early_stopping=True
            )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_summary = summary.replace("_", " ")

            return final_summary