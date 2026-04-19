import mlflow
from pyvi import ViTokenizer
import torch
import re

from src.pipeline.crawl_news import CrawlNews


class SummarizerPipeline:
    def __init__(self, model_name: str, device: str = None):
        self.device = device
        self.crawl = CrawlNews()

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

    def _clean_text(self, text: str) -> str:
        """Làm sạch văn bản: bỏ ký tự đặc biệt, chuẩn hoá khoảng trắng."""
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _build_input(self, crawl_result: dict) -> str:
        """
        Ghép các field từ kết quả crawl thành chuỗi input cho model.
        Format giống lúc training: title[SEP]description[SEP]content
        (khớp với cột 'Combined' được tạo bởi processed_data.py)
        """
        title       = self._clean_text(crawl_result.get('title', ''))
        description = self._clean_text(crawl_result.get('description', ''))
        content     = self._clean_text(crawl_result.get('content', ''))
        category    = self._clean_text(crawl_result.get('category', ''))

        combined = '[SEP]'.join(filter(None, [title, description, content, category]))
        return combined

    def predict(self, url: str, max_length: int = 128) -> str:
        crawl_result = self.crawl.crawl(url)
        if not crawl_result:
            raise ValueError(f"Không thể crawl dữ liệu từ URL: {url}")

        text = self._build_input(crawl_result)
        if not text:
            raise ValueError("Bài báo không có nội dung để tóm tắt.")

        segmented_text = ViTokenizer.tokenize(text)

        inputs = self.tokenizer(
            segmented_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],  
                max_length=max_length,
                num_beams=4,         
                length_penalty=2.0,  
                no_repeat_ngram_size=3, 
                early_stopping=True,
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        final_summary = summary.replace('_', ' ')
        return final_summary