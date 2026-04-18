
from pyvi import ViTokenizer
from torch.utils.data import Dataset

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