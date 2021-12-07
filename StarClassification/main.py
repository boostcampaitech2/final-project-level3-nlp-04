from transformers import AutoTokenizer, AutoModel
import transformers

model = AutoModel.from_pretrained("klue/roberta-large")
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

