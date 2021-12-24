import os
import sys
import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration


def review_transfer(review_text):
    # model = BartForConditionalGeneration.from_pretrained('samgin/Style_transfer')
    model = BartForConditionalGeneration.from_pretrained('/opt/ml/final-project-level3-nlp-04/style_transfer')
    tokenizer = get_kobart_tokenizer()


    text = review_text

    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output
