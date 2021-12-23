import os
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from kss import split_sentences

def gen_rev(restaurant, menu, food, delvice):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_folder, 'ReviewGeneration/checkpoint_ep5.pth')
    checkpoint = torch.load(model_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                        bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>', sep_token='<sep>')

    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2', pad_token_id=tokenizer.eos_token_id).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    start = f"음식점은 {restaurant} 메뉴는 {menu} 음식 점수는 {food} 배달 점수는 {delvice} 리뷰는"
    input_ids = tokenizer.encode(start)
    review_list = []

    gen_ids = model.generate(torch.tensor([input_ids]).to(device),
                             max_length=150,
                             repetition_penalty=2.0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True,
                             top_p=0.9,
                             top_k=100,
                             temperature=0.5,
                             do_sample=True,
                             num_return_sequences=4,
                             )
    for ids in gen_ids:
        generated = tokenizer.decode(ids.tolist())
        generated = generated[len(start):].replace('<unk>', '').replace('~~', '~')
        generated = ' '.join(split_sentences(generated, use_heuristic=True)[:-1])
        review_list.append(generated)

    return review_list