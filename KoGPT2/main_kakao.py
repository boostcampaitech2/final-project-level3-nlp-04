import torch
from torch.utils.data import DataLoader # 데이터로더
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
from kogpt2.data import Read_Dataset
import gluonnlp
from kogpt2.model.sample import sample_sequence
from tqdm import tqdm
import subprocess
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100,
					help="epoch 를 통해서 학습 범위를 조절합니다.")
parser.add_argument('--save_path', type=str, default='',
					help="학습 결과를 저장하는 경로입니다.")
parser.add_argument('--load_path', type=str, default='./checkpoint/Alls/KoGPT2_checkpoint_296000.tar', #
					help="학습된 결과를 불러오는 경로입니다.")
parser.add_argument('--samples', type=str, default="samples/",
					help="생성 결과를 저장할 경로입니다.")
parser.add_argument('--data_file_path', type=str, default='./review.csv',
					help="학습할 데이터를 불러오는 경로입니다.")
parser.add_argument('--batch_size', type=int, default=32,
					help="batch_size 를 지정합니다.")
args = parser.parse_args()

pytorch_kogpt2 = {
	'url':
		'checkpoint/pytorch_kogpt2_676e9bcfa7.params',
	'fname': 'pytorch_kogpt2_676e9bcfa7.params',
	'chksum': '676e9bcfa7'
}

kogpt2_config = {
	"initializer_range": 0.02,
	"layer_norm_epsilon": 1e-05,
	"n_ctx": 1024,
	"n_embd": 768,
	"n_head": 12,
	"n_layer": 12,
	"n_positions": 1024,
	"vocab_size": 50000
}

def auto_enter(text):
	text = (text.replace("   ", "\n"))
	text = text.split("\n")

	text = [t.lstrip() for t in text if t != '']
	return "\n\n".join(text)

def get_gpu_memory_map():
	"""Get the current gpu usage.

    Returns
    -------
    usage: dict
       Keys are device ids as integers.
       Values are memory usage as integers in MB.
    """
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		], encoding='utf-8')
	# Convert lines into a dictionary
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map

def main(epoch, save_path, load_path, samples, data_file_path, batch_size):
	ctx = 'cuda'
	cachedir = '~/kogpt2/'

	import torch
	from transformers import AutoTokenizer, AutoModelForCausalLM
	tok = AutoTokenizer.from_pretrained(
		'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
		bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token=None, sep_token='<sep>'
	)
	model = AutoModelForCausalLM.from_pretrained(
		'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
		pad_token_id=tok.eos_token_id,
		torch_dtype='auto', low_cpu_mem_usage=True
	).to(device='cuda', non_blocking=True)

	vocab = tok.get_vocab()
	vocab = gluonnlp.vocab.BERTVocab(vocab,
									 mask_token=None,
									 sep_token='<sep>',
									 cls_token=None,
									 unknown_token='<unk>',
									 padding_token='<pad>',
									 bos_token='<s>',
									 eos_token='</s>')

	model.train()
	count = 0

	dataset = Read_Dataset(data_file_path, vocab, tok)
	print("Read_Dataset ok")
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	learning_rate = 3e-5
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	print('KoGPT-2 Transfer Learning Start')
	avg_loss = (0.0, 0.0)
	sents = []

	word = "음식점은 롯데리아, 메뉴는 불고기버거, 음식은 5점, 서비스와 배달은 1점<sep>"
	print(args)
	for epoch in range(1, epoch+1):
		pbar = tqdm(data_loader)
		for idx, data in enumerate(pbar):
			optimizer.zero_grad()

			data = torch.stack(data[0]) # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
			data = data.transpose(1,0)
			data = data.to(ctx)
			# model = model.to(ctx)

			outputs = model(data, labels=data)
			loss, logits = outputs[:2]
			loss = loss.to(ctx)
			loss.backward()
			avg_loss = (avg_loss[0] * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)
			optimizer.step()
			pbar.set_description('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}'.format(epoch, count, loss, avg_loss[0] / avg_loss[1]))

		# generator 진행
		sent = sample_sequence(model, tok, vocab, sent=word, text_size=100, temperature=0.7, top_p=0.8, top_k=40)
		sent = sent.replace("<unused0>", "\n") # 비효율적이지만 엔터를 위해서 등장
		sent = auto_enter(sent)
		sent = sent.replace('<pad>', '')
		print(sent)

		sents.append(f'{epoch} : {sent}')

		# 모델 저장
		# try:
		if not epoch % 50:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss
			}, f'{save_path}/KoGPT2_checkpoint_{str(epoch)}.tar')
		# except:
		#    pass

	sents = '\n'.join(sents)
	f = open(samples + f'{word}_sample.txt', 'w', encoding="utf-8")
	f.write(sents)
	f.close()

if __name__ == "__main__":
	assert args.save_path
	args.save_path = os.path.join('checkpoint/', args.save_path)
	main(args.epoch, args.save_path, args.load_path, args.samples, args.data_file_path, args.batch_size)