import os
import torch
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.model.sample import sample_sequence
from kogpt2.utils import get_tokenizer
from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2Config, GPT2LMHeadModel
import gluonnlp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=0.7,
					help="temperature 를 통해서 글의 창의성을 조절합니다.")
parser.add_argument('--top_p', type=float, default=0.9,
					help="top_p 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--top_k', type=int, default=40,
					help="top_k 를 통해서 글의 표현 범위를 조절합니다.")
parser.add_argument('--text_size', type=int, default=250,
					help="결과물의 길이를 조정합니다.")
parser.add_argument('--loops', type=int, default=3,
					help="글을 몇 번 반복할지 지정합니다. 0은 무한반복입니다.")
parser.add_argument('--tmp_sent', type=str, default="메뉴는 도넛 별점을 5점인 리뷰를 만들어줘<sep>",
					help="글의 시작 문장입니다.")
parser.add_argument('--load_path', type=str, default="./checkpoint/way4/KoGPT2_checkpoint_50.tar",
					help="학습된 결과물을 저장하는 경로입니다.")

args = parser.parse_args()


def auto_enter(text):
	text = (text.replace("   ", "\n"))
	text = text.split("\n")

	text = [t.lstrip() for t in text if t != '']
	return "\n\n".join(text)


def main(temperature = 0.7, top_p = 0.8, top_k = 40, tmp_sent = "", text_size = 100, loops = 0, load_path = ""):
	ctx = 'cuda'

	# Device 설정
	device = torch.device(ctx)
	# 저장한 Checkpoint 불러오기
	checkpoint = torch.load(load_path, map_location=device)

	from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
	model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	tok = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
												  bos_token='<s>', eos_token='</s>', unk_token='<unk>',
												  pad_token='<pad>', mask_token='<mask>', sep_token='<sep>')
	vocab = tok.get_vocab()

	# # KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
	# kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
	# kogpt2model.load_state_dict(checkpoint['model_state_dict'])

	# kogpt2model.eval()
	vocab = gluonnlp.vocab.BERTVocab(vocab,
									 mask_token=None,
									 sep_token='<sep>',
									 cls_token=None,
									 unknown_token='<unk>',
									 padding_token='<pad>',
									 bos_token='<s>',
									 eos_token='</s>')

	# tok_path = get_tokenizer()
	# model, vocab = kogpt2model, vocab_b_obj
	# tok = SentencepieceTokenizer(tok_path)

	if loops:
		num = 1
	else:
		num = 0

	load_path = '/'.join(load_path.split("/")[:-1])

	print("ok : ",load_path)

	if not(os.path.isdir(load_path)):
		os.makedirs(os.path.join(load_path))

	if tmp_sent == "":
		tmp_sent = input('input : ')



	# "메뉴는 "메뉴" 별점은 "별점[SEP]리뷰
	way3 = [
		"메뉴는 찜 별점은 5점<sep>",
		"메뉴는 치킨 별점은 1점<sep>",
		"메뉴는 샤브샤브 별점은 5점<sep>",
		"메뉴는 간장게장 별점은 1점<sep>",
		"메뉴는 김밥 별점은 5점<sep>",
		"메뉴는 찌개 별점은 1점<sep>",
		"메뉴는 랍스타 별점은 5점<sep>",
		"메뉴는 버섯전골 별점은 1점<sep>",
	]

	# "메뉴는"메뉴"별점은"별점"인 리뷰를 만들어줘"[SEP]리뷰
	way4 = [
		"메뉴는 찜 별점은 5점인 리뷰를 만들어줘<sep>",
		"메뉴는 치킨 별점은 1점인 리뷰를 만들어줘<sep>",
		"메뉴는 샤브샤브 별점은 5점인 리뷰를 만들어줘<sep>",
		"메뉴는 간장게장 별점은 1점인 리뷰를 만들어줘<sep>",
		"메뉴는 김밥 별점은 5점인 리뷰를 만들어줘<sep>",
		"메뉴는 찌개 별점은 1점인 리뷰를 만들어줘<sep>",
		"메뉴는 랍스타 별점은 5점인 리뷰를 만들어줘<sep>",
		"메뉴는 버섯전골 별점은 1점인 리뷰를 만들어줘<sep>",
	]

	for idx, tmp_sent in enumerate(way3, 1):
		num = 1
		sents = []
		f = open(os.path.join(load_path, f'generated_texts{idx}.txt'), 'w', encoding="utf-8")

		while 1:
			sent = tmp_sent

			toked = tok.tokenize(sent)

			if len(toked) > 1022:
				break

			sent = sample_sequence(model, tok, vocab, sent, text_size, temperature, top_p, top_k)
			sent = sent.replace("//", "\n") # 비효율적이지만 엔터를 위해서 등장
			sent = sent.replace("</s>", "")
			sent = auto_enter(sent)
			sents.append(sent)

			head = [load_path, tmp_sent, sent.replace('<pad>', ''), str(text_size), str(temperature), str(top_p), str(top_k)]
			# print("head : ", head)
			# for h in head:
			# 	print(h)
			f.write(', '.join(head) + '\n')

			#tmp_sent = ""

			if num != 0:
				if num >= loops:
					print("good")
					break
				num += 1

		f.close()


if __name__ == "__main__":
	# execute only if run as a script
	main(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, tmp_sent=args.tmp_sent, text_size=args.text_size, loops=args.loops+1, load_path=args.load_path)