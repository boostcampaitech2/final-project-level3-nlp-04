from PIL import Image
from urllib.request import urlopen
from chatbot.model.elastic_img.elastic_search import ElasticSearchRetrieval
from chatbot.model.elastic_img.utils_qa import get_args

def gen_img(text, topk=4):
    tmp_list = []

    data_args = get_args()

    retriever = ElasticSearchRetrieval(data_args)

    context_list, score_list, id_list, answer = retriever.elastic_retrieval(text, topk)
    print(context_list)
    for img_url in answer:
        url = str(img_url)
        # urllib.request.urlretrieve(url, "./test.jpg")
        try:
            img = Image.open(urlopen(url))
        except:
            img = Image.open(urlopen(url.split(" ")[0]))
        tmp_list.append(img)
    return tmp_list