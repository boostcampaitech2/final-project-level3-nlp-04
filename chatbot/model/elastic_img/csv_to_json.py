import json
import csv
# restaurant_name, menu, preprocessed_review_context, image_url

with open('./elastic_image.csv', 'r') as file_csv:
    dic = dict()
    reader = csv.DictReader(file_csv)
    for i, row in enumerate(reader):
        dic[i] = row
    with open('./elastic_image_2.json', 'w', encoding='UTF-8') as file_json:
        json.dump(dic, file_json, ensure_ascii=False)

