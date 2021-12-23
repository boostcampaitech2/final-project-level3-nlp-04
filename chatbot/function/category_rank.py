import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from datasets import load_dataset
from collections import defaultdict

class RankReview():
    def __init__(self, subway="강남역") -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                      'data')
        self.reviews = self.get_ranked_stores(subway)

    def get_ranked_stores(self, subway):
        # data 불러오기
        dataset = load_dataset('samgin/FooReview')
        df = pd.DataFrame()
        df['restaurant_name'] = dataset['train']['restaurant']
        df['subway'] = dataset['train']['subway']
        df['food'] = dataset['train']['food']
        df['delvice'] = dataset['train']['delvice']
        df['category_name'] = dataset['train']['category_name']
        df.sort_values('restaurant_name', inplace=True)

        # df = pd.read_csv(os.path.join(self.data_path, 'FooReview/combined_211219dataset.csv'))
        subway_data = df[df.subway.str.contains(subway)]

        # list에는 전체 별점 평균, 리뷰 개수, 5점 개수, 3점 개수, 1점 개수
        classif_category = defaultdict(dict)

        category = ['1인분 주문', '프랜차이즈', '치킨', '피자/양식', '중국집', '한식', '일식/돈까스', '족발/보쌈', '야식', '분식', '카페/디저트']
        classif_category = {c: {} for c in category}

        old_resta = subway_data.iloc[0]
        info_list = [0 for _ in range(11)]
        food_stars = 0
        delvice_stars = 0
        review_cnt = 0

        for i in range(len(subway_data)):
            data = subway_data.iloc[i]
            resta_name = data.restaurant_name
            if resta_name != old_resta.restaurant_name:
                if old_resta['category_name'] is not None and review_cnt > 50:
                    info_list[0] = round(food_stars / review_cnt, 3) # 평균 평점 계산
                    info_list[1] = review_cnt   # 리뷰 개수
                    info_list[5] = round(delvice_stars / review_cnt, 3)  # 평균 평점 계산
                    info_list[9] = round((food_stars + delvice_stars) / 2 / review_cnt, 3)
                    for j in range(2, 5):   # 별점 5, 3, 1 분포
                        info_list[j] = round((info_list[j]/review_cnt) * 100, 1)
                    for j in range(6, 9):  # 별점 5, 3, 1 분포
                        info_list[j] = round((info_list[j] / review_cnt) * 100, 1)
                    # resta_stars[resta_name] = info_list

                    cats = old_resta.category_name
                    if ',' in cats:
                        tmp_cats = cats.split(',')
                        for c in tmp_cats:
                            classif_category[c.lstrip()][old_resta.restaurant_name] = info_list
                    else : 
                        classif_category[cats][old_resta.restaurant_name] = info_list

                old_resta = data
                food_stars = 0
                delvice_stars = 0
                review_cnt = 0
                info_list = [0 for _ in range(11)]
            
            if data.food == 5:
                info_list[2] += 1
            elif data.food == 3:
                info_list[3] += 1
            elif data.food == 1:
                info_list[4] += 1

            if data.food == 0:
                info_list[10] += 1

            if data.delvice == 5:
                info_list[6] += 1
            elif data.delvice == 3:
                info_list[7] += 1
            elif data.delvice == 1:
                info_list[8] += 1
            
            food_stars += data.food
            delvice_stars += data.delvice
            review_cnt += 1

        sorted_reviews = dict()
        for c in category:
            sorted_reviews[c] = sorted(classif_category[c].items(), key=lambda x:[x[1][9], x[1][1], x[1][0], x[1][5]],
                                       reverse=True)

        return sorted_reviews

    def get_by_category(self, category):
        return self.reviews[category]