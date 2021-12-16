import sys
sys.path.append('/opt/ml/final-project-level3-nlp-04')

from core.sql_helper import SqlHelper
from config import DB_CONFIG

from collections import defaultdict

class RankReview():
    def __init__(self, subway="강남역") -> None:
        self.reviews = self.get_ranked_stores(subway)
        pass

    def get_ranked_stores(self, subway):
        # data 불러오기
        db = SqlHelper(**DB_CONFIG)
        subway_data = db.get_df(f'Select category_name, restaurant_name, total_star from review where subway = "2호선 강남역"')

        # list에는 전체 별점 평균, 리뷰 개수, 5점 개수, 3점 개수, 1점 개수
        classif_category = defaultdict(dict)

        category = ['1인분 주문', '프랜차이즈', '치킨', '피자/양식', '중국집', '한식', '일식/돈까스', '족발/보쌈', '야식', '분식', '카페/디저트']
        for c in category:
            classif_category[c] = {}

        old_resta = subway_data.iloc[0]
        info_list = [0 for _ in range(5)]
        tot_stars = 0
        review_cnt = 0

        for i in range(len(subway_data)):
            data = subway_data.iloc[i]
            resta_name = data.restaurant_name
            if resta_name != old_resta.restaurant_name:
                if old_resta['category_name'] is not None and review_cnt > 50:
                    info_list[0] = round(tot_stars / review_cnt, 3) # 평균 평점 계산
                    info_list[1] = review_cnt   # 리뷰 개수
                    for i in range(2, 5):   # 별점 5, 3, 1 분포
                        info_list[i] = round((info_list[i]/review_cnt) * 100, 1)
                    # resta_stars[resta_name] = info_list

                    cats = old_resta.category_name
                    if ',' in cats:
                        tmp_cats = cats.split(',')
                        for c in tmp_cats:
                            classif_category[c.lstrip()][old_resta.restaurant_name] = info_list
        #                    print(c.lstrip(), old_resta.restaurant_name, classif_category[c.lstrip()][old_resta.restaurant_name])
                    else : #and resta_name not in classif_category[cats]:
                        classif_category[cats][old_resta.restaurant_name] = info_list
        #                 print(cats, old_resta.restaurant_name, classif_category[cats][old_resta.restaurant_name])

                old_resta = data
                tot_stars = 0
                review_cnt = 0
                info_list = [0 for _ in range(5)]
            
            if data.total_star == 5:
                info_list[2] += 1
            elif data.total_star == 3:
                info_list[3] += 1
            elif data.total_star == 1:
                info_list[4] += 1
            
            tot_stars += data.total_star
            review_cnt += 1

        sorted_reviews = dict()
        for c in category:
            sorted_reviews[c] = sorted(classif_category[c].items(), key=lambda x:x[1], reverse=True)

        return sorted_reviews

    def get_by_category(self, category):
        return self.reviews[category]