import pandas as pd
import config as c
from core.sql_helper import SqlHelper
from core.log_helper import LogHelper

def make_pilot_star_tagging_data():
    review_df = pd.read_csv('./data/review.csv')

    star5 = review_df[review_df.star == 5].sample(50)
    star4 = review_df[review_df.star == 4].sample(50)
    star3 = review_df[review_df.star == 3].sample(50)
    star2 = review_df[review_df.star == 2].sample(50)
    star1 = review_df[review_df.star == 1].sample(50)

    pilot_df = pd.concat([star1, star2, star3, star4, star5])
    pilot_df = pilot_df.sample(len(pilot_df)).reset_index(drop=True)

    pilot_df.to_csv('./data/pilot_df.csv', index=False, encoding='utf8')


def make_star_tagging_data():
    sql_helper = SqlHelper(**c.DB_CONFIG)

    query = 'select * from review'

