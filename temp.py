from core.sql_helper import SqlHelper
import config as c


sql_helper = SqlHelper(**c.DB_CONFIG)

query = "select restaurant_name, menu, preprocessed_review_context, image_url from preprocessed_review where image_url != '-1'"

df = sql_helper.get_df(query)
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv("./data/elastic_image.csv", index=False)

print(len(df))