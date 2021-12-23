# import pandas as pd
#
# data_path = "./data/" # customize data path
# yelp_neg_path = data_path + "/dialect/1_train_dialect.csv"
# df_neg = pd.read_csv(yelp_neg_path)
# yelp_neg_dataset = df_neg['sentence'].tolist()
# yelp_neg_dataset = '\n'.join(yelp_neg_dataset)
#
# yelp_pos_path = data_path + "/dialect/1_train_standard.csv"
# df_pos = pd.read_csv(yelp_pos_path)
# yelp_pos_dataset = df_pos['sentence'].tolist()
# yelp_pos_dataset = '\n'.join(yelp_pos_dataset)
#
# with open('./data/dialect/dialect.txt','w') as f:
#   f.writelines(yelp_neg_dataset)
#
# with open('./data/dialect/standard.txt', 'w') as f:
#   f.writelines(yelp_pos_dataset)

