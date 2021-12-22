import re
import pandas as pd
from tqdm import tqdm
from glob import glob


DATA_PATH = './data/dialect'
TRAIN_PATH = 'train'
VALID_PATH = 'valid'
filter_file = '*.txt'
match_pattern_1 = r'[1-9]:\s.*\(.+\)\/\(.+\).*'
match_pattern_3 = '[1-9]\t.*\(.+\)\/\(.+\).*'
remove_pattern = r'\( | \)'

def word_strip(target_word):
    word = target_word.lstrip('(')
    word = word.rstrip(')')
    return word

def get_sentence(target_file_list, file_name, standard_sentence_list, dialect_sentence_list, num):
    with open(file_name, 'r') as f:
        full_sentence_list = f.readlines()
        for target_sentece in full_sentence_list:
            standard_sentence = ''
            dialect_sentence = ''
            if num != 3:
                target = re.findall(match_pattern_1, target_sentece)
            else:
                target = re.findall(match_pattern_3, target_sentece)
            try:
                if target != []:
                    if num != 3:
                        filtered_target_sentence = target[0].split(': ')[1]
                    else:
                        filtered_target_sentence = target[0].split('\t')[1]
                    split_sentence_list = filtered_target_sentence.split()
                    for word in split_sentence_list:
                        if '/' in word:
                            dialect_word, standard_word = word.split('/')
                            dialect_word = word_strip(dialect_word)
                            standard_word = word_strip(standard_word)
                            dialect_sentence += ' ' + dialect_word
                            standard_sentence += ' ' + standard_word
                        else:
                            dialect_sentence += ' ' + word
                            standard_sentence += ' ' + word

                    dialect_sentence_list.append(dialect_sentence)
                    standard_sentence_list.append(standard_sentence)
                    target_file_list.append(file_name)

            except:
                continue

    return dialect_sentence_list, standard_sentence_list,  target_file_list

def filtering_and_save(file_list, dialect_sentence_list,standard_sentence_list, save_path):
    dialect_dict = {'id': file_list, 'sentence': dialect_sentence_list}
    standard_dict = {'id': file_list, 'sentence': standard_sentence_list}
    df_dialect = pd.DataFrame(dialect_dict)
    df_standard = pd.DataFrame(standard_dict)
    df_dialect['sentence'] = df_dialect['sentence'].map(lambda x: x.replace('(', "").replace(')', ""))
    df_standard['sentence'] = df_standard['sentence'].map(lambda x: x.replace('(', "").replace(')', ""))
    df = pd.DataFrame({'id': df_standard.id, 'standard': df_standard.sentence, 'dialect': df_dialect.sentence})
    temp = (df['standard'].map(lambda x: len(x.split())) == df['dialect'].map(lambda x: len(x.split())))
    df = df.loc[temp]
    temp2 = (df['standard'] != df['dialect'])
    df = df.loc[temp2]
    df = df.reset_index()
    df = df.drop(['index', 'id'], axis=1)
    new_df_standard = df[['standard']].rename(columns={'standard' : 'sentence'})
    new_df_dialect = df[['dialect']].rename(columns={'dialect' : 'sentence'})
    new_df_dialect.to_csv('/'.join([save_path, 'dialect.csv']), index=False, encoding='utf-8')
    new_df_standard.to_csv('/'.join([save_path, 'standard.csv']), index=False, encoding='utf-8')

def main():
    for i in range(1, 6):
        train_dir = '/'.join([DATA_PATH, str(i), TRAIN_PATH])
        valid_dir = '/'.join([DATA_PATH, str(i), VALID_PATH])
        train_text_list = glob('/'.join([train_dir, filter_file]))
        valid_text_list = glob('/'.join([valid_dir, filter_file]))
        train_dialect_sentence_list = []
        train_standard_sentence_list = []
        valid_dialect_sentence_list = []
        valid_standard_sentence_list = []
        target_file_list = []


        print(f'start {i} train!!!!!')
        for target_file in tqdm(train_text_list):
            train_dialect_sentence_list, train_standard_sentence_list,  target_file_list = get_sentence(target_file_list, target_file, train_dialect_sentence_list, train_standard_sentence_list, i)
        filtering_and_save(target_file_list,train_dialect_sentence_list, train_standard_sentence_list, train_dir)

        target_file_list = []
        print(f'start {i} valid!!!!!')
        for target_file in tqdm(valid_text_list):
            valid_dialect_sentence_list, valid_standard_sentence_list, target_file_list = get_sentence(target_file_list, target_file, valid_dialect_sentence_list, valid_standard_sentence_list, i)

        filtering_and_save(target_file_list, valid_dialect_sentence_list, valid_standard_sentence_list, valid_dir)



if __name__ == '__main__':
    main()




