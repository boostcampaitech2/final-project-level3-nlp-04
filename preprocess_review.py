import re
import config as c

from db.sql_helper import SqlHelper


def preprocess(text):
    text = re.sub('[^0-9ㄱ-ㅎㅏ-ㅣ가-힣^~!?., ]', "", text)  # 지정된 문자 말고 제외
    text = re.sub('\s', " ", text)  # 화이트스페이스 띄어쓰기로 변경
    text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣^~?!.,])\1{2,}', r'\1\1', text)  # 반복되는 문자 2개로 축소
    return text


def main():
    sql_helper = SqlHelper(c.HOST, c.PORT, c.DB_NAME, c.USER, c.PASSWD)

    query = "select * from test_preprocessed_review order by insert_time desc limit 1"

    preprocessed_review_df = sql_helper.get_df(query)

    if preprocessed_review_df.empty:
        query = "select * from test_review"
    else:
        insert_time = preprocessed_review_df.iloc[0].insert_time
        query = f"select * from test_review where insert_time > {insert_time}"

    review_df = sql_helper.get_df(query)

    if review_df is not None:
        review_df['preprocessed_review_context'] = review_df.review_context.apply(lambda x: preprocess(x))

        sql_helper.insert(review_df, table_name='test_preprocessed_review')  # TODO 크롤러 완성되면 table name 수정할 것

        print('DB 저장 완료!')


if __name__ == '__main__':
    main()
