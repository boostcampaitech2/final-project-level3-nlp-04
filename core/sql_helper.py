import pymysql
from sqlalchemy import *
import pandas as pd
from collections import OrderedDict
from core.log_helper import LogHelper

pymysql.install_as_MySQLdb()

class SqlHelper:
    def __init__(self, host, port, db_name, user, passwd):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.user = user
        self.passwd = passwd

    def insert(self, df, table_name='review'):
        conn = None

        try:
            # Open database connection
            engine = create_engine(f'mysql://{self.user}:{self.passwd}@{self.host}/{self.db_name}?charset=utf8mb4')

            conn = engine.connect()

            df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

        except Exception as e:
            LogHelper().i(e)
        finally:
            if conn is not None:
                # disconnect from server
                conn.close()

    def get_table_name(self):
        result = None
        conn = None

        try:
            # Open database connection
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                   passwd=self.passwd, db=self.db_name, charset='utf8mb4',
                                   autocommit=True, cursorclass=pymysql.cursors.DictCursor)

            cursor = conn.cursor()
            cursor.execute('show tables')

            result = [table['Tables_in_review'] for table in cursor.fetchall()]

        except Exception as e:
            LogHelper().i(e)
        finally:
            if conn is not None:
                # disconnect from server
                conn.close()

        return result

    def get_df(self, query):
        data_frame = None
        conn = None

        try:
            # Open database connection
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                   passwd=self.passwd, db=self.db_name, charset='utf8mb4',
                                   autocommit=True, cursorclass=pymysql.cursors.DictCursor)

            cursor = conn.cursor()
            cursor.execute(query)

            result = cursor.fetchall()
            result = map(OrderedDict, result)
            data_frame = pd.DataFrame(result)
            data_frame.columns = map(str.lower, data_frame.columns)

        except Exception as e:
            LogHelper().i(e)
        finally:
            if conn is not None:
                # disconnect from server
                conn.close()

        return data_frame
