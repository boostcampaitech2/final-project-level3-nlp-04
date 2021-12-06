import pymysql
import pandas as pd


class SqlHelper:
    def __init__(self, host, port, db_name, user, passwd):
        self.host = host
        self.port = port
        self.db_name = db_name
        self.user = user
        self.passwd = passwd

    def insert(self, query):
        conn = None

        try:
            # Open database connection
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                   passwd=self.passwd, db=self.db_name, charset='utf8',
                                   autocommit=True, cursorclass=pymysql.cursors.DictCursor)

            cursor = conn.cursor()
            cursor.execute(query)

        except Exception as e:
            print(e)
        finally:
            if conn is not None:
                # disconnect from server
                conn.close()

    def update(self, query):
        conn = None

        try:
            # Open database connection
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                   passwd=self.passwd, db=self.db_name, charset='utf8',
                                   autocommit=True, cursorclass=pymysql.cursors.DictCursor)

            cursor = conn.cursor()
            cursor.execute(query)

        except Exception as e:
            print(e)
        finally:
            if conn is not None:
                # disconnect from server
                conn.close()


    def get_df(self, query):
        data_frame = None
        conn = None

        try:
            # Open database connection
            conn = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                   passwd=self.passwd, db=self.db_name, charset='utf8',
                                   autocommit=True, cursorclass=pymysql.cursors.DictCursor)

            cursor = conn.cursor()
            cursor.execute(query)

            result = cursor.fetchall()
            data_frame = pd.DataFrame(result)
            data_frame.columns = map(str.lower, data_frame.columns)
        except Exception as e:
            print(e)
        finally:
            if conn is not None:
                # disconnect from server
                conn.close()

        return data_frame
