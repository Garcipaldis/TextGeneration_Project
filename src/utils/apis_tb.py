import pandas as pd
import os, sys
from tensorflow import keras
from sqlalchemy import create_engine
import pymysql

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from src.utils.models import LSTM_Generator
from src.utils.folders_tb import Folders

settings_file = root + os.sep + 'src' + os.sep + "utils" + os.sep + "settings_sql.json"

class FlaskFuncs(LSTM_Generator, Folders):
    """Class designed to operate with the server.py API script."""

    def __init__(self, df, root, settings_file, selection='1_Base_Quote_LSTM.h5'):
        LSTM_Generator.__init__(self, df)
        self.root = root
        self.settings_file = settings_file

        self.load_model(self.root + os.sep + 'models' + os.sep + selection)

        loaded_json = self.read_json(self.settings_file)

        self.IP_DNS = loaded_json["IP_DNS"]
        self.PORT = loaded_json["PORT"]
        self.USER = loaded_json["USER"]
        self.PASSWORD = loaded_json["PASSWORD"]
        self.BD_NAME = loaded_json["BD_NAME"]

        self.SQL_ALCHEMY = 'mysql+pymysql://' + self.USER + ':' + self.PASSWORD + '@' + self.IP_DNS + ':' + str(self.PORT) + '/' + self.BD_NAME

    def insert_df_to_mysql(self, input_df=None, option=1, table_name=''):
        """Inserts dataframe into MySQL server.
            - Args:
                - input_df: Desired dataframe to insert as a table. If None, the class attribute dataframe will be inserted.
                - option: 1 for inserting class attribute dataframe and 2 for using the input_df.
                - table_name: required if option 2 is selected."""

        engine = create_engine(self.SQL_ALCHEMY)
        if option == 1:
            df = self.data
            df.to_sql('jorge_garcia_navarro', engine, index=False, if_exists='replace')
        elif option == 2:
            df = input_df
            df.to_sql(table_name, engine, index=False, if_exists='replace')

        return 'Dataframe correctly inserted into MySQL Server.'

    def give_json(self):
        """ Reads a csv file with pandas and returns a json.
        """
        return self.data.to_json()

    def connect(self):
        # Open database connection
        self.db = pymysql.connect(host=self.IP_DNS,
                                user=self.USER, 
                                password=self.PASSWORD, 
                                database=self.BD_NAME, 
                                port=self.PORT)
        # prepare a cursor object using cursor() method
        self.cursor = self.db.cursor()
        print("Connected to MySQL server [" + self.BD_NAME + "]")
        return self.db

    def close(self):
        # disconnect from server
        self.db.close()
        print("Close connection with MySQL server [" + self.BD_NAME + "]")

    def execute_get_sql(self, sql):
        """SELECT"""
        results = None
        print("Executing:\n", sql)
        try:
            # Execute the SQL command
            self.cursor.execute(sql)
            # Fetch all the rows in a list of lists.
            results = self.cursor.fetchall()
        except Exception as error:
            print(error)
            print ("Error: unable to fetch data")
        
        return results

if __name__ == '__main__':
    df = pd.read_csv(root + os.sep + 'data' + os.sep + 'BASE.csv', index_col=0)
    FF = FlaskFuncs(df, root, settings_file)
    string = 'Certainty of death. Small chance of success.'
    print(FF.get_predicction('Base_Quote_LSTM.h5', string, temperature=0.2))
    FF.insert_df_to_mysql()