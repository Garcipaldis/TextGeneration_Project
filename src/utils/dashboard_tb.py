import streamlit as st
import pandas as pd
from PIL import Image
import os, sys
import requests

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

from src.utils.mining_data_tb import Preprocessor
from src.utils.visualization_tb import Visualizer
from src.utils.apis_tb import FlaskFuncs

dfpath = root + os.sep + 'data' + os.sep + 'BASE.csv'
df = pd.read_csv(dfpath)
settings_file = root + os.sep + 'src' + os.sep + "utils" + os.sep + "settings_sql.json"

modelpath = root + os.sep + 'models'


class StreamFuncs(Visualizer, FlaskFuncs, Preprocessor):

    def __init__(self, df, root, settings_file):
        # get self.text_in_words
        Preprocessor.__init__(self, df)
        self.preprocess(option='word', mode='base')
        # preprocess in character-level
        FlaskFuncs.__init__(self, df, root, settings_file)
        self.root = root
        self.settings_file = settings_file

    def greet(self):
        """Method returning a streamlit text template.
        """
        st.title('Text Generation Machine Learning Project (GAN-Dalf)')
        st.subheader('Project made by: Jorge García Navarro')
        image = Image.open(self.root + os.sep + 'resources' + os.sep + 'writing.jpg')
        st.image(image)
        st.write("""Text Generation is currently one of the most challenging fields in the Artificial Intelligence spectrum. 
                    The purpose of this project is to generate memorable movie quotes by the use of Natural Language Processing tecniques, 
                    Long-Short-Term Memory neural networks and Generative Adversarial Networks. 
                    The results of this investigation serves as an introduction to the complexity of the matter at hand, 
                    revealing interesting new paths for future projects.""")
        st.subheader('Memorable Quote Example')
        st.write("""- *If by my life or death I can protect you, I will. You have my sword.*""")
        st.write('Title: The Fellowship of the Ring')

    def barchart_page(self):
        """ Returns word stats barchart on a template.
        """
        st.title('Text Generation Machine Learning Project (GAN-Dalf)')
        st.subheader('Corpus Word Stats')
        top = st.sidebar.select_slider("Top Values",
                                        options=range(3, 21),
                                        value=10)
        horizontal = st.sidebar.checkbox('Horizontal')
        if horizontal:
            st.pyplot(self.plot_word_barchart(top=top))
        else:
            st.pyplot(self.plot_word_barchart(x='word', y='count',top=top, sort=1, show_values=True))

    def flask_page(self):
        """ Connects to the Flask API and returns the response json on a template.
        """
        st.title('Text Generation Machine Learning Project (GAN-Dalf)')
        st.subheader('Base Dataframe from API Flask')
        r = requests.get(url="http://localhost:6060/info?token_id=B53814652")
        response = r.json()
        show_json = st.sidebar.checkbox('Show returned Json')
        df = pd.DataFrame(response).dropna()
        df.drop('Unnamed: 0', axis=1, inplace=True)
        if show_json:
            st.write(response)
        else:
            st.dataframe(df)

    def model_page(self):
        """Infrastructure for model prediction."""
        st.title('Text Generation Machine Learning Project (GAN-Dalf)')
        st.subheader('Character-level LSTM Text Generation Model')
        quote_length = st.sidebar.select_slider('Quote Length:', options=range(1, 501), value=40)
        temperature = st.sidebar.select_slider('Temperature:', options=range(2, 11), value=3)
        temp = temperature/10
        st.subheader('Type input text. Leave blank for random')
        sentence = st.text_input('Input your sentence here (min 40 characters):', value='')
        st.subheader('Prediction')
        if sentence == '':
            st.write(self.predict(quote_len=quote_length, temperature=temp))
        else:
            try:
                st.write(self.predict(sentence=sentence, quote_len=quote_length, temperature=temp))
            except:
                st.write('Invalid sequence length')

    def sql_page(self):
        """Brings the model_comparison table from MySQL server."""
        st.title('Text Generation Machine Learning Project (GAN-Dalf)')
        st.subheader('SQL Model Scores Table')
        loaded_json = self.read_json(self.settings_file)

        self.IP_DNS = loaded_json["IP_DNS"]
        self.PORT = loaded_json["PORT"]
        self.USER = loaded_json["USER"]
        self.PASSWORD = loaded_json["PASSWORD"]
        self.BD_NAME = loaded_json["BD_NAME"]

        self.SQL_ALCHEMY = 'mysql+pymysql://' + self.USER + ':' + self.PASSWORD + '@' + self.IP_DNS + ':' + str(self.PORT) + '/' + self.BD_NAME

        self.connect()
        scores_df = pd.DataFrame(self.execute_get_sql('SELECT * FROM model_comparison'))
        scores_df.columns = ['ID','Model', 'Parameters', 'Loss', 'RMSE', 'Accuracy']
        self.close()

        st.dataframe(scores_df)