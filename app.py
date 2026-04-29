import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
from streamlit.elements.widgets import button

nltk.download('stopwords')
nltk.download('punkt')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # remove non-alphanumeric
    text = [i for i in text if i.isalnum()]

    # remove stopwords & punctuation
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS spam classifier')

input_sms= st.text_input('Enter your message')

if st.button('predict'):
    # preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)
    # Display
    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")

