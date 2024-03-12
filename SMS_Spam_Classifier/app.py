import streamlit as st
import pickle

#lets load the saved vectorizer and naive bayes classifier
tfidf=pickle.load(open('vectorizer.pkl',"rb"))
model=pickle.load(open('model.pkl','rb'))

#transform_text function for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')

ps=PorterStemmer()

def transform_text(text):
    #convert words to lower case and tokenize the text
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    #removing special charecters by selection only aplhanumeric words
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #Removing stopwords and puntuation marks 
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #stemming the words
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

#streamlit code
st.title("SMS SPAM CLASSIFIER")
input_sms = st.text_area('Enter Message')

if st.button('Predict'):
    #preprocessing
    transformed_sms= transform_text(input_sms)

    #vectorize
    vector_input=tfidf.transform([transformed_sms])

    #predict
    result=model.predict(vector_input)

    #display
    if result==1:
        st.header('Spam')
    else:
        st.header('Not Spam')
        



