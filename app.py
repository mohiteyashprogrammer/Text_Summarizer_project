from transformers import pipeline 
import streamlit as st
from PIL import Image 
from utils import set_background

set_background("C:\\Users\\yash mohite\\OneDrive\\Desktop\\Text_Summarizer_project\\OIP.jpg")

# tab name and favicon
# st.set_page_config(page_title='Text Summarizer', page_icon='📖', layout='centered')

# loading model from Hugging Face
model_name = "facebook/bart-large-cnn"

# You can also Use your trained model i use BART because i train on only 1 epoch you can train on higher epoch and more data
# summarizer = pipeline('summarization', model=model_name, framework="tf") 

# import pipeline
summarizer = pipeline('summarization', model=model_name, framework="tf")

st.write("""
# Text Summarizer 🎨 
Using Hugging Face Transformers 🤗
""")

with st.form(key='my_form'):
    input_text = st.text_area('Enter your Text', height=300)

    columns = st.columns(2)
    min_words = columns[0].number_input('Minimum words', value=30)
    max_words = columns[1].number_input('Maximum words', value=130)

    summarize_button = st.form_submit_button('Summarize!')

if summarize_button:
    summary = summarizer(input_text, max_length=max_words, min_length=min_words, do_sample=False)
    st.subheader('Result 🎉')
    st.info(summary[0]['summary_text'])
    st.write('**Length:** ' + str(len(summary[0]['summary_text'].split(' '))) + ' words')



