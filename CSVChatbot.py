import streamlit as st 
from pandasai.llm.openai import OpenAI
import os
import pandas as pd
import re
from pandasai import PandasAI
import matplotlib
matplotlib.use('TkAgg')
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']




openai_api_key = os.getenv("OPENAI_API_KEY")


def preprocess_text(text):
    # Convert the text to lowercase
    lowercased_text = text.lower()
    
    # Remove special characters and non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', lowercased_text)
    
    return cleaned_text


def is_relevant(query, ctext):
    # Preprocess the question for keyword matching
    processed_question = preprocess_text(query)

    # Split the question into individual words
    question_keywords = processed_question.split()

    # Check if any keywords are present in the PDF text
    for keyword in question_keywords:
        if keyword in ctext:
            return True
    
    return False


st.title("LLM Powered CSV Chatbot can perform Prompt-driven data analysis with PandasAI")
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])


# create an LLM by instantiating OpenAI object, and passing API token
llm = OpenAI(api_token=openai_api_key)

# create PandasAI object, passing the LLM
pandas_ai = PandasAI(llm)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    column_names = df.columns.tolist()  # Get column names as a list

    # Split and flatten the column names
    co = [item.lower() for name in column_names for item in re.split(r'[\s.]+', name)]
    additional_words=['identify','calculate','retrieve','find','locate','discover','uncover','determine','ascertain','extract','collect','gather','acquire','obtain','pinpoint','detect','search','investigate','scrutinize','summary','document','analyze','review','examine','inspect','assess','evaluate','survey','scan','review','study','look','explore','check','verify','cross reference','compare','contrast','match','differentiate','classify','categorize','catalog','list','summarize','synthesize','comprehend','interpret','decode','translate','compose','describe','explain']
    combined_list = co + additional_words

    st.write(df.head(5))
    prompt = st.text_area("Enter your prompt:")

    # Generate output
    if st.button("Generate"):
        if is_relevant(prompt, combined_list):
            
            if prompt:
                # call pandas_ai.run(), passing dataframe and prompt
                with st.spinner("Generating response..."):
                    st.write(pandas_ai.run(df, prompt))
            else:
                st.warning("Please enter a prompt.")

        else:
            st.write('The information is not present in provided document.')