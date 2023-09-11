import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Define a confidence threshold for the LLM model
CONFIDENCE_THRESHOLD = 0.8  # You can adjust this threshold as needed
 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    
 
load_dotenv()


def preprocess_text(text):
    # Convert the text to lowercase
    lowercased_text = text.lower()
    
    # Remove special characters and non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', lowercased_text)
    
    return cleaned_text


def is_relevant(query, pdf_text):
    # Preprocess the question for keyword matching
    processed_question = preprocess_text(query)

    # Split the question into individual words
    question_keywords = processed_question.split()

    # Check if any keywords are present in the PDF text
    for keyword in question_keywords:
        if keyword in pdf_text:
            return True
    
    return False


def is_response_relevant(response, pdf_content):
    # Preprocess the response and PDF content for consistency
    processed_response = preprocess_text(response)
    processed_pdf_content = preprocess_text(pdf_content)

    # Check if any keywords from the response are present in the PDF content
    response_keywords = processed_response.split()
    for keyword in response_keywords:
        if keyword in processed_pdf_content:
            return True
    
    return False

 
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)

        if query:
            if is_relevant(query, text):
                docs = VectorStore.similarity_search(query=query,k=3)
                llm = OpenAI(model_name='gpt-3.5-turbo')
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                #response = chain.run(input_documents=docs, question=query)


                # Construct a prompt that includes the context of the PDF
                prompt = f"You are a PDF Chatbot, please provide information or answer of those questions that is strongly related to the content of the PDF document. If the query is strongly related to the context of the PDF, provide a relevant response. If the query is not related or weakly related and out of context with respect to the PDF content, respond with 'I don't know., {query}"
                response = chain.run(input_documents=docs, question=prompt)


                st.write(response)
            else:
                st.write('The information is not present in the PDF.')


 
        #if query:
            #docs = VectorStore.similarity_search(query=query,k=3)
            #llm = OpenAI()
            #chain = load_qa_chain(llm=llm, chain_type="stuff")
            #with get_openai_callback() as cb:
                #response = chain.run(input_documents=docs, question=query)
                #print(cb)
            #st.write(response)
 
if __name__ == '__main__':
    main()