import os
import torch
import random
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# Function to load and process the document
def load_and_process_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create vector store
def create_vector_store(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return FAISS.from_texts(texts, embeddings)

# Function to set up language model
def setup_language_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=1000,
        max_new_tokens=150
    )

    return HuggingFacePipeline(pipeline=pipe)

# Function to set up QA system
def setup_qa_system(docsearch, local_llm):
    return RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=docsearch.as_retriever())

# Function to generate random farm-related data
def generate_farm_data():
    data = {
        "Reading #": range(1, 101),
        "Soil Moisture (%)": [round(random.uniform(10, 40), 2) for _ in range(100)],
        "Nitrogen Content (mg/kg)": [round(random.uniform(50, 300), 2) for _ in range(100)],
        "pH": [round(random.uniform(5.5, 8.5), 2) for _ in range(100)],
        "Temperature (°C)": [round(random.uniform(15, 35), 2) for _ in range(100)]
    }
    return pd.DataFrame(data)

# Main function for Streamlit app
def main():
    st.title("Retrieval-Augmented Generation (RAG) Farm Insights")

    # Tabs for QA and Data Generation
    tab1, tab2 = st.tabs(["Ask Questions", "Farm Data"])

    with tab1:
        st.header("Query the Knowledge Base")

        # Specify the file path for the document
        file_path = os.path.join(os.path.dirname(_file_), "info.txt")

        # Load and process the document
        if os.path.exists(file_path):
            texts = load_and_process_document(file_path)

            # Create vector store
            docsearch = create_vector_store(texts)

            # Set up language model
            local_llm = setup_language_model()

            # Set up QA system
            qa = setup_qa_system(docsearch, local_llm)

            # Query input
            question = st.text_input("Enter your question about farming:", "")

            if question:
                answer = qa.run(question)
                st.write(f"*Answer:* {answer}")
        else:
            st.error(f"File 'info.txt' not found in {os.path.dirname(_file_)}.")

    with tab2:
        st.header("Last 100 Farm Data Readings")

        # Generate and display random farm data
        data = generate_farm_data()
        st.dataframe(data)

if _name_ == "_main_":
    main()