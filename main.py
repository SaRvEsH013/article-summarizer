import os
import streamlit as st
import pickle
import time
import faiss
import langchain
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# Set the OpenAI API key
st.title("URL Document Question Answering")
llm=ChatOpenAI(
    temperature=0.0,
    model_name="openai/gpt-3.5-turbo",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=st.secrets["OPENAI_API_KEY"])

docs = []

if "url_count" not in st.session_state:
    st.session_state.url_count = 1
if "urls" not in st.session_state:
    st.session_state.urls = [""]

st.sidebar.title("Input URLs")

# Render all URL input fields
for i in range(st.session_state.url_count):
    st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1}", value=st.session_state.urls[i], key=f"url_{i}")

# Add another URL
if st.sidebar.button("Add another URL"):
    st.session_state.url_count += 1
    st.session_state.urls.append("")

# Process URLs
if st.sidebar.button("Process URLs"):
    url_list = [u.strip() for u in st.session_state.urls if u.strip()]
    if url_list:
        st.write(f"Processing {len(url_list)} URLs...")

        loaders =  UnstructuredURLLoader(url_list).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(loaders)

        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-mpnet-base-v2')
        vectors = encoder.encode([doc.page_content for doc in docs])

        #print docs
        st.write(f"Loaded {len(docs)} documents from URLs.")


        with open('vectors.pkl', 'wb') as f:
            pickle.dump(vectors, f)

        st.write("Done")
    else:
        st.warning("Please enter at least one valid URL.")

# take input query
query = st.text_input("Enter your question about the documents:")


# Load vectors and create vector store
if st.button("Ask Question"):
    if query:
        loaded_vectors = pickle.load(open('vectors.pkl', 'rb'))
        index = faiss.IndexFlatL2(loaded_vectors.shape[1])
        index.add(loaded_vectors)

        encoder = SentenceTransformer('all-mpnet-base-v2')
        query_vector = encoder.encode([query])
        _, indices = index.search(query_vector, k=3)

        # st.write(f"Query: {query}")
        # st.write(f"Indices of retrieved documents: {indices[0]}")
        # len of st.session_state.urls should match the number of documents

        url_list = [u.strip() for u in st.session_state.urls if u.strip()]
        if url_list:
            loaders =  UnstructuredURLLoader(url_list).load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(loaders)

        retrieved_docs = [docs[i] for i in indices[0]]
        # st.write(f"Retrieved documents: {retrieved_docs}")

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        )

        chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )

        response = chain.run(
            context="\n\n".join([doc.page_content for doc in retrieved_docs]),
            question=query
        )

        st.write("Response:")
        st.write(response)


        


        



