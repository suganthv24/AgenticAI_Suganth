import streamlit as st

from main import (
    load_dataset,
    build_vector_db,
    generate_response
)

st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Local RAG Chatbot with Ollama")

DATA_FILE = "cat-facts.txt"


@st.cache_resource
def initialize_rag():

    dataset = load_dataset(DATA_FILE)

    build_vector_db(dataset)

    return True


initialize_rag()

query = st.text_input(
    "Ask a question about the document"
)

if st.button("Search"):

    if query.strip():

        with st.spinner("Retrieving relevant information..."):

            answer, docs = generate_response(query)

        st.subheader("Answer")

        st.write(answer)

