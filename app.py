
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

# Project document path
BASE_DIR = Path(__file__).parent
DOCUMENT_PATH = BASE_DIR / "documents" / "university_knowledge.txt"

st.set_page_config(
    page_title="RAG vs Fine-Tuning",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 RAG vs Fine-Tuning")
st.subheader("Practical Comparison for University Knowledge")

st.write(
    "Ask questions about university information "
    "and retrieve relevant answers using RAG."
)

@st.cache_resource
def load_rag_system():
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        knowledge = file.read()

    chunks = [
        text.strip()
        for text in knowledge.split("\n\n")
        if text.strip()
    ]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index, chunks

try:
    model, index, chunks = load_rag_system()

    question = st.text_input(
        "Enter your university-related question:"
    )

    if st.button("Search") and question.strip():

        question_embedding = model.encode([question])
        question_embedding = np.array(
            question_embedding
        ).astype("float32")

        distances, results = index.search(
            question_embedding, k=2
        )

        st.subheader("Retrieved Information")

        for result in results[0]:
            if result >= 0:
                st.info(chunks[result])

        st.caption(
            "This demo retrieves relevant document passages. "
            "It does not yet generate an LLM response."
        )

except FileNotFoundError:
    st.error(
        "University knowledge file not found. "
        "Check documents/university_knowledge.txt."
    )

st.divider()

st.subheader("Project Modules")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### RAG")
    st.write("Retrieves relevant information from university documents.")

with col2:
    st.markdown("### Fine-Tuning")
    st.write("Prepares a model to learn domain-specific information.")
