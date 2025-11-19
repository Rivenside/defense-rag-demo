# app.py
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from huggingface_hub import snapshot_download
import os

# Load the index
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},           # force CPU
    encode_kwargs={"normalize_embeddings": True},
    # ↓↓↓ THIS IS THE MAGIC LINE THAT FIXES THE META TENSOR ERROR ↓↓↓
    cache_folder="/tmp/hf_cache",             # forces proper download path on HF
)
if not os.path.exists("faiss_index"):
    with st.spinner("First launch — downloading real FAISS index from 50+ defense PDFs (30–90 seconds, happens once)..."):
        snapshot_download(
            repo_id="Rivenside/faiss-index",
            repo_type="dataset",
            local_dir="faiss_index",
            allow_patterns=["*.faiss", "*.pkl"],
        )
        st.success("Real defense-grade index loaded and cached!")
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True   # ← REQUIRED for real indexes
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Prompt template
template = """You are an expert aerospace/defense technical assistant.
Answer the question based ONLY on the following context. Cite sources when possible.

Context:
{context}

Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.set_page_config(page_title="Defense RAG Chatbot", page_icon="✈️")
st.title("✈️ Defense & Aerospace RAG Chatbot")
st.caption("Built with LangChain • FAISS • Groq • Streamlit | Nov 2025")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if question := st.chat_input("Ask a question about F-35, MIL-STDs, aircraft maintenance, etc."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})