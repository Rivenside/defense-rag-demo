from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path

print("Loading PDFs from ./data ...")
docs = []
for pdf_path in Path("data").glob("*.pdf"):
    print(f"  → {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    docs.extend(loader.load())

print(f"Loaded {len(docs)} pages total")

print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")

print("Creating embeddings and FAISS index...")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("faiss_index")

print("DONE! Index saved to ./faiss_index")
print("You can now run: streamlit run streamlit_app.py")