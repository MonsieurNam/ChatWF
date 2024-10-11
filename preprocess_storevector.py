import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,        # Điều chỉnh chunk_size nếu cần
        chunk_overlap=400,      # Điều chỉnh chunk_overlap nếu cần
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def save_vectorstore(vectorstore, path="vectorstore.pkl"):
    with open(path, "wb") as f:
        pickle.dump(vectorstore, f)

def main():
    pdf_path = "./data/data_content.pdf"  
    raw_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    save_vectorstore(vectorstore)
    print("Vectorstore has been saved successfully.")

if __name__ == '__main__':
    main()
