import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from htmlTemplates import css, bot_template, user_template
from groq import Groq
import os
from typing import Any, List, Mapping, Optional
from pydantic import BaseModel, Field
import time

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_TOKEN")

class GroqWrapper(LLM, BaseModel):
    client: Groq = Field(default_factory=lambda: Groq(api_key=groq_api_key))
    model_name: str = Field(default="mixtral-8x7b-32768")
    system_prompt: str = Field(default=    "Bạn là trợ lý chuyên về lịch sử Việt Nam. Bạn cung cấp câu trả lời chính xác và chi tiết "
    "dựa trên nội dung tài liệu được cung cấp. Nếu không biết, bạn sẽ trả lời 'Tôi không biết'. "
    "Bạn không tạo ra thông tin sai lệch và luôn trả lời bằng tiếng Việt.")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=stop,
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error during completion: {e}")
            return "Error occurred while generating response."

    @property
    def _llm_type(self) -> str:
        return "groq"

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "system_prompt": self.system_prompt}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, system_prompt):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    if not groq_api_key:
        raise ValueError("GROQ_API_TOKEN is not set in the environment variables")

    llm = GroqWrapper(system_prompt=system_prompt)
    st.success(f"Initialized GroqWrapper with model: {llm.model_name}")

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Add new messages to the beginning of the list
    new_messages = []
    for message in st.session_state.chat_history[-2:]:  # Get the last two messages (user question and AI response)
        if message.type == 'human':
            new_messages.append({"role": "user", "content": message.content})
        else:
            new_messages.append({"role": "assistant", "content": message.content})
    
    st.session_state.messages = new_messages + st.session_state.messages

def clear_chat_history():
    st.session_state.messages = []
    st.session_state.chat_history = []

def main():
    st.set_page_config(page_title="PDF Chat Assistant", page_icon="📚", layout="wide")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header("📚 Chat with Your PDFs")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("📄 Document Upload")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True, type="pdf")
        
        st.subheader("🔧 System Prompt")
        system_prompt = st.text_area(
            "Customize AI behavior:",
                "Bạn là trợ lý chuyên về lịch sử Việt Nam. Bạn cung cấp câu trả lời chính xác và chi tiết "
    "dựa trên nội dung tài liệu được cung cấp. Nếu không biết, bạn sẽ trả lời 'Tôi không biết'. "
    "Bạn không tạo ra thông tin sai lệch và luôn trả lời bằng tiếng Việt.",
            height=100
        )
        
        if st.button("🔍 Process Documents", type="primary"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore, system_prompt)
                    st.success("Documents processed successfully! You can now ask questions.")
            else:
                st.warning("Please upload PDF documents before processing.")

        if st.button("🧹 Clear Chat History"):
            clear_chat_history()
            st.rerun()

    with col1:
        st.subheader("💬 Chat Interface")
        user_question = st.text_input("Ask a question about your documents:", key="user_input")
        
        if user_question:
            if st.session_state.conversation:
                with st.spinner("AI is thinking..."):
                    handle_userinput(user_question)
            else:
                st.warning("Please upload and process documents before asking questions.")

        # Display chat messages with animation
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                with st.empty():
                    for j in range(len(message["content"]) + 1):
                        partial_message = message["content"][:j]
                        st.markdown(bot_template.replace("{{MSG}}", partial_message + "▌"), unsafe_allow_html=True)
                        time.sleep(0.01)
                    st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

if __name__ == '__main__':
    main()