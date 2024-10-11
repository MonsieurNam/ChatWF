import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from htmlTemplates import css, bot_template, user_template
from groq import Groq
import os
from typing import Any, List, Mapping, Optional
from pydantic import BaseModel, Field
import time
import pickle

# Load environment variables
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_TOKEN")

class GroqWrapper(LLM, BaseModel):
    client: Groq = Field(default_factory=lambda: Groq(api_key=groq_api_key))
    model_name: str = Field(default="mixtral-8x7b-32768")
    system_prompt: str = Field(default=
        "Bạn là trợ lý chuyên về lịch sử Việt Nam. \
        Bạn cung cấp câu trả lời chính xác và chi tiết dựa trên nội dung tài liệu được cung cấp. \
        Nếu không biết, bạn sẽ trả lời Tôi không biết. \
        Bạn không tạo ra thông tin sai lệch.\
        Bạn luôn trả lời tôi bằng tiếng Việt."
    )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,  # Giảm temperature để tăng độ chính xác
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=stop,
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Error during completion: {e}")
            return "Đã xảy ra lỗi khi tạo phản hồi."

    @property
    def _llm_type(self) -> str:
        return "groq"

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "system_prompt": self.system_prompt}

# Hàm tải vectorstore đã lưu
def load_vectorstore(path="vectorstore.pkl"):
    with open(path, "rb") as f:
        vectorstore = pickle.load(f)
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

    # Thêm tin nhắn mới vào danh sách
    new_messages = []
    for message in st.session_state.chat_history[-2:]:  # Lấy hai tin nhắn cuối
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

    # Khởi tạo session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("📚 Chat với Tài liệu PDF của bạn")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 System Prompt")
        system_prompt = st.text_area(
            "Tùy chỉnh hành vi AI:",
            (
                "Bạn là trợ lý chuyên về lịch sử Việt Nam. \
                Bạn cung cấp câu trả lời chính xác và chi tiết dựa trên nội dung tài liệu được cung cấp. \
                Nếu không biết, bạn sẽ trả lời Tôi không biết. \
                Bạn không tạo ra thông tin sai lệch và luôn trả lời bằng tiếng Việt."
            ),
            height=150
        )

        # Kiểm tra xem vectorstore đã được tải chưa
        if st.session_state.vectorstore is None:
            if os.path.exists("vectorstore.pkl"):
                with st.spinner("Đang tải vectorstore..."):
                    st.session_state.vectorstore = load_vectorstore()
                    st.success("Vectorstore đã được tải thành công.")
            else:
                st.error("Vectorstore không tồn tại. Vui lòng chạy script tiền xử lý để tạo vectorstore.")

        if st.button("🧹 Xóa lịch sử trò chuyện"):
            clear_chat_history()
            st.experimental_rerun()

    with col1:
        st.subheader("💬 Giao diện Trò chuyện")
        user_question = st.text_input("Đặt câu hỏi về tài liệu của bạn:", key="user_input")
        
        if user_question:
            if st.session_state.vectorstore and st.session_state.vectorstore is not None:
                if st.session_state.conversation is None:
                    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, system_prompt)
                with st.spinner("AI đang suy nghĩ..."):
                    handle_userinput(user_question)
            else:
                st.warning("Vectorstore chưa được tải. Vui lòng kiểm tra lại.")

        # Hiển thị tin nhắn trò chuyện
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
