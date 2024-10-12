import os
import pickle
from io import BytesIO
import fitz
import streamlit as st
from PyPDF2 import PdfReader
from langdetect import detect
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import Optional, List, Mapping, Any
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from groq import Groq
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever

load_dotenv()
groq_api_key = os.getenv("GROQ_API_TOKEN")

css = '''
<style>
.chat-message {
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.user .chat-message {
    background-color: #dcf8c6;
}
.bot .chat-message {
    background-color: #f1f0f0;
}
</style>
'''

user_template = '''
<div class="chat-message user">
    <div style="display: flex; align-items: center;">
        <img src="https://i.imgur.com/6ZQ1qTm.png" width="30" height="30" style="margin-right: 10px;">
        <div>{{MSG}}</div>
    </div>
</div>
'''

bot_template = '''
<div class="chat-message bot">
    <div style="display: flex; align-items: center;">
        <img src="https://i.imgur.com/nGF1K8f.png" width="30" height="30" style="margin-right: 10px;">
        <div>{{MSG}}</div>
    </div>
</div>
'''

# Define GroqWrapper class
class GroqWrapper(LLM, BaseModel):
    client: Groq = Field(default_factory=lambda: Groq(api_key=groq_api_key))
    model_name: str = Field(default="llama3-8b-8192")
    system_prompt: str = Field(default=(
        "Bạn là trợ lý AI chuyên về lịch sử Việt Nam và luôn luôn trả lời bằng tiếng Việt. "
        "Bạn cung cấp câu trả lời chính xác, chi tiết dựa trên nội dung tài liệu được cung cấp. "
        "Nếu không biết, bạn sẽ trả lời 'Tôi không biết'. "
        "Bạn không bao giờ trả lời bằng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác ngoài tiếng Việt. "
        "Hãy chỉ sử dụng tiếng Việt trong tất cả các phản hồi của bạn."
    ))

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # Build messages list including conversation history
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add conversation history from session_state (if any)
            if 'messages' in st.session_state and st.session_state.messages:
                for msg in st.session_state.messages[::-1]:  # From oldest to newest
                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    else:
                        messages.append({"role": "assistant", "content": msg["content"]})

            # Add user's new message
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Reduce temperature for less randomness
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=stop,
            )
            return completion.choices[0].message.content
        except Exception as e:
            st.error(f"Lỗi khi tạo phản hồi: {e}")
            return "Đã xảy ra lỗi khi tạo phản hồi."

    @property
    def _llm_type(self) -> str:
        return "groq"

    def get_num_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "system_prompt": self.system_prompt}

# Functions to process the PDF and create retriever
@st.cache_data
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=5000,        # Adjust chunk_size if needed
        chunk_overlap=500,      # Adjust chunk_overlap if needed
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Removed caching from this function
def create_documents(chunks):
    docs = [Document(page_content=chunk) for chunk in chunks]
    return docs

# Removed caching from this function
def get_bm25_retriever(docs):
    retriever = BM25Retriever.from_documents(docs)
    return retriever

# Function to create conversation chain
def get_conversation_chain(retriever):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    if not groq_api_key:
        raise ValueError("GROQ_API_TOKEN is not set in the environment variables")

    llm = GroqWrapper()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    modified_question = user_question

    response = st.session_state.conversation({'question': modified_question})
    st.session_state.chat_history = response['chat_history']

    ai_response = st.session_state.chat_history[-1].content

    try:
        language = detect(ai_response)
    except:
        language = 'unknown'

    if language != 'vi':
        st.warning("AI đã trả lời bằng ngôn ngữ khác. Đang yêu cầu AI trả lời lại bằng tiếng Việt...")
        modified_question = user_question + " Vui lòng trả lời bằng tiếng Việt."
        response = st.session_state.conversation({'question': modified_question})
        st.session_state.chat_history = response['chat_history']
        ai_response = st.session_state.chat_history[-1].content

    # Update message history
    st.session_state.messages.insert(0, {"role": "assistant", "content": ai_response})
    st.session_state.messages.insert(0, {"role": "user", "content": user_question})

def clear_chat_history():
    st.session_state.messages = []
    st.session_state.chat_history = []

# Functions related to PDF processing (keep as is)
def pymupdf_parse_page(pdf_path: str, page_number: int = 0) -> str:
    text = ""
    try:
        with fitz.open(pdf_path) as file:
            if page_number < 0 or page_number >= file.page_count:
                st.error(f"Số trang {page_number + 1} vượt quá phạm vi cho tệp '{pdf_path}'.")
                return ""
            page = file.load_page(page_number)
            text += page.get_text()
    except Exception as e:
        st.error(f"Lỗi mở tệp PDF '{pdf_path}': {e}")
        return ""
    text = text[:230000]
    return text

def pymupdf_render_page_as_image(pdf_path: str, page_number: int = 0, zoom: float = 1.5) -> bytes:
    try:
        with fitz.open(pdf_path) as doc:
            if page_number < 0 or page_number >= doc.page_count:
                st.error(f"Số trang {page_number + 1} vượt quá phạm vi cho tệp '{pdf_path}'.")
                return b""
            page = doc.load_page(page_number)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            return img_bytes
    except Exception as e:
        st.error(f"Lỗi chuyển đổi trang PDF '{pdf_path}': {e}")
        return b""

def parse_data_detail(file_path: str):
    sections = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('#')
                if len(parts) < 3:
                    st.warning(f"Bỏ qua dòng không hợp lệ trong data_detail.txt: {line}")
                    continue
                start_page = int(parts[0])
                end_page = int(parts[1])
                name = parts[2].strip()
                sections.append({
                    'start': start_page,
                    'end': end_page,
                    'name': name
                })
    except FileNotFoundError:
        st.error(f"Không tìm thấy tệp '{file_path}'. Vui lòng đảm bảo nó tồn tại.")
    except Exception as e:
        st.error(f"Lỗi đọc tệp '{file_path}': {e}")
    return sections

def get_page_numbers(section):
    return list(range(section['start'], section['end'] + 1))

def initialize_session_state(total_pages=0):
    if 'total_pages' not in st.session_state:
        st.session_state.total_pages = total_pages

def main():
    st.set_page_config(page_title="Giáo dục Tiểu học Khóa 48-A2", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # Initialize session_state variables
    if 'retriever' not in st.session_state:
        # Process data_content.pdf
        pdf_path = "./data/data_content.pdf"  
        raw_text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(raw_text)
        docs = create_documents(text_chunks)
        retriever = get_bm25_retriever(docs)
        st.session_state.retriever = retriever

    if 'conversation' not in st.session_state:
        st.session_state.conversation = get_conversation_chain(st.session_state.retriever)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Continue displaying the user interface
    st.title("Giáo dục Tiểu học Khóa 48-A2")

    # Define data directory and path to data_detail.txt
    data_dir = "./data"
    data_detail_path = "./data_detail.txt"

    # Parse data_detail.txt to get sections
    sections = parse_data_detail(data_detail_path)

    if not sections:
        st.error("Không tìm thấy phần hợp lệ. Vui lòng kiểm tra tệp data_detail.txt của bạn.")
        return

    # Sidebar to select main section
    st.sidebar.header("Chọn Phần Chính")
    main_sections = [section for section in sections if section['name'].startswith("NHỮNG QUỐC GIA") or section['name'].startswith("XÂY DỰNG")]
    selected_main_section = st.sidebar.selectbox("Chọn một phần chính:", [section['name'] for section in main_sections])

    # Find the selected main section
    selected_main_section_details = next((s for s in sections if s['name'] == selected_main_section), None)

    # Handle sub-section selection if any
    if selected_main_section and selected_main_section.startswith("XÂY DỰNG"):
        st.sidebar.header("Chọn Phần Con")
        sub_sections = [section for section in sections if section['start'] >= selected_main_section_details['start'] and section['start'] <= selected_main_section_details['end']]
        selected_sub_section_name = st.sidebar.selectbox("Chọn một phần con:", [section['name'] for section in sub_sections])

        # Find details of the selected sub-section
        selected_sub_section = next((s for s in sections if s['name'] == selected_sub_section_name), None)

        if not selected_sub_section:
            st.error("Không tìm thấy phần đã chọn.")
            return

        # Create a list of page numbers for the selected section
        page_numbers = get_page_numbers(selected_sub_section)
    else:
        # If the main section is section I
        selected_sub_section = selected_main_section_details
        page_numbers = get_page_numbers(selected_main_section_details)

    total_pages = len(page_numbers)

    # Initialize session state
    initialize_session_state(total_pages=total_pages)

    # Display options
    st.sidebar.header("Tùy Chọn Hiển Thị")
    show_text = st.sidebar.checkbox("Hiển Thị Văn Bản Đã Trích Xuất", value=True)
    zoom_factor = st.sidebar.slider("Mức Thu Phóng", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    # Display main content
    st.header(f"📄 {selected_sub_section['name'] if selected_sub_section else selected_main_section}")

    for idx, page_num in enumerate(page_numbers, start=1):
        pdf_filename = f"page_{page_num:03}.pdf"
        pdf_path = os.path.join(data_dir, pdf_filename)

        if not os.path.isfile(pdf_path):
            st.error(f"Không tìm thấy tệp PDF '{pdf_filename}' trong '{data_dir}'.")
            continue

        try:
            img_bytes = pymupdf_render_page_as_image(pdf_path, page_number=0, zoom=zoom_factor)
            if img_bytes:
                st.image(img_bytes, caption=f"Trang {page_num}", use_column_width=True)
            else:
                st.error("Không thể hiển thị hình ảnh trang.")
        except Exception as e:
            st.error(f"Lỗi hiển thị trang PDF: {e}")

        st.markdown("---")

    # Sidebar: Page Information
    st.sidebar.header("Thông Tin Trang")

    for idx, page_num in enumerate(page_numbers, start=1):
        with st.sidebar.expander(f"📄 Trang {page_num}"):
            pdf_filename = f"page_{page_num:03}.pdf"
            pdf_path = os.path.join(data_dir, pdf_filename)

            if not os.path.isfile(pdf_path):
                st.error(f"Không tìm thấy tệp PDF '{pdf_filename}' trong '{data_dir}'.")
                continue

            page_text = pymupdf_parse_page(pdf_path)

            if show_text:
                st.text_area("Văn Bản Đã Trích Xuất:", value=page_text, height=150)

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                try:
                    with open(pdf_path, 'rb') as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        label="📥 Tải Xuống Trang PDF",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Lỗi chuẩn bị tải xuống PDF: {e}")

            if show_text:
                with download_col2:
                    try:
                        buffer = BytesIO()
                        buffer.write(page_text.encode('utf-8'))
                        buffer.seek(0)
                        safe_section_name = "".join(c for c in selected_sub_section['name'] if c.isalnum() or c in (' ', '_', '-')).rstrip()
                        download_filename = f"{safe_section_name}_Trang_{page_num}.txt"
                        st.download_button(
                            label="📄 Tải Xuống Văn Bản Đã Trích Xuất",
                            data=buffer,
                            file_name=download_filename,
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Lỗi chuẩn bị tải xuống văn bản: {e}")

    # Chat functionality in sidebar
    st.sidebar.header("💬 Chat với Tài liệu")
    user_question = st.sidebar.text_input("Đặt câu hỏi về tài liệu của bạn:", key="user_input")

    if user_question:
        handle_userinput(user_question)

    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.sidebar.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.sidebar.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

    if st.sidebar.button("🧹 Xóa lịch sử trò chuyện"):
        clear_chat_history()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
