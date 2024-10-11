# 📚 PDF Chat Assistant optimize for Vietnamese !

Welcome to the **PDF Chat Assistant**! This application allows you to upload PDF files and chat with them using the power of AI. Powered by **LangChain**, **FAISS**, and **Groq's LLM**, it brings your static documents to life by giving you answers to your questions based on their content!


## 🚀 Getting Started

### Prerequisites

To get started with the **PDF Chat Assistant**, you will need to have:

- **Python 3.8+**
- **Streamlit**
- **Hugging Face Transformers**
- **FAISS**
- **Groq API Token**

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MonsieurNam/ChatWF.git
   ```

2. **Install the Required Packages**

   Install all dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**

   Create a `.env` file in the root directory and add your **Groq API Token**:

   ```plaintext
   GROQ_API_TOKEN=your_groq_api_key_here
   ```
4. **Setup vectorsotre file**
    ```plaintext
   Run preprocess_storevector.py to make vectorstore.pkl 
   ```
### 🎯 How to Use

1. **Run the Application**

   Start the application using the following command:

   ```bash
   streamlit run app.py for original ver
   streamlit run newapp.py for fix ver with some edit
   streamlit run mix.py for static ver
   ```

2. **Upload PDFs**

   Use the document uploader on the right side of the page to upload your PDF files.

3. **Ask Questions**

   Once your documents are processed, you can start asking questions based on the content of your PDFs in the chat interface.

4. **Customize the AI**

   Modify the system prompt to change how the AI responds. For example, you can ask it to be more casual, technical, or creative!


## 🛠️ Project Architecture

### Key Components:

- **Streamlit**: Frontend interface and chat system.
- **LangChain**: Manages conversational logic and connects your queries with document content.
- **Hugging Face Embeddings**: Transforms document text into meaningful vector representations.
- **FAISS**: Efficiently stores and retrieves document vectors to match relevant content with user questions.
- **Groq API**: The powerhouse behind generating intelligent responses based on document data.

### Workflow

1. **Upload PDFs**: The user uploads PDFs that need to be processed.
2. **Text Processing**: Text is extracted and split into chunks for easier embedding.
3. **Vectorization**: The document chunks are embedded using Hugging Face models.
4. **Conversational Retrieval**: The AI searches through the document vectors to retrieve relevant information and answers questions.

## 🌟 Customization Options

- **System Prompt**: Customize the behavior of the AI by changing the system prompt.
- **Temperature**: Adjust the temperature to control the creativity of responses.

## Technologies Used:
- **Python**: Core programming language.
- **Streamlit**: For the web interface.
- **PyPDF2**: For extracting text from PDFs.
- **LangChain & FAISS**: For conversational AI and document vectorization.
- **Groq API**: Powers the conversational responses.

## 🐞 Troubleshooting & Error Handling

If you encounter any issues, don't worry! The app is equipped with robust error handling for:

- **PDF Processing Failures**: Receive user-friendly error messages if a document fails to process.
- **Embedding Issues**: Alerts you when there are issues creating embeddings for your document.
- **API Problems**: Notifies you if there’s a problem connecting to the Groq API.

## 🎨 Aesthetic Chat Interface

The chat interface is built for maximum readability and style:

- **User Messages**: Dark theme with light text.
- **AI Responses**: Highlighted in a contrasting color to distinguish between user and bot messages.
- **Smooth UI**: Responsive and interactive, designed for ease of use!

## 💡 Future Plans

- **Multi-Language Support**: Expanding the tool to support documents in multiple languages.
- **Improved Query Handling**: Adding support for more complex queries and contextual conversations.
- **Larger Document Handling**: Optimizing performance for bigger datasets and more complex documents.

## 📧 Feedback & Contributions

Feedback and contributions to improve this project are welcomed! Feel free to submit pull requests or open issues.

---

### 🏷️ License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

Enjoy chatting with your PDFs! 🚀

--- 
