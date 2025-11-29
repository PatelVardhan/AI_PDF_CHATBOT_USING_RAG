import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
import google.generativeai as genai
import tempfile
import os

# ---------------------------
# 1. Page Setup
# ---------------------------
st.set_page_config(page_title="Chat with Your PDF", layout="wide")
st.title("📘 Chat with Your PDF using RAG")

st.sidebar.header("Upload and Model Settings")

# ---------------------------
# 2. File Upload
# ---------------------------
uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])
use_gemini = st.sidebar.checkbox("Use Gemini model (requires API key)")
gemini_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # ---------------------------
    # 3. Load and Split PDF
    # ---------------------------
    st.sidebar.info("Processing document...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # ---------------------------
    # 4. Create Embeddings + Vector Store
    # ---------------------------
    st.sidebar.info("Creating knowledge base...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs, embedding_model)

    st.success(f"✅ Loaded {len(docs)} text chunks from your document.")

    # ---------------------------
    # 5. Choose Model
    # ---------------------------
    if use_gemini and gemini_key:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        model_type = "Gemini"
    else:
        model = pipeline("text2text-generation", model="google/flan-t5-base")
        model_type = "FLAN-T5"

    st.sidebar.success(f"Using {model_type} model.")

    # ---------------------------
    # 6. Chat Interface
    # ---------------------------
    st.subheader("💬 Ask Questions about Your PDF")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("🧠 Enter your question:")

    if st.button("Ask") and user_query:
        # Retrieve top 3 relevant chunks
        retrieved_docs = vector_db.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in retrieved_docs])

        prompt = f"""
        You are an intelligent assistant. Use the context below to answer the question accurately and concisely.

        Context:
        {context}

        Question: {user_query}
        Answer:
        """

        # Generate answer
        if model_type == "Gemini":
            response = model.generate_content(prompt)
            answer = response.text
        else:
            response = model(prompt, max_length=300, do_sample=True, temperature=0.7)
            answer = response[0]['generated_text']

        # Display and store chat
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", answer))

    # ---------------------------
    # 7. Display Chat History
    # ---------------------------
    for role, text in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑‍💻 {role}:** {text}")
        else:
            st.markdown(f"**🤖 {role}:** {text}")

else:
    st.info("⬅️ Please upload a PDF from the sidebar to get started.")
