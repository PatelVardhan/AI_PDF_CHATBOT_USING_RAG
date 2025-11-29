This project is an interactive Streamlit-based application that enables users to upload a PDF and ask questions related to its content.
It uses RAG (Retrieval-Augmented Generation) to fetch the most relevant text chunks from the PDF and generate accurate answers using either Google Gemini or Flan-T5.

Features :

-Upload and process PDFs instantly
=Semantic search with FAISS
-Smart text chunking
-Chat-style Q&A about the document
-Supports Gemini API or open-source FLAN-T5

Tech Used :
Streamlit, FAISS, HuggingFace Embeddings, PyPDFLoader, Gemini / FLAN-T5

How it Works

-Upload PDF
-Split into text chunks & embed
-Search relevant chunks
-Combine context + question → generate answer
