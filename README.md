 📚 PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload one or more PDF documents and ask natural language questions. The chatbot retrieves the most relevant document chunks using FAISS and generates grounded answers using Google Gemini.

 Features

- Upload multiple PDFs
- Automatic text extraction
- Intelligent chunking
- Gemini Embeddings
- FAISS Vector Search
- Semantic Retrieval
- Grounded Question Answering
- Streamlit UI

Tech Stack

- Python
- Streamlit
- LangChain
- Google Gemini API
- FAISS
- PyPDF2

Architecture

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
FAISS
↓
Similarity Search
↓
Gemini
↓
Answer



Installation

git clone ...

pip install -r requirements.txt

streamlit run app.py

Future Improvements

- Page citations
- OCR support
- Conversation memory
- Pinecone integration
- Hybrid Search
