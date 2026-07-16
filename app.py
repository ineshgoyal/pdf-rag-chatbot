import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS

# ---------------------------------
# Load environment variables
# ---------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ---------------------------------
# Extract text from uploaded PDFs
# ---------------------------------
def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ---------------------------------
# Split text into smaller chunks
# ---------------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# ---------------------------------
# Create and save FAISS database
# ---------------------------------
def get_vector_store(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ---------------------------------
# Answer user's question
# ---------------------------------
def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retrieve the 4 most relevant chunks
    docs = db.similarity_search( user_question,k=4)

    # Combine retrieved PDF chunks
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Gemini model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )
# Major part of the code is to create a prompt for the LLM that includes the context from the PDF and the user's question. The prompt instructs the model to answer strictly based on the provided PDF context, without using any external knowledge or inventing information. If the answer is not available in the PDF, it should respond with a specific message.
    prompt = f"""
You are a PDF question-answering assistant.

STRICT RULES:

1. Answer ONLY using the PDF context provided below.
2. Do NOT use your own general knowledge.
3. If the answer is not present in the context, say exactly:
   "Answer not available in the uploaded PDF."
4. Do not invent information.

PDF CONTEXT:
{context}

USER QUESTION:
{user_question}

ANSWER:
"""

    response = llm.invoke(prompt) #crucial part of the code where the prompt is sent to the LLM for generating a response based on the PDF context and user question.

    st.subheader("Answer")
    st.write(response.content)


# ---------------------------------
# Main Streamlit application
# ---------------------------------
def main():

    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="📚"
    )

    st.title("📚 Chat with Your PDF")
    st.write(
        "Upload a PDF, process it, and ask questions "
        "based only on the PDF content."
    )

    # Sidebar
    with st.sidebar:

        st.title("Upload PDF")

        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):

            if not pdf_docs:
                st.warning("Please upload at least one PDF.")

            else:

                with st.spinner("Processing your PDF..."):

                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error(
                            "No readable text was found in the PDF."
                        )

                    else:

                        chunks = get_text_chunks(raw_text)

                        get_vector_store(chunks)

                        st.success(
                            "PDF processed successfully!"
                        )

    # Question box
    user_question = st.text_input(
        "Ask a question from your PDF"
    )

    if user_question:

        if not os.path.exists("faiss_index"):
            st.warning(
                "Please upload and process a PDF first."
            )

        else:
            user_input(user_question)


# ---------------------------------
# Run application
# ---------------------------------
if __name__ == "__main__":
    main()