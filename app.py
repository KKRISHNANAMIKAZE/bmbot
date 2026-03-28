import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="Chat PDF")

PDF_FOLDER_PATH = "publications"
FAISS_INDEX_PATH = "faiss_index"
CHECKSUM_FILE_PATH = "checksum.txt"

# ✅ GROQ CLIENT
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- PDF PROCESSING ---------------- #

def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=800)
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local(FAISS_INDEX_PATH)


def calculate_checksum(folder_path):
    hasher = hashlib.md5()
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()


def process_pdfs():
    new_checksum = calculate_checksum(PDF_FOLDER_PATH)

    if not os.path.exists(FAISS_INDEX_PATH) or (
        os.path.exists(CHECKSUM_FILE_PATH) and open(CHECKSUM_FILE_PATH).read() != new_checksum
    ):
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text_from_folder(PDF_FOLDER_PATH)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)

            with open(CHECKSUM_FILE_PATH, "w") as f:
                f.write(new_checksum)

            st.success("Processing complete and FAISS index updated.")

# ---------------- GROQ CHAIN ---------------- #

def get_groq_response(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ latest working
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq Error: {str(e)}"

# ---------------- UTIL ---------------- #

def list_paper_titles(docs):
    return "\n".join([doc.metadata.get("title", "Untitled") for doc in docs])


def list_author_papers(author_name, docs):
    papers = []

    for doc in docs:
        if author_name.lower() in doc.page_content.lower():
            title = doc.metadata.get("title", "Untitled")

            prompt = f"""
Summarize this research paper in 5–6 lines:

{doc.page_content[:1200]}
"""

            summary = get_groq_response(prompt)

            papers.append(f"Title: {title}\nSummary: {summary}\n")

    return "\n".join(papers)


def list_pdf_files_with_keyword(folder_path, keyword):
    return [
        f for f in os.listdir(folder_path)
        if f.endswith(".pdf") and keyword.lower() in f.lower()
    ]


def list_pdf_files_by_author(folder_path, author_name):
    pdf_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            pdf_reader = PdfReader(pdf_path)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            if author_name.lower() in text.lower():
                pdf_files.append(filename)

    return pdf_files

# ---------------- MAIN LOGIC ---------------- #

def user_input(user_question):

    if "list" in user_question.lower() and "papers on" in user_question.lower():
        keyword = user_question.split("on")[-1].strip()
        files = list_pdf_files_with_keyword(PDF_FOLDER_PATH, keyword)

        if files:
            result = "\n".join(f"{i+1}. {f}" for i, f in enumerate(files))
            st.write("Matching PDF Files:\n", result)
        else:
            st.write("No matching PDF files found.")
        return

    if "what work" in user_question.lower() or "contributions done by" in user_question.lower():
        author_name = user_question.split("by")[-1].strip()
        files = list_pdf_files_by_author(PDF_FOLDER_PATH, author_name)

        if files:
            result = "\n".join(f"{i+1}. {f}" for i, f in enumerate(files))
            st.write(f"Papers by {author_name}:\n", result)
        else:
            st.write("No papers found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search(user_question)

    if "list" in user_question.lower() and "titles" in user_question.lower():
        st.write("Paper Titles:\n", list_paper_titles(docs))
        return

    if "contribution" in user_question.lower() or "work by" in user_question.lower():
        author_name = user_question.split("by")[-1].strip()
        st.write(f"Papers by {author_name}:\n", list_author_papers(author_name, docs))
        return

    context = "\n".join([doc.page_content for doc in docs])[:1500]

    # 🔥 MAIN ANSWER
    prompt = f"""
Based on the context, describe the research done.

Context:
{context}

Question:
{user_question}

Answer clearly:
"""
    answer = get_groq_response(prompt)

    st.markdown("**Reply:**")
    st.write(answer)

    # 🔥 FUTURE PROSPECTS (same as your old UI)
    future_prompt = f"""
Based on the context, suggest future research directions.

Context:
{context}

Question:
{user_question}

Future Prospects:
"""
    future = get_groq_response(future_prompt)

    st.markdown("**Future Prospects:**")
    st.write(future)

# ---------------- UI ---------------- #

def main():
    st.header("Chat with your BEE Lab, BM Dept, NIT Rourkela🔍📝")

    process_pdfs()

    user_question = st.text_area("Ask your question:")

    if user_question:
        user_input(user_question)

st.markdown("""
<style>
.fixed-text {
    position: fixed;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 12px;
}
</style>

<div class="fixed-text">@2024 by BEE Lab, NIT Rourkela</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()