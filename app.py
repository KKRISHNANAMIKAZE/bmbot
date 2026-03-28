import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os

# ---------------- CONFIG ---------------- #

st.set_page_config(page_title="Chat PDF")

PDF_FOLDER_PATH = "publications"
FAISS_INDEX_PATH = "faiss_index"

# ✅ GROQ CLIENT
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- PDF PROCESSING ---------------- #

def get_pdf_text_from_folder(folder_path):
    text = ""
    if not os.path.exists(folder_path):
        return text

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


def process_pdfs():
    raw_text = get_pdf_text_from_folder(PDF_FOLDER_PATH)

    if raw_text.strip() == "":
        return False

    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)
    return True

# ---------------- GROQ MODEL ---------------- #

def get_conversational_chain():

    def run_chain(inputs):
        context = inputs["context"][:1200]   # ✅ reduced size
        question = inputs["question"]

        prompt = f"""
You are an intelligent research assistant.

Summarize clearly:
- main idea
- method used
- key contribution

Do NOT copy full text. Keep it concise (5–7 lines).

Context:
{context}

Question:
{question}

Answer:
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            return {"text": response.choices[0].message.content}

        except Exception as e:
            return {"text": f"⚠️ Groq Error: {str(e)}"}

    return run_chain

# ---------------- UTIL ---------------- #

def list_paper_titles(docs):
    return "\n".join([doc.metadata.get("title", "Untitled") for doc in docs])


def list_author_papers(author_name, docs):
    papers = []
    chain = get_conversational_chain()

    for doc in docs:
        if author_name.lower() in doc.page_content.lower():
            title = doc.metadata.get("title", "Untitled")

            summary = chain({
                "context": doc.page_content[:1200],
                "question": "Summarize this paper"
            }).get("text", "")

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

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(FAISS_INDEX_PATH):
        st.warning("Processing PDFs...")
        success = process_pdfs()

        if not success:
            st.error("No PDFs found or readable.")
            return

    db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(user_question)

    if "list" in user_question.lower() and "titles" in user_question.lower():
        titles_text = list_paper_titles(docs)
        st.write("Paper Titles:\n", titles_text)
        current_response_text = titles_text

    elif "contribution" in user_question.lower() or "work by" in user_question.lower():
        author_name = user_question.split("by")[-1].strip()
        author_papers_text = list_author_papers(author_name, docs)
        st.write(f"Papers by {author_name}:\n", author_papers_text)
        current_response_text = author_papers_text

    else:
        context = "\n".join([doc.page_content for doc in docs])

        chain = get_conversational_chain()

        response = chain({
            "context": context,
            "question": user_question
        })

        current_response_text = response.get("text", "")

        st.markdown("**Reply:**")
        st.write(current_response_text)

    if 'history' not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "question": user_question,
        "reply": current_response_text
    })

    feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 8])
    feedback_col3.write("Provide feedback:")

    feedback_col4, feedback_col5 = st.columns([1, 1])
    if feedback_col4.button("👍"):
        st.write("Thanks for your feedback!")
    if feedback_col5.button("👎"):
        st.write("Sorry to hear that.")

# ---------------- UI ---------------- #

def main():
    st.header("Chat with your BEE Lab, BM Dept, NIT Rourkela🔍📝")

    if 'history' not in st.session_state:
        st.session_state.history = []

    user_question = st.text_area("""Welcome to the BEE Lab chatbot!

Ask questions about research papers and get summarized answers.

How can I assist you today?""")

    if user_question:
        user_input(user_question)

    st.sidebar.subheader("Chat History")

    if st.session_state.history:
        for i, chat in enumerate(reversed(st.session_state.history[-10:])):
            if st.sidebar.button(f"Topic {i+1}: {chat['question']}", key=f"history_button_{i}"):
                st.sidebar.write(chat['reply'])

st.markdown("""
<style>
.fixed-text {
    position: fixed;
    right: 10px;
    font-size: 12px;
}
.fixed-text-1 {
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
}
</style>

<div class="fixed-text fixed-text-1">@2024 by BEE Lab, NIT Rourkela</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()