import streamlit as st
import requests

# the URL of your FastAPI app
API_URL = "http://localhost:8000"

# page setup
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="centered"
)

# title
st.title("📄 Document Q&A API")
st.write("Upload a document and ask questions about it!")

# ── SECTION 1 - Upload Document ──────────────────────
st.header("Step 1 — Upload Your Document")

# tab for file upload or text input
tab1, tab2 = st.tabs(["Upload File", "Paste Text"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a .txt or .pdf file",
        type=["txt", "pdf"]
    )
    if st.button("Ingest File", key="file_btn"):
        if uploaded_file:
            # send file to /ingest
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{API_URL}/ingest", files=files)
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error("Something went wrong!")
        else:
            st.warning("Please upload a file first!")

with tab2:
    text_input = st.text_area(
        "Paste your text here",
        height=200,
        placeholder="Paste any paragraph or document text here..."
    )
    if st.button("Ingest Text", key="text_btn"):
        if text_input:
            # send text to /ingest
            response = requests.post(
                f"{API_URL}/ingest",
                data={"text": text_input}
            )
            if response.status_code == 200:
                st.success(response.json()["message"])
            else:
                st.error("Something went wrong!")
        else:
            st.warning("Please paste some text first!")

# ── SECTION 2 - Ask Questions ────────────────────────
st.header("Step 2 — Ask a Question")

# show conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# question input
question = st.chat_input("Ask a question about your document...")

if question:
    # show user question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # send to /ask
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )

    if response.status_code == 200:
        answer = response.json()["answer"]
        # show AI answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    else:
        st.error("Something went wrong! Make sure you uploaded a document first.")