import streamlit as st
import os
from rag import process_pdfs, ask_question

st.set_page_config(page_title="AskDocs AI", layout="wide")

st.title("🤖 AskDocs AI")
st.caption("Chat with your documents using AI")

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

# SIDEBAR
st.sidebar.header("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    file_paths = []

    for file in uploaded_files:
        path = os.path.join("temp_" + file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        file_paths.append(path)

    if st.sidebar.button("⚡ Process Documents"):
        process_pdfs(file_paths)
        st.session_state.processed = True
        st.sidebar.success("Documents processed!")

# CHAT DISPLAY
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT
query = st.chat_input("Ask anything about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if not st.session_state.processed:
        response = "Please upload and process documents first."
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = ask_question(query, st.session_state.messages)

                st.markdown(answer)

                # SHOW SOURCES
                with st.expander("📌 Sources"):
                    for i, d in enumerate(docs):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(d.page_content[:300] + "...")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })