from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

DB_PATH = "db"

# PROCESS MULTIPLE FILES
def process_pdfs(file_paths):
    all_docs = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )
    db.persist()


def ask_question(query, chat_history):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})  # 🔥 reduce noise

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    # ⚠️ VERY IMPORTANT: memory only for follow-ups, not full reasoning
    last_q = ""
    if len(chat_history) > 0:
        last_q = chat_history[-1]["content"]

    llm = OllamaLLM(model="phi3")

    prompt = f"""
You are a strict AI assistant.

RULES:
- Answer ONLY using the provided context.
- DO NOT use outside knowledge.
- If answer is not clearly in context, say: "I don't know".
- Ignore irrelevant chat history.

Context:
{context}

Question:
{query}

Answer:
"""

    answer = llm.invoke(prompt)

    return answer, docs