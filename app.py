import asyncio
import sys
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3-70b-8192"


def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # RecursiveCharacterTextSplitter splits by paragraph, line, sentence, word in order
    # chunk_overlap=200 ensures context is not lost at chunk boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN")
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


def create_rag_chain(vector_store):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,  # low temperature for factual, grounded answers
        max_tokens=1024
    )

    # top 4 most similar chunks retrieved per query
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # prompt constrains the LLM to answer strictly from retrieved context
    # prevents hallucination by explicitly instructing it not to guess
    prompt = PromptTemplate(
        template="""You are a helpful assistant that answers questions strictly based on the provided document context.

Context:
{context}

Question:
{question}

Instructions:
- Answer only from the context above
- If the answer is not present, say "I couldn't find this in the document"
- Be concise and direct

Answer:""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LCEL pipeline: retrieve → format → prompt → llm → parse output
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def get_answer(rag_tuple, question):
    chain, retriever = rag_tuple
    answer = chain.invoke(question)

    # retriever called separately to extract page-level source metadata
    source_docs = retriever.invoke(question)
    sources = list(set(
        f"Page {doc.metadata.get('page', 0) + 1} — {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
        for doc in source_docs
    ))

    return answer, sources


def init_session():
    defaults = {
        "all_histories": {},   # {pdf_name: [{question, answer, sources}]}
        "vector_stores": {},   # {pdf_name: faiss_vector_store}
        "rag_chain": None,
        "pdf_processed": False,
        "processed_pdfs": set(),
        "active_pdf": None,
        "show_history": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_history():
    pdf = st.session_state.active_pdf
    if pdf and pdf in st.session_state.all_histories:
        return st.session_state.all_histories[pdf]
    return []


def switch_to_pdf(pdf_name):
    vector_store = st.session_state.vector_stores[pdf_name]
    st.session_state.rag_chain = create_rag_chain(vector_store)
    st.session_state.active_pdf = pdf_name
    st.session_state.pdf_processed = True


def main():
    st.set_page_config(page_title="DocuChat", layout="wide")
    init_session()

    h1, h2 = st.columns([3, 1])
    with h1:
        st.title("DocuChat")
        if st.session_state.active_pdf:
            st.caption(f"Chatting with: {st.session_state.active_pdf}")
        else:
            st.caption("Chat with your PDFs using LLaMA 3 70B + RAG")
    with h2:
        st.write("")
        st.write("")
        if st.button("Chat History", use_container_width=True):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()

    st.divider()

    if st.session_state.show_history:
        st.subheader("Chat History")

        if not st.session_state.all_histories:
            st.info("No history yet.")
        else:
            pdf_names = [p for p, h in st.session_state.all_histories.items() if h]

            if not pdf_names:
                st.info("No conversations yet.")
            else:
                tabs = st.tabs(pdf_names)
                for tab, pdf_name in zip(tabs, pdf_names):
                    with tab:
                        for i, chat in enumerate(st.session_state.all_histories[pdf_name], 1):
                            st.markdown(f"**Q{i}: {chat['question']}**")
                            st.write(chat["answer"])
                            if chat["sources"]:
                                with st.expander("Sources"):
                                    for s in chat["sources"]:
                                        st.caption(f"- {s}")
                            if i < len(st.session_state.all_histories[pdf_name]):
                                st.divider()

        if st.button("Close History"):
            st.session_state.show_history = False
            st.rerun()

        return

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("Upload PDF")

        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            os.makedirs("docs", exist_ok=True)
            pdf_path = f"docs/{uploaded_file.name}"

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            already_done = uploaded_file.name in st.session_state.processed_pdfs

            if already_done:
                st.success(f"'{uploaded_file.name}' already processed.")
                if st.session_state.active_pdf != uploaded_file.name:
                    switch_to_pdf(uploaded_file.name)
                    st.rerun()
            else:
                st.success(f"Uploaded: {uploaded_file.name}")
                if st.button("Process PDF", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        st.info("Chunking PDF...")
                        chunks = load_and_split_pdf(pdf_path)

                        st.info("Creating embeddings...")
                        vector_store = create_vector_store(chunks)

                        st.info("Building RAG chain...")
                        st.session_state.vector_stores[uploaded_file.name] = vector_store
                        st.session_state.processed_pdfs.add(uploaded_file.name)
                        st.session_state.all_histories[uploaded_file.name] = []
                        st.session_state.rag_chain = create_rag_chain(vector_store)
                        st.session_state.active_pdf = uploaded_file.name
                        st.session_state.pdf_processed = True

                    st.success("Ready. Start asking questions.")
                    st.rerun()

        if st.session_state.processed_pdfs:
            st.divider()
            st.subheader("Your PDFs")

            for pdf_name in st.session_state.processed_pdfs:
                is_active = pdf_name == st.session_state.active_pdf
                btn_label = f"• {pdf_name}" if is_active else pdf_name

                if st.button(btn_label, key=f"switch_{pdf_name}", use_container_width=True):
                    if not is_active:
                        switch_to_pdf(pdf_name)
                        st.rerun()

        if st.session_state.pdf_processed:
            st.divider()
            st.subheader("Details")
            st.markdown(f"**Model:** `{LLM_MODEL}`")
            st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
            st.markdown(f"**Vector DB:** FAISS")
            st.markdown(f"**Chunk size:** 1000 chars")
            st.markdown(f"**Top-K:** 4 chunks")

            if st.button("Clear Chat", use_container_width=True):
                if st.session_state.active_pdf:
                    st.session_state.all_histories[st.session_state.active_pdf] = []
                st.rerun()

    with right_col:
        if st.session_state.active_pdf:
            st.subheader(f"Chat — {st.session_state.active_pdf}")
        else:
            st.subheader("Chat")

        if not st.session_state.pdf_processed:
            st.info("Upload and process a PDF to start chatting.")
            st.markdown("""
            **How it works:**
            - Upload any PDF and click Process
            - Ask questions in plain English
            - Answers include exact page references
            - Switch between multiple PDFs anytime
            - Full history saved per PDF
            """)

        else:
            history = get_active_history()
            for chat in history:
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant"):
                    st.write(chat["answer"])
                    if chat["sources"]:
                        with st.expander("View Sources"):
                            for s in chat["sources"]:
                                st.caption(f"- {s}")

            user_question = st.chat_input("Ask anything about this PDF...")

            if user_question:
                with st.chat_message("user"):
                    st.write(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer, sources = get_answer(
                            st.session_state.rag_chain,
                            user_question
                        )
                    st.write(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for s in sources:
                                st.caption(f"- {s}")

                entry = {"question": user_question, "answer": answer, "sources": sources}
                st.session_state.all_histories[st.session_state.active_pdf].append(entry)


if __name__ == "__main__":
    main()