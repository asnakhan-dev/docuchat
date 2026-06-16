# ============================================================
# IMPORTS — ye sab libraries hain jo RAG pipeline chalati hain
# ============================================================

import streamlit as st
# Streamlit — Python se seedha web UI banata hai, koi HTML/CSS nahi chahiye
# Recruiters ke liye: rapid prototyping tool, data apps ke liye industry standard

import os
# os — operating system se interact karne ke liye
# Yahan use hoga: file paths aur environment variables padhne ke liye

from dotenv import load_dotenv
# python-dotenv — .env file se API keys padhta hai
# Best practice: API keys code mein hardcode nahi karte, .env mein rakhte hain

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader — PDF file ko padhta hai aur text extract karta hai page by page
# Internally pypdf use karta hai

from langchain.text_splitter import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter — extracted text ko chhote chhote chunks mein todta hai
# "Recursive" kyunki pehle paragraphs pe split karta hai, phir sentences, phir words
# Ye important hai kyunki LLM ek saath poora document nahi padh sakta

from langchain_community.vectorstores import Chroma
# Chroma — vector database hai
# Ye embeddings (numbers) ko store karta hai aur similarity search karta hai
# Local disk pe save hota hai — koi external server nahi chahiye

from langchain_community.embeddings import HuggingFaceEmbeddings
# HuggingFaceEmbeddings — text ko numbers (vectors) mein convert karta hai
# Model use hoga: all-MiniLM-L6-v2 — small, fast, aur accurate
# Free hai — koi API key nahi chahiye HuggingFace ke liye

from langchain_groq import ChatGroq
# ChatGroq — Groq API ka LangChain wrapper
# Groq pe llama-3.3-70b-versatile run hoga — fast aur free

from langchain.chains import RetrievalQA
# RetrievalQA — poori RAG chain ko ek saath jodta hai
# Retrieval (ChromaDB se chunks dhundho) + QA (Groq se answer lo)

from langchain.prompts import PromptTemplate
# PromptTemplate — LLM ko dene wala prompt format karta hai
# Context + User Question ko ek structured prompt mein daalta hai

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

load_dotenv()
# .env file load karo — iske baad os.getenv() se API key milegi

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Environment variable se Groq API key lo
# Directly code mein likhna KABHI mat karo — security risk hai

CHROMA_DB_PATH = "chroma_db"
# ChromaDB local folder path — yahan embeddings save hongi disk pe
# Baar baar PDF process nahi karna padega — ek baar store, baar baar use

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# HuggingFace ka embedding model
# all-MiniLM-L6-v2 — RAG ke liye most popular choice
# Small (80MB), fast, aur English text ke liye excellent accuracy

LLM_MODEL = "llama-3.3-70b-versatile"
# Groq pe available LLaMA 3.3 70B model
# 70B parameters — bahut powerful, aur Groq pe lightning fast hai

# ============================================================
# PART 2 — PDF PROCESSING FUNCTIONS
# ============================================================

def load_and_split_pdf(pdf_path):
    """
    PDF ko load karke chhote chunks mein todta hai.
    
    Kyun chunking zaroori hai?
    LLM ka context window limited hota hai — wo poora 
    100 page ka document ek saath nahi padh sakta.
    Isliye hum document ko chhote pieces mein tod dete hain.
    """
    
    # ---- Step 1: PDF Load karo ----
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    # PyPDFLoader har page ko ek alag Document object banata hai
    # documents = [Page1, Page2, Page3, ...]
    # Har Document mein 2 cheezein hoti hain:
    #   - page_content: us page ka text
    #   - metadata: {"source": "file.pdf", "page": 0}
    
    # ---- Step 2: Text ko chunks mein todo ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        # Har chunk maximum 1000 characters ka hoga
        # Kyun 1000? — embedding model ke liye optimal size
        # Bahut bada chunk = relevant info dhoondna mushkil
        # Bahut chota chunk = context incomplete

        chunk_overlap=200,
        # Consecutive chunks mein 200 characters common rahenge
        # Kyun overlap? — agar answer chunk boundary pe ho
        # toh bhi milega, information loss nahi hoga
        # Example: chunk1 ends at "...neural net" 
        #          chunk2 starts from "neural net works by..."

        length_function=len,
        # Characters count karne ke liye Python ka built-in len() use karo

        separators=["\n\n", "\n", ".", " ", ""]
        # Kahan se split kare — priority order mein:
        # 1. Pehle double newline (paragraph boundary) pe try karo
        # 2. Phir single newline pe
        # 3. Phir sentence end (.) pe
        # 4. Phir space pe
        # 5. Last resort: characters pe
        # "Recursive" naam isliye hai — pehle bada separator try karta hai
    )
    
    chunks = text_splitter.split_documents(documents)
    # split_documents() — Document objects ko split karta hai
    # Metadata automatically carry hoti hai har chunk mein
    # Matlab har chunk ko pata hai wo kaunse PDF ke kaunse page se aaya
    
    return chunks
    # Return: list of Document objects (chunks)
    # Example: 50 page PDF → ~200-300 chunks


def create_vector_store(chunks):
    """
    Chunks ko embeddings mein convert karke ChromaDB mein save karta hai.
    
    Embedding kya hoti hai?
    Text ko numbers ki list mein convert karna.
    "cat" → [0.2, 0.8, 0.1, ...]  (384 numbers)
    "kitten" → [0.19, 0.79, 0.11, ...] (similar numbers!)
    Similar meaning = similar numbers = similarity search possible
    """
    
    # ---- Step 3: Embedding Model load karo ----
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # "all-MiniLM-L6-v2" — HuggingFace ka popular embedding model
        # Pehli baar run hoga toh ~80MB download hoga automatically
        
        model_kwargs={"device": "cpu"},
        # CPU pe run karo — GPU nahi chahiye is model ke liye
        # Laptop pe bhi fast kaam karta hai
        
        encode_kwargs={"normalize_embeddings": True}
        # Normalize = sab embeddings same scale pe laao (0 to 1)
        # Kyun? — cosine similarity ke liye accurate results milte hain
    )
    
    # ---- Step 4: ChromaDB mein store karo ----
    vector_store = Chroma.from_documents(
        documents=chunks,
        # Wo chunks jo humne upar banaye — text + metadata

        embedding=embeddings,
        # Embedding model — har chunk ka vector banayega

        persist_directory=CHROMA_DB_PATH
        # "chroma_db" folder mein locally save hoga
        # Iska fayda: ek baar process karo, baar baar use karo
        # App restart hone pe dubara PDF process nahi karna padega
    )
    
    return vector_store
    # Return: ChromaDB vector store object
    # Iske andar similarity search kar sakte hain


def load_existing_vector_store():
    """
    Agar PDF pehle se process ho chuki hai toh
    ChromaDB se seedha load karo — time bachao.
    
    Kyun ye function zaroori hai?
    Embedding generation slow process hai.
    Same PDF baar baar process karna waste hai.
    Agar chroma_db folder already exist karta hai
    toh seedha wahan se load karo.
    """
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
        # Same embedding model use karna zaroori hai
        # Jo model store karne mein use hua, same se load karo
        # Different model = different vector space = wrong results
    )
    
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        # Existing chroma_db folder se load karo
        embedding_function=embeddings
    )
    
    return vector_store

# ============================================================
# PART 3 — RAG CHAIN (RETRIEVAL + GROQ LLM)
# ============================================================

def create_rag_chain(vector_store):
    """
    Poori RAG pipeline ko ek saath jodta hai.
    
    RAG ka flow:
    User Question → Embed Question → ChromaDB Search 
    → Relevant Chunks → Prompt → Groq LLM → Answer
    """
    
    # ---- Step 1: Groq LLM initialize karo ----
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        # .env se load ki hui API key use karo
        # Kabhi bhi hardcode mat karo

        model_name=LLM_MODEL,
        # "llama-3.3-70b-versatile" — powerful aur fast
        # Groq pe ye model 300+ tokens/second generate karta hai

        temperature=0.2,
        # Temperature = creativity level (0.0 to 1.0)
        # 0.0 = robot jaise exact answers, har baar same
        # 1.0 = creative but kabhi kabhi galat bhi bol sakta hai
        # 0.2 = document based accurate answers ke liye best
        # RAG mein hamesha low temperature rakho — facts chahiye creativity nahi

        max_tokens=1024,
        # Ek answer mein maximum 1024 tokens generate karega
        # 1 token ≈ 0.75 words, toh ~768 words ka answer
        # Groq ke free tier mein sufficient hai
    )
    
    # ---- Step 2: Retriever banao ----
    retriever = vector_store.as_retriever(
        search_type="similarity",
        # Cosine similarity use karke relevant chunks dhundho
        # User ka question embed karo → ChromaDB mein similar vectors dhundo
        # Jo vector sabse paas ho → wo chunk most relevant hai

        search_kwargs={"k": 4}
        # k=4 matlab top 4 most relevant chunks retrieve karo
        # Kyun 4? — enough context milta hai, 
        # bahut zyada chunks = prompt bada = slow response
        # Experiment kar sakte ho: k=3 (fast) ya k=6 (more context)
    )
    
    # ---- Step 3: Custom Prompt Template banao ----
    prompt_template = """
You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
- Answer ONLY based on the context provided above
- If the answer is not in the context, say "I couldn't find this information in the document"
- Be concise and clear
- Mention which part of the document supports your answer

ANSWER:
"""
    # Ye prompt template bahut important hai — RAG ka "brain" hai
    # {context} = ChromaDB se retrieved chunks yahan aayenge
    # {question} = user ka sawaal yahan aayega
    # Instructions isliye diye:
    #   1. Hallucination rokne ke liye — "only from context"
    #   2. Honest response — agar nahi pata toh bataye
    #   3. Source mention karna — transparency ke liye
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
        # LangChain ko batao ki template mein 
        # kaunse variables replace honge
    )
    
    # ---- Step 4: RetrievalQA Chain banao ----
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        # Groq ka LLaMA model — answer generate karega

        chain_type="stuff",
        # "stuff" = sab chunks ek saath prompt mein daal do
        # Simple aur effective for small-medium documents
        # Alternatives:
        #   "map_reduce" = bade documents ke liye, chunks alag process hote hain
        #   "refine" = iteratively answer improve karta hai

        retriever=retriever,
        # ChromaDB retriever — relevant chunks dhundega

        chain_type_kwargs={"prompt": PROMPT},
        # Hamara custom prompt use karo

        return_source_documents=True,
        # Answer ke saath source chunks bhi return karo
        # UI mein "Sources" section dikhayenge
        # Recruiter ko impress karta hai — transparency!
    )
    
    return rag_chain


def get_answer(rag_chain, user_question):
    """
    User ka question RAG chain mein bhejta hai
    aur answer + sources return karta hai.
    """
    
    # RAG chain invoke karo user ke question ke saath
    result = rag_chain.invoke({"query": user_question})
    # Internally ye sab hota hai:
    # 1. user_question ko embed karo (HuggingFace model)
    # 2. ChromaDB mein top-4 similar chunks dhundo
    # 3. Chunks + question ko prompt mein daalo
    # 4. Groq LLM ko bhejo
    # 5. Answer wapis aao
    
    answer = result["result"]
    # LLM ka generated answer
    
    source_documents = result["source_documents"]
    # Wo chunks jinse answer generate hua
    # Har chunk mein metadata hai — page number, source file
    
    # Source pages extract karo metadata se
    sources = []
    for doc in source_documents:
        page_num = doc.metadata.get("page", 0) + 1
        # +1 isliye kyunki PDF pages 0 se start hoti hain code mein
        # But users ke liye Page 1 se dikhana natural hai
        
        source_file = doc.metadata.get("source", "Unknown")
        # PDF file ka naam
        
        sources.append(f"Page {page_num} — {source_file}")
    
    # Duplicate sources hata do
    sources = list(set(sources))
    
    return answer, sources
    # Dono return karo — answer aur sources list
    
# ============================================================
# PART 4 — STREAMLIT UI
# Streamlit top to bottom execute karta hai har baar
# jab user koi action leta hai
# ============================================================

def main():

    # Page configuration — browser tab title aur layout
    st.set_page_config(
        page_title="DocuChat",
        page_icon="",
        layout="wide"
    )

    st.title("DocuChat")
    st.markdown("Chat with your PDF using LLaMA 3.3 + RAG")
    st.divider()

    # ---- Session State ----
    # Streamlit har action pe page reload karta hai
    # session_state mein jo save karo wo reload ke baad bhi rehta hai
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # ---- Two column layout ----
    # Left = PDF upload, Right = Chat
    left_col, right_col = st.columns([1, 2])

    # ================================================
    # LEFT COLUMN — PDF Upload
    # ================================================
    with left_col:
        st.subheader("Upload PDF")

        uploaded_file = st.file_uploader(
            label="Choose a PDF file",
            type=["pdf"]
        )

        if uploaded_file is not None:
            # File disk pe save karo
            # PyPDFLoader file path chahta hai, direct object nahi leta
            pdf_path = f"docs/{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"Uploaded: {uploaded_file.name}")

            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing... this may take a minute"):
                    # PDF load aur chunk karo
                    st.info("Reading and chunking PDF...")
                    chunks = load_and_split_pdf(pdf_path)

                    # Embeddings banao aur ChromaDB mein store karo
                    st.info("Creating embeddings...")
                    vector_store = create_vector_store(chunks)

                    # RAG chain banao
                    st.info("Building RAG chain...")
                    st.session_state.rag_chain = create_rag_chain(vector_store)

                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []
                    # Naya PDF process hone pe chat history clear karo

                st.success("Done. Ask your questions now.")

        # PDF process hone ke baad stats dikhao
        # Ye recruiter ko dikhata hai ki tum apna stack samajhte ho
        if st.session_state.pdf_processed:
            st.divider()
            st.subheader("Details")
            st.markdown(f"**Model:** `{LLM_MODEL}`")
            st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
            st.markdown(f"**Vector DB:** ChromaDB")
            st.markdown(f"**Chunk size:** 1000 chars")
            st.markdown(f"**Overlap:** 200 chars")
            st.markdown(f"**Top-K:** 4 chunks")

            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    # ================================================
    # RIGHT COLUMN — Chat Interface
    # ================================================
    with right_col:
        st.subheader("Chat")

        if not st.session_state.pdf_processed:
            # PDF upload nahi hui toh guide karo user ko
            st.info("Upload and process a PDF from the left panel to start.")
            st.markdown("""
            **How it works:**
            - Upload any PDF document
            - The app splits it into chunks and creates embeddings
            - Ask questions in plain English
            - Get answers with exact page references
            """)

        else:
            # Chat history dikhao — purane sawal jawab
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(chat["question"])

                with st.chat_message("assistant"):
                    st.write(chat["answer"])

                    # Sources expander mein hide karo — clean UI
                    if chat["sources"]:
                        with st.expander("View Sources"):
                            for source in chat["sources"]:
                                st.caption(f"- {source}")

            # Question input — page ke bottom pe stick karta hai
            user_question = st.chat_input("Ask anything about your PDF...")

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
                            for source in sources:
                                st.caption(f"- {source}")

                # Conversation history mein save karo
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer,
                    "sources": sources
                })


# ============================================================
# ENTRY POINT
# Seedha run karo toh main() call hoga
# Koi import kare toh nahi chalega automatically
# ============================================================
if __name__ == "__main__":
    main()