# DocuChat

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge)](https://docuchat-asnakhan.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-GPT--OSS--120B-orange?style=for-the-badge)](https://groq.com)

> A production-ready multi-document RAG application that lets users chat with PDFs using Groq, ChromaDB, HuggingFace Embeddings, and Streamlit.

---

## Live Demo

**Application**

https://docuchat-asnakhan.streamlit.app/

> Hosted on Streamlit Community Cloud. The application may take a few seconds to wake up after inactivity.
---

## Demo 

https://github.com/user-attachments/assets/1ceb1c54-9def-4034-a3d4-7c8d3939bfc0

<br><br>

<img width="1918" height="1078" alt="Upload and process a PDF" src="https://github.com/user-attachments/assets/d9a2d52d-abb2-4326-9d08-643894d8d051" />

<br><br>

<img width="1917" height="1078" alt="Chat with sources and page citations" src="https://github.com/user-attachments/assets/d5b6d237-94d8-4fc7-8df2-b2bac2d28c47" />

<br><br>

<img width="1918" height="1078" alt="Multi-PDF history view" src="https://github.com/user-attachments/assets/6c6d15bb-2a4d-4be1-9fc5-9d2b80a1b244" />


---

## Overview

DocuChat solves a common problem — reading long documents to find specific information is slow. It chunks any uploaded PDF, embeds the content into a vector database, and uses retrieval-augmented generation to answer questions strictly from the document's actual content, with page-level source citations to prevent hallucination.

The app supports multiple PDFs simultaneously, each stored in its own isolated vector collection, with persistent per-document chat history

---

## Why Retrieval-Augmented Generation?

Large Language Models can generate incorrect or hallucinated answers when they lack relevant context.

DocuChat uses Retrieval-Augmented Generation (RAG) to first retrieve the most relevant sections from uploaded documents and then generate answers grounded in that retrieved context. Every response includes page-level citations, improving transparency and reliability.

## Features

- **Chat with any PDF** — upload a document and ask questions in plain English
- **Source-grounded answers** — every response includes the exact page number it was derived from
- **Multi-document support** — process and switch between multiple PDFs without losing context or mixing content
- **Persistent chat history** — conversation history is preserved per document
- **Production-style deployment** — containerized with Docker and deployed via an automated CI/CD pipeline

---
## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — openai/gpt-oss-120b |
| Embeddings | HuggingFace Embeddings (`all-MiniLM-L6-v2`) — local |
| Vector Database | ChromaDB (persistent, per-document collections) |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |
| Containerization | Docker |
| CI/CD | GitHub Actions → Docker Hub |
| Deployment | Streamlit Cloud |

---

## RAG Pipeline

1. **Ingestion** — the uploaded PDF is parsed and split into overlapping chunks (1000 characters, 200 character overlap) using a recursive character splitter, preserving context across chunk boundaries.
2. **Embedding** — each chunk is converted into a vector using HuggingFace's `all-MiniLM-L6-v2` model running locally, and stored in a dedicated ChromaDB collection on disk for that document.
3. **Retrieval** — when a question is asked, it is embedded using the same model, and the top 4 most relevant chunks are retrieved via cosine similarity search.
4. **Generation** — the retrieved chunks are passed as context to `openai/gpt-oss-120b` via Groq API, using a prompt template that constrains the model to answer only from the provided context, reducing hallucination.
5. **Response** — the answer is returned along with the exact page numbers it was sourced from.

```
PDF Upload → Chunking → Local Embeddings → ChromaDB (disk)
                                                │
User Question → Embed Question → Similarity Search
                                                │
                                Retrieved Context + Question
                                                │
                                openai/gpt-oss-120b (Groq)
                                                │
                                    Answer + Page Citations
```
---

## System Architecture

```text
                 User
                  │
                  ▼
         Streamlit Web Interface
                  │
                  ▼
          Document Processing
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   PyPDF Parser      Text Chunking
        │                   │
        └─────────┬─────────┘
                  ▼
      HuggingFace Embeddings
                  │
                  ▼
      ChromaDB Vector Store
                  │
                  ▼
      Groq GPT-OSS-120B API
                  │
                  ▼
      Answer with Page Citations

```
---

## Running Locally

**Prerequisites:** Python 3.11+, a Groq API key, and a HuggingFace access token.

```bash
# Clone the repository
git clone https://github.com/asnakhan-dev/docuchat.git
cd docuchat

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# Create a .env file with:
# GROQ_API_KEY=your_groq_api_key


# Run the app
streamlit run app.py
```
---

## Running with Docker

```bash
docker build -t docuchat .
docker run -p 8501:8501 --env-file .env docuchat
```

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:
1. Installs dependencies and validates the build
2. Builds a Docker image of the application
3. Pushes the image to Docker Hub

This automates the build process and keeps the containerized image in sync with the latest code.

---

## Project Structure

```
docuchat/
├── .github/workflows/    # CI/CD pipeline configuration
├── app.py                 # Main application
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md
```
---

<div align="center">

Built with Streamlit · Groq · ChromaDB · Docker

<br>

**Portfolio Project**

This repository is intended for portfolio and educational demonstration purposes only. The source code may not be copied, redistributed, or submitted as your own work without the author's permission.

<br>

© 2026 Asna Khan. All rights reserved.

</div>
