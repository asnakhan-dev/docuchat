# DocuChat

A multi-document RAG (Retrieval-Augmented Generation) application that lets you chat with your PDFs using natural language. Upload any PDF, ask questions, and get accurate answers with exact page citations — powered by LLaMA 3.3 70B.

**Live App:** [docuchat-asnakhan.streamlit.app](https://docuchat-asnakhan.streamlit.app/)

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

## Features

- **Chat with any PDF** — upload a document and ask questions in plain English
- **Source-grounded answers** — every response includes the exact page number it was derived from
- **Multi-document support** — process and switch between multiple PDFs without losing context or mixing content
- **Persistent chat history** — conversation history is preserved per document
- **Production-style deployment** — containerized with Docker and deployed via an automated CI/CD pipeline

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — LLaMA 3.3 70B Versatile |
| Orchestration | LangChain (LCEL) |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| Frontend | Streamlit |
| PDF Parsing | PyPDF |
| Containerization | Docker |
| CI/CD | GitHub Actions → Docker Hub |
| Deployment | Streamlit Cloud |

## How It Works

1. **Ingestion** — the uploaded PDF is parsed and split into overlapping chunks (1000 characters, 200 character overlap) using a recursive character splitter, preserving context across chunk boundaries.
2. **Embedding** — each chunk is converted into a vector representation via the HuggingFace Inference API and stored in a dedicated ChromaDB collection for that document.
3. **Retrieval** — when a question is asked, it is embedded using the same model, and the top 4 most relevant chunks are retrieved via cosine similarity search.
4. **Generation** — the retrieved chunks are passed as context to LLaMA 3.3 70B via Groq, using a prompt template that constrains the model to answer only from the provided context, reducing hallucination.
5. **Response** — the answer is returned along with the exact page numbers it was sourced from.

```
PDF Upload → Chunking → Embeddings → ChromaDB
                                          │
User Question → Embed Question → Similarity Search
                                          │
                              Retrieved Context + Question
                                          │
                                  LLaMA 3.3 70B (Groq)
                                          │
                              Answer + Page Citations
```

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
# HF_TOKEN=your_huggingface_token

# Run the app
streamlit run app.py
```

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

## Project Structure

```
docuchat/
├── .github/workflows/    # CI/CD pipeline configuration
├── app.py                 # Main application
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
└── README.md
```
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Asna Khan**
GitHub: [@asnakhan-dev](https://github.com/asnakhan-dev)
