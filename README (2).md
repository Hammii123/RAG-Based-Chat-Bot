# 📚 DocuRAG — Chat with Your PDF

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its content.

The application extracts text from the PDF, splits it into chunks, converts the chunks into vector embeddings, stores them in a **FAISS** vector index, retrieves the most relevant chunks for a user's question, and sends the retrieved context to **GPT-OSS 120B through Groq** to generate a grounded answer.

---

## 🚀 Features

- 📄 Upload PDF documents
- 🔎 Extract text from PDFs using `pypdf`
- ✂️ Split documents into overlapping chunks
- 🧠 Generate embeddings using an open-source Sentence Transformers model
- 🗃️ Store embeddings in FAISS
- 🔍 Perform semantic similarity search
- 🤖 Generate answers using `openai/gpt-oss-120b` through Groq
- 💬 Chat-style Streamlit interface
- 🔎 Display retrieved context for transparency/debugging
- 🔐 Use Streamlit Secrets for the Groq API key
- ☁️ Deploy easily on Streamlit Community Cloud

---

## 🏗️ RAG Architecture

```text
                 USER
                  │
                  ▼
          ┌─────────────────┐
          │ Upload PDF      │
          │ Streamlit       │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Extract Text    │
          │ pypdf           │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Create Chunks   │
          │ + Overlap       │
          └────────┬────────┘
                   │
                   ▼
          ┌──────────────────────┐
          │ Sentence Transformers│
          │ Embeddings           │
          └──────────┬───────────┘
                     │
                     ▼
              ┌─────────────┐
              │    FAISS    │
              │ Vector Index│
              └──────┬──────┘
                     │
              User asks question
                     │
                     ▼
          ┌─────────────────────┐
          │ Embed Question      │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ FAISS Similarity    │
          │ Search              │
          └──────────┬──────────┘
                     │
                Top-K Chunks
                     │
                     ▼
          ┌─────────────────────┐
          │ Groq GPT-OSS 120B   │
          │ Question + Context  │
          └──────────┬──────────┘
                     │
                     ▼
                  ANSWER
```

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application language |
| Streamlit | Frontend/UI |
| pypdf | PDF text extraction |
| Sentence Transformers | Text embeddings |
| `all-MiniLM-L6-v2` | Open-source embedding model |
| FAISS | Vector similarity search |
| NumPy | Vector processing |
| Groq | LLM inference |
| GPT-OSS 120B | Open-weight language model |
| GitHub | Source code hosting |
| Streamlit Community Cloud | Deployment |

---

## 📁 Project Structure

```text
pdf-rag-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ How the Application Works

### 1. Upload a PDF

The user uploads a PDF through Streamlit.

```text
PDF
 ↓
Streamlit file uploader
```

### 2. Extract Text

The application uses `pypdf` to extract text from every page.

```python
reader = PdfReader(uploaded_file)
```

Page numbers are preserved in the extracted text so that the application can provide useful source context.

### 3. Create Chunks

The extracted document is divided into overlapping chunks.

Current settings:

```text
Chunk size:    1000 words
Chunk overlap: 150 words
```

The overlap helps preserve context between neighboring chunks.

### 4. Generate Embeddings

Each chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
Vector
```

### 5. Store Vectors in FAISS

The vectors are stored in an in-memory FAISS index.

The application uses normalized embeddings and inner-product search, which corresponds to cosine similarity.

### 6. User Asks a Question

For example:

```text
What is the main objective of this document?
```

The question is converted into an embedding.

### 7. Retrieve Relevant Chunks

FAISS searches for the most semantically similar chunks.

The application retrieves:

```text
TOP_K = 5
```

chunks by default.

### 8. Generate the Answer

The retrieved chunks are added to the prompt and sent to:

```text
Groq
  ↓
openai/gpt-oss-120b
```

The model is instructed to answer using only the retrieved document context.

---

## 🔐 Environment Variable

The application requires a Groq API key.

For local development, set:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Do **not** put your API key directly inside `app.py`.

Never commit your API key to GitHub.

---

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-chatbot.git
```

Move into the project:

```bash
cd pdf-rag-chatbot
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

### 5. Start Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud

### Step 1 — Push the project to GitHub

Create a GitHub repository and upload:

```text
app.py
requirements.txt
README.md
.gitignore
```

### Step 2 — Open Streamlit Community Cloud

Go to Streamlit Community Cloud and connect your GitHub account.

### Step 3 — Create the application

Select:

```text
Repository: YOUR_USERNAME/pdf-rag-chatbot
Branch: main
Main file: app.py
```

### Step 4 — Add the API key

Open the application's **Advanced settings / Secrets** section.

Add:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

Save the secret and deploy the application.

---

## ⚠️ Current Limitations

This is an MVP/learning-focused RAG application.

### 1. FAISS is in-memory

The FAISS index is rebuilt when a document is processed.

It is not currently a persistent multi-user vector database.

### 2. Scanned PDFs

The current version uses `pypdf` and does not include OCR.

Therefore, image-only/scanned PDFs may not produce usable text.

### 3. One active document

The UI is designed around the currently uploaded PDF rather than a persistent document collection.

### 4. Basic chunking

The application uses word-based chunking rather than advanced semantic or structure-aware chunking.

### 5. No reranking

Retrieved chunks are currently ranked using vector similarity only.

---

## 🔮 Future Improvements

Possible improvements include:

- [ ] OCR support for scanned PDFs
- [ ] Persistent FAISS indexes
- [ ] Multiple PDF support
- [ ] Document management
- [ ] Metadata filtering
- [ ] Better semantic chunking
- [ ] Recursive text splitting
- [ ] Hybrid search
- [ ] Reranking models
- [ ] Conversation memory
- [ ] Source/page citations
- [ ] RAG evaluation
- [ ] Answer confidence scoring
- [ ] Streaming LLM responses
- [ ] User authentication
- [ ] Persistent database
- [ ] Production monitoring

---

## 🧠 RAG Concepts Demonstrated

This project is useful for learning the core concepts behind Retrieval-Augmented Generation:

```text
Document Processing
       ↓
Chunking
       ↓
Tokenization
       ↓
Embeddings
       ↓
Vector Search
       ↓
Retrieval
       ↓
Context Construction
       ↓
Prompt Grounding
       ↓
LLM Generation
```

A key distinction:

**Chunking and tokenization are different operations.**

This application creates chunks using words. The embedding model then performs its own model-specific tokenization internally.

---

## 🧪 Example Questions

After uploading a PDF, try questions such as:

```text
What is the main objective of this document?

Summarize the key findings.

What methodology was used?

What are the main conclusions?

What problems does the document identify?

According to the document, what are the recommended solutions?
```

You can also inspect the **Retrieved Context** section to understand which document passages were supplied to the LLM.

---

## 🔒 Security Notes

Never commit secrets such as:

```text
GROQ_API_KEY
```

Do not put API keys in:

```text
app.py
README.md
GitHub commits
screenshots
```

For Streamlit Community Cloud, use the application's Secrets configuration.

---

## 📜 License

You can add a license depending on how you plan to distribute the project.

For an open-source portfolio project, an MIT License is a common choice.

---

## 👨‍💻 Project Goal

This project demonstrates how to build a RAG application from the ground up without hiding the core retrieval pipeline behind a large framework.

The main goal is to understand:

```text
How does RAG actually work?
```

rather than simply calling a high-level RAG framework.

---

## ⭐ If You Like This Project

Consider giving the repository a star and improving the project with additional RAG features such as OCR, reranking, hybrid retrieval, persistent vector storage, and evaluation.
