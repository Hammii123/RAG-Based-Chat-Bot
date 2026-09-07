import os
import re
from typing import List, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="DocuRAG - PDF Chat",
    page_icon="📚",
    layout="wide",
)

MODEL_NAME = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_embedding_model():
    """Load the open-source embedding model once per Streamlit process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    """Create the Groq client from Streamlit secrets or environment variables."""
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF -> text
# -----------------------------
def extract_pdf_text(uploaded_file) -> str:
    """Extract text from all pages of an uploaded PDF."""
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()

        if text:
            pages.append(f"[Page {page_number}]\n{text}")

    return "\n\n".join(pages)


# -----------------------------
# Text -> chunks
# -----------------------------
def create_chunks(text: str) -> List[str]:
    """
    Create overlapping word-based chunks.

    We use words rather than raw characters so chunks are easier to interpret.
    """
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start = max(0, end - CHUNK_OVERLAP)

    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def build_faiss_index(chunks: List[str]) -> Tuple[faiss.Index, np.ndarray]:
    """Embed chunks and build a cosine-similarity FAISS index."""
    model = load_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Inner product on normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def retrieve_chunks(
    query: str,
    chunks: List[str],
    index: faiss.Index,
    top_k: int = TOP_K,
) -> List[Tuple[int, float, str]]:
    """Retrieve the most semantically similar chunks."""
    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((int(idx), float(score), chunks[int(idx)]))

    return results


# -----------------------------
# RAG prompt + generation
# -----------------------------
def answer_question(query: str, retrieved_chunks):
    """Send retrieved context to Groq and generate a grounded answer."""
    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local environment "
            "or Streamlit Cloud Secrets."
        )

    context_parts = []

    for rank, (idx, score, chunk) in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"--- Retrieved Chunk {rank} | similarity={score:.3f} ---\n{chunk}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the supplied document context.

Rules:
1. Do not invent facts that are not supported by the context.
2. If the answer cannot be found in the context, clearly say:
   "I couldn't find that information in the uploaded document."
3. Give a concise, useful answer.
4. When possible, mention the page number shown in the context.
5. You may summarize or explain the retrieved information, but do not introduce
   unsupported claims.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Answer based only on the document context above.
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 DocuRAG — Chat with your PDF")
st.caption(
    "Open-source RAG pipeline: PDF → chunks → embeddings → FAISS → Groq GPT-OSS"
)

with st.sidebar:
    st.header("⚙️ RAG Settings")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**LLM:** `{MODEL_NAME}`")
    st.write(f"**Chunk size:** {CHUNK_SIZE} words")
    st.write(f"**Chunk overlap:** {CHUNK_OVERLAP} words")
    st.write(f"**Top-K retrieval:** {TOP_K}")

    if st.button("🗑️ Clear document"):
        for key in ["chunks", "faiss_index", "document_name", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()


uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="Upload a text-based PDF. Scanned/image-only PDFs need OCR, which this version does not include.",
)

if uploaded_file is not None:
    if st.session_state.get("document_name") != uploaded_file.name:
        with st.spinner("Reading, chunking and indexing your PDF..."):
            try:
                text = extract_pdf_text(uploaded_file)

                if not text.strip():
                    st.error(
                        "No extractable text was found. This may be a scanned/image-only PDF."
                    )
                    st.stop()

                chunks = create_chunks(text)

                if not chunks:
                    st.error("The PDF did not produce any usable text chunks.")
                    st.stop()

                index, _ = build_faiss_index(chunks)

                st.session_state["document_name"] = uploaded_file.name
                st.session_state["chunks"] = chunks
                st.session_state["faiss_index"] = index
                st.session_state["messages"] = []

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")
                st.stop()

    st.success(
        f"Indexed **{st.session_state['document_name']}** — "
        f"{len(st.session_state['chunks'])} chunks ready for retrieval."
    )

    # Chat history
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about your PDF...")

    if query:
        st.session_state.setdefault("messages", []).append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving relevant passages..."):
                    retrieved = retrieve_chunks(
                        query,
                        st.session_state["chunks"],
                        st.session_state["faiss_index"],
                        TOP_K,
                    )

                with st.spinner("Generating grounded answer..."):
                    answer = answer_question(query, retrieved)

                st.markdown(answer)

                with st.expander("🔎 Retrieved context"):
                    for rank, (idx, score, chunk) in enumerate(retrieved, start=1):
                        st.markdown(
                            f"**Chunk {rank} · similarity {score:.3f}**\n\n{chunk}"
                        )

                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")

else:
    st.info("Upload a PDF to build the RAG index and start chatting.")

st.divider()
st.caption(
    "Note: FAISS is an in-memory vector index in this MVP. "
    "The index is rebuilt when the document is uploaded or the Streamlit session restarts."
)
