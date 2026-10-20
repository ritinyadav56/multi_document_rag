
import os
import hashlib
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever


# ============================================================
# 1. ENVIRONMENT AND CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
)

COLLECTION_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "multi_document_rag"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


# Cached objects prevent repeatedly loading models and databases.
_embeddings = None
_vector_store = None
_llm = None


# ============================================================
# 2. CLEAN MODEL RESPONSE
# ============================================================

def extract_clean_text(response) -> str:
    """
    Extract readable text from a LangChain/Gemini response.

    Handles:
    - AIMessage objects
    - Normal strings
    - Lists containing text blocks
    - Dictionaries containing text
    - Responses containing extra metadata
    """

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                if item.get("type") == "text":
                    text_value = item.get("text", "")

                    if isinstance(text_value, str):
                        text_parts.append(text_value)

        return "\n".join(text_parts).strip()

    if isinstance(content, dict):
        text_value = content.get("text")

        if isinstance(text_value, str):
            return text_value.strip()

    return str(content).strip()


# ============================================================
# 3. LOCAL EMBEDDING MODEL
# ============================================================

def get_embeddings():
    """
    Load the local HuggingFace embedding model.

    The embedding model runs locally. It does not require
    a Gemini API call to convert text into embeddings.
    """

    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


# ============================================================
# 4. PERSISTENT CHROMA VECTOR DATABASE
# ============================================================

def get_vector_store():
    """
    Open or create the persistent Chroma vector database.
    """

    global _vector_store

    if _vector_store is None:

        CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )

    return _vector_store


# ============================================================
# 5. GEMINI LANGUAGE MODEL
# ============================================================

def get_llm():
    """
    Create the Gemini chat model using GEMINI_API_KEY
    from the project's .env file.
    """

    global _llm

    if _llm is None:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found. "
                "Add your Gemini API key to the .env file."
            )

        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0,
            max_retries=2,
        )

    return _llm


# ============================================================
# 6. TEXT SPLITTER
# ============================================================

def get_text_splitter():
    """
    Split PDF text into overlapping chunks.

    Default:
        Chunk size = 1000 characters
        Chunk overlap = 150 characters
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        add_start_index=True,
    )


# ============================================================
# 7. DOCUMENT ID GENERATOR
# ============================================================

def _make_chunk_id(document: Document) -> str:
    """
    Generate a stable ID for each document chunk.
    """

    metadata = document.metadata

    unique_text = "|".join([
        str(metadata.get("document_name", "")),
        str(metadata.get("page", "")),
        str(metadata.get("start_index", "")),
        document.page_content,
    ])

    return hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 8. BUILD OR UPDATE VECTOR STORE
# ============================================================

def build_or_update_vector_store(
    file_items: Iterable[tuple[str, str]]
) -> int:
    """
    Load, split, embed, and index PDF documents.

    Parameters
    ----------
    file_items:
        An iterable of:
        (local_pdf_path, original_filename)

    Returns
    -------
    int:
        Number of chunks indexed.

    Example
    -------
    [
        ("C:/temp/resume.pdf", "resume.pdf"),
        ("C:/temp/report.pdf", "report.pdf")
    ]
    """

    vector_store = get_vector_store()
    splitter = get_text_splitter()

    total_chunks = 0

    for local_path, original_filename in file_items:

        pdf_path = Path(local_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Only PDF files are supported: {original_filename}"
            )

        # Load PDF pages.
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        if not pages:
            continue

        # Store consistent document metadata.
        for page_document in pages:

            page_document.metadata["document_name"] = (
                original_filename
            )

            # PyPDFLoader normally uses zero-based page indexes.
            page_number = page_document.metadata.get("page", 0)

            page_document.metadata["page"] = (
                int(page_number) + 1
            )

            page_document.metadata["source"] = (
                original_filename
            )

        # Split pages into smaller chunks.
        chunks = splitter.split_documents(pages)

        if not chunks:
            continue

        # Remove previous chunks for this document so that
        # re-indexing an updated PDF does not retain old chunks.
        try:
            vector_store.delete(
                where={
                    "document_name": original_filename
                }
            )
        except Exception:
            # A new or empty collection may have nothing to delete.
            pass

        chunk_ids = []

        for chunk in chunks:

            chunk.metadata["document_name"] = (
                original_filename
            )

            chunk.metadata["source"] = original_filename

            chunk_ids.append(
                _make_chunk_id(chunk)
            )

        # Add chunks and their embeddings to Chroma.
        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids,
        )

        total_chunks += len(chunks)

    return total_chunks


# ============================================================
# 9. LIST INDEXED DOCUMENTS
# ============================================================

def list_indexed_documents() -> list[str]:
    """
    Return unique PDF names stored in Chroma.
    """

    vector_store = get_vector_store()

    result = vector_store.get(
        include=["metadatas"]
    )

    document_names = set()

    for metadata in result.get("metadatas") or []:

        if not metadata:
            continue

        name = metadata.get("document_name")

        if name:
            document_names.add(name)

    return sorted(document_names)


# ============================================================
# 10. MULTI-QUERY RETRIEVER
# ============================================================

def _make_multiquery_retriever(k: int = 5):
    """
    Use Gemini to generate alternative search queries
    and retrieve relevant document chunks from Chroma.
    """

    vector_store = get_vector_store()
    llm = get_llm()

    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        },
    )

    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
        include_original=True,
    )


# ============================================================
# 11. RETRIEVE AND DEDUPLICATE DOCUMENTS
# ============================================================

def _retrieve(
    question: str,
    k: int = 5
) -> list[Document]:
    """
    Retrieve relevant chunks and remove duplicate results.
    """

    if not question or not question.strip():
        return []

    retriever = _make_multiquery_retriever(k)

    documents = retriever.invoke(
        question.strip()
    )

    unique_documents = []
    seen = set()

    for document in documents:

        key = (
            document.metadata.get("document_name", ""),
            document.metadata.get("page", ""),
            document.page_content.strip(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_documents.append(document)

    return unique_documents


# ============================================================
# 12. FORMAT RETRIEVED CONTEXT
# ============================================================

def _format_context(
    documents: list[Document]
) -> str:
    """
    Format retrieved chunks for the Gemini prompt.
    """

    formatted_parts = []

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata

        document_name = metadata.get(
            "document_name",
            "Unknown document"
        )

        page_number = metadata.get(
            "page",
            "Unknown"
        )

        formatted_parts.append(
            f"""
[Source {index}]
Document: {document_name}
Page: {page_number}

Content:
{document.page_content}
""".strip()
        )

    return "\n\n".join(formatted_parts)


# ============================================================
# 13. SOURCE PAYLOAD
# ============================================================

def _source_payload(
    documents: list[Document]
) -> list[dict]:
    """
    Return source information for the Streamlit interface.
    """

    sources = []
    seen = set()

    for document in documents:

        metadata = document.metadata

        document_name = metadata.get(
            "document_name",
            "Unknown document"
        )

        page_number = metadata.get(
            "page",
            None
        )

        excerpt = document.page_content.strip()

        key = (
            document_name,
            page_number,
            excerpt,
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "document_name": document_name,
            "page": page_number,
            "excerpt": excerpt,
            "source": metadata.get(
                "source",
                document_name
            ),
        })

    return sources


# ============================================================
# 14. CHATGPT-STYLE ANSWER PROMPT
# ============================================================

def _answer_prompt(
    question: str,
    context: str
) -> str:
    """
    Build the grounded question-answering prompt.
    """

    return f"""
You are a helpful document question-answering assistant.

Answer the user's question using the supplied document context.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:

1. Answer naturally and clearly, like ChatGPT.

2. Begin with a direct answer or a short summary when appropriate.

3. Use bullet points when listing multiple skills, facts,
   categories, findings, or recommendations.

4. Use numbered lists when explaining sequential steps.

5. Use headings only when they improve readability.

6. Keep simple answers concise. Provide more detail when the
   question requires it.

7. Include relevant document names and page numbers when useful.

8. Use only facts supported by the supplied document context.

9. If the evidence is incomplete, explain what is missing.

10. If the answer is not available in the supplied context,
    clearly say that the documents do not provide enough
    information to answer confidently.

11. Do not invent facts, sources, quotations, or page numbers.

12. Do not output Python dictionaries, JSON, raw tool output,
    signatures, response metadata, or explanations about
    internal processing.

13. Return only the final answer in readable Markdown.

FINAL ANSWER:
"""


# ============================================================
# 15. ANSWER A QUESTION
# ============================================================

def answer_question(
    question: str,
    k: int = 5
) -> dict:
    """
    Answer using indexed PDFs when they contain relevant evidence.
    Otherwise, answer using Gemini's general knowledge.

    Returns:
        {"answer": "...", "sources": [...]}
    """

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
        }

    question = question.strip()

    try:
        # Retrieve possible evidence from the indexed PDFs.
        documents = _retrieve(question=question, k=k) if list_indexed_documents() else []

        # Ask Gemini whether the retrieved text actually helps answer
        # this question. This avoids forcing unrelated PDF text into answers.
        use_documents = False

        if documents:
            context = _format_context(documents)
            routing_prompt = f"""
You are a strict evidence-relevance classifier for a PDF question-answering assistant.

Decide whether the supplied PDF excerpts contain relevant information that can
meaningfully answer the user's question.

Return exactly one word:
DOCUMENTS - only if the excerpts contain relevant evidence for answering the question.
GENERAL - if the excerpts are irrelevant, only loosely related, or do not contain
enough information to answer the question.

Do not answer the user's question. Do not use outside knowledge for this decision.

USER QUESTION:
{question}

RETRIEVED PDF EXCERPTS:
{context}

CLASSIFICATION:
"""
            route_response = get_llm().invoke(routing_prompt)
            route = extract_clean_text(route_response).strip().upper()
            use_documents = route.startswith("DOCUMENTS")

        if use_documents:
            context = _format_context(documents)
            prompt = _answer_prompt(question=question, context=context)
            response = get_llm().invoke(prompt)
            answer = extract_clean_text(response)

            if not answer:
                answer = "I couldn't generate an answer from the document evidence."

            return {
                "answer": answer,
                "sources": _source_payload(documents),
            }

        # No useful PDF evidence: answer from Gemini's general knowledge.
        general_prompt = f"""
You are a helpful general-purpose AI assistant, powered by Gemini.

The user's question was not sufficiently answered by the indexed PDF documents.
Answer the question using your general knowledge.

Instructions:
- Answer naturally and clearly, like ChatGPT.
- Start with the direct answer.
- Use headings, bullets, or numbered steps when helpful.
- Keep simple answers concise and explain complex topics clearly.
- Do not claim that information came from the PDFs.
- If the question asks for current or very recent facts that you cannot verify,
  clearly mention that limitation rather than pretending to have live information.
- If you are uncertain, say so.
- Return only the final answer in readable Markdown.
- Do not return JSON, Python dictionaries, or internal processing details.

USER QUESTION:
{question}

ANSWER:
"""
        response = get_llm().invoke(general_prompt)
        answer = extract_clean_text(response)

        if not answer:
            answer = "I couldn't generate an answer. Please try asking again."

        return {
            "answer": answer,
            "sources": [],
        }

    except Exception as error:
        return {
            "answer": (
                "I couldn't complete the question. Please check your Gemini API "
                "configuration and application logs.\n\n"
                f"Error: {error}"
            ),
            "sources": [],
        }


# ============================================================
# 16. COMPARE TWO DOCUMENTS
# ============================================================

def compare_documents(
    document_a: str,
    document_b: str,
    question: str,
    k: int = 5,
) -> dict:
    """
    Compare two indexed documents using their relevant chunks.

    Returns:
    {
        "answer": "...",
        "sources": [...]
    }
    """

    if document_a == document_b:

        return {
            "answer": (
                "Please select two different documents to compare."
            ),
            "sources": [],
        }

    if not question or not question.strip():

        return {
            "answer": "Please enter a comparison question.",
            "sources": [],
        }

    try:

        vector_store = get_vector_store()

        # Search only within document A.
        docs_a = vector_store.similarity_search(
            question,
            k=k,
            filter={
                "document_name": document_a
            },
        )

        # Search only within document B.
        docs_b = vector_store.similarity_search(
            question,
            k=k,
            filter={
                "document_name": document_b
            },
        )

        if not docs_a and not docs_b:

            return {
                "answer": (
                    "No relevant content was found in either "
                    "document. Check that both PDFs are indexed."
                ),
                "sources": [],
            }

        context_a = _format_context(docs_a)
        context_b = _format_context(docs_b)

        prompt = f"""
You are a document comparison assistant.

Compare the following two documents using only the
provided evidence.

DOCUMENT A: {document_a}

{context_a if context_a else "No relevant evidence was retrieved."}


DOCUMENT B: {document_b}

{context_b if context_b else "No relevant evidence was retrieved."}


USER QUESTION:
{question}

INSTRUCTIONS:

- Start with a short summary of the main finding.
- Use bullet points or a Markdown table when useful.
- Clearly distinguish information from Document A and Document B.
- Explain important similarities and differences.
- Do not assume that missing information means a fact is false.
- Do not invent facts or unsupported comparisons.
- If one document lacks relevant information, say so explicitly.
- Keep the answer readable and appropriately concise.
- Return only the final answer in Markdown.
"""

        response = get_llm().invoke(prompt)

        answer = extract_clean_text(response)

        sources = _source_payload(
            docs_a + docs_b
        )

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as error:

        return {
            "answer": (
                "Document comparison failed. Check your API "
                "configuration and application logs.\n\n"
                f"Error: {error}"
            ),
            "sources": [],
        }


# ============================================================
# 17. EVALUATE ANSWER AGAINST DOCUMENT EVIDENCE
# ============================================================

def evaluate_question(
    question: str,
    k: int = 5,
) -> dict:
    """
    Evaluate whether the indexed documents support an answer.

    Possible labels:
    - Supported
    - Partially supported
    - Not supported
    """

    if not question or not question.strip():

        return {
            "label": "Not evaluated",
            "explanation": "Please enter a question.",
            "sources": [],
        }

    try:

        documents = _retrieve(
            question=question,
            k=k,
        )

        if not documents:

            return {
                "label": "Not supported",
                "explanation": (
                    "No relevant evidence was retrieved from "
                    "the indexed documents."
                ),
                "sources": [],
            }

        context = _format_context(documents)

        prompt = f"""
You are an evidence evaluation assistant.

Evaluate whether the document evidence is sufficient
to answer the user's question.

USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{context}

Choose exactly one label:

Supported
Partially supported
Not supported

Evaluation rules:

- Supported: The retrieved evidence directly supports
  the relevant answer.

- Partially supported: Some information is supported,
  but important details are missing or uncertain.

- Not supported: The evidence does not support the answer.

Do not use outside knowledge to fill gaps.

Return your result in this exact format:

Label: <one label>

Explanation:
<Brief explanation based on the evidence>
"""

        response = get_llm().invoke(prompt)

        evaluation_text = extract_clean_text(response)

        match = re.search(
            r"Label\s*:\s*"
            r"(Supported|Partially supported|Not supported)",
            evaluation_text,
            flags=re.IGNORECASE,
        )

        if match:
            label = match.group(1).title()

            if label.lower() == "partially supported":
                label = "Partially supported"

        else:
            label = "Unable to classify"

        explanation_match = re.search(
            r"Explanation\s*:\s*(.*)",
            evaluation_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        explanation = (
            explanation_match.group(1).strip()
            if explanation_match
            else evaluation_text
        )

        return {
            "label": label,
            "explanation": explanation,
            "sources": _source_payload(documents),
        }

    except Exception as error:

        return {
            "label": "Evaluation failed",
            "explanation": str(error),
            "sources": [],
        }


# ============================================================
# 18. OPTIONAL: QUICK PIPELINE STATUS
# ============================================================

def get_pipeline_status() -> dict:
    """
    Return basic information about the configured pipeline.
    """

    return {
        "embedding_model": EMBEDDING_MODEL,
        "gemini_model": GEMINI_MODEL,
        "chroma_directory": str(CHROMA_DIR),
        "collection_name": COLLECTION_NAME,
        "indexed_documents": list_indexed_documents(),
    }

