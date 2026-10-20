
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from rag_pipeline import (
    build_or_update_vector_store,
    answer_question,
    list_indexed_documents,
    compare_documents,
    evaluate_question,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multi-Document RAG Assistant")
st.caption(
    "Upload PDFs, ask questions, retrieve evidence, "
    "and compare documents using your RAG pipeline."
)

# --------------------------------------------------
# SIDEBAR: PDF UPLOAD AND INDEXING
# --------------------------------------------------

st.sidebar.header("📄 Document Management")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload your resume, reports, manuals, or other PDFs.",
)

if st.sidebar.button(
    "📥 Index Uploaded PDFs",
    use_container_width=True,
):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:
        file_items = []
        temp_paths = []

        try:
            with st.spinner(
                "Reading PDFs, generating embeddings, and indexing..."
            ):
                for uploaded_file in uploaded_files:
                    # Preserve the original filename for source citations.
                    with NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False,
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = Path(temp_file.name)

                    temp_paths.append(temp_path)

                    file_items.append(
                        (str(temp_path), uploaded_file.name)
                    )

                chunk_count = build_or_update_vector_store(file_items)

            st.sidebar.success(
                f"Successfully indexed {chunk_count} chunks."
            )

        except Exception as exc:
            st.sidebar.error(f"Indexing failed: {exc}")

        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

# --------------------------------------------------
# SHOW INDEXED DOCUMENTS
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("📚 Indexed Documents")

indexed_documents = list_indexed_documents()

if indexed_documents:
    for document in indexed_documents:
        st.sidebar.write(f"📄 {document}")
else:
    st.sidebar.info("No indexed PDFs found.")

if st.sidebar.button("🔄 Refresh Document List"):
    st.rerun()

# --------------------------------------------------
# MAIN QUESTION-ANSWERING INTERFACE
# --------------------------------------------------

st.subheader("💬 Ask Your Documents")

st.write(
    "Ask questions about the contents of your indexed PDFs. "
    "Answers are generated using retrieved document evidence."
)

with st.form("question_form"):
    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Examples:\n"
            "• What skills are mentioned in my resume?\n"
            "• What projects are listed in my resume?\n"
            "• Summarize the main points of this report."
        ),
        height=110,
    )

    top_k = st.slider(
        "Chunks to retrieve per search query",
        min_value=1,
        max_value=10,
        value=5,
    )

    ask_button = st.form_submit_button(
        "🔍 Get Answer",
        type="primary",
        use_container_width=True,
    )

if ask_button:
    if not question.strip():
        st.warning("Please enter a question.")

    elif not list_indexed_documents():
        st.warning(
            "Index PDFs first. Upload a PDF in the sidebar "
            "and click 'Index Uploaded PDFs'."
        )

    else:
        try:
            with st.spinner(
                "Searching your documents and generating an answer..."
            ):
                result = answer_question(
                    question.strip(),
                    k=top_k,
                )

            st.markdown("### 💡 Answer")
            st.markdown(result["answer"])

            st.markdown("### 📚 Retrieved Sources")

            sources = result.get("sources", [])

            if sources:
                for index, source in enumerate(sources, start=1):
                    title = (
                        f"{index}. {source['document']} "
                        f"— Page {source['page']}"
                    )

                    with st.expander(title):
                        st.write(source["excerpt"])

                        st.caption(
                            "Source citations refer to the indexed PDF."
                        )
            else:
                st.info("No relevant source chunks were retrieved.")

        except Exception as exc:
            st.error(f"Question failed: {exc}")

# --------------------------------------------------
# EVIDENCE EVALUATION
# --------------------------------------------------

st.divider()
st.subheader("🧪 Evidence Evaluation")

st.write(
    "Check whether the retrieved PDF content supports a question or claim."
)

with st.form("evaluation_form"):
    evaluation_question = st.text_input(
        "Question or claim to evaluate",
        placeholder=(
            "Example: Does the resume mention Python and SQL?"
        ),
    )

    evaluate_button = st.form_submit_button(
        "Evaluate Evidence",
        use_container_width=True,
    )

if evaluate_button:
    if not evaluation_question.strip():
        st.warning("Please enter a question or claim.")

    elif not list_indexed_documents():
        st.warning("Upload and index PDFs first.")

    else:
        try:
            with st.spinner("Evaluating document evidence..."):
                evaluation = evaluate_question(
                    evaluation_question.strip()
                )

            st.markdown(
                f"**Support status:** {evaluation['support_label']}"
            )

            st.write(evaluation["assessment"])

            st.write(
                f"**Retrieved chunks:** "
                f"{evaluation['retrieved_chunks']}"
            )

            for index, source in enumerate(
                evaluation.get("sources", []),
                start=1,
            ):
                with st.expander(
                    f"Evidence {index}: {source['document']} "
                    f"— Page {source['page']}"
                ):
                    st.write(source["excerpt"])

        except Exception as exc:
            st.error(f"Evidence evaluation failed: {exc}")

# --------------------------------------------------
# DOCUMENT COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("⚖️ Compare Documents")

indexed_documents = list_indexed_documents()

if len(indexed_documents) < 2:
    st.info(
        "Index at least two different PDFs to compare documents."
    )

else:
    col1, col2 = st.columns(2)

    with col1:
        document_a = st.selectbox(
            "Select the first document",
            indexed_documents,
            key="document_a",
        )

    with col2:
        document_b = st.selectbox(
            "Select the second document",
            indexed_documents,
            index=1,
            key="document_b",
        )

    comparison_question = st.text_input(
        "What would you like to compare?",
        placeholder=(
            "Example: Compare the skills and projects in both resumes."
        ),
    )

    if st.button(
        "⚖️ Compare Documents",
        use_container_width=True,
    ):
        if document_a == document_b:
            st.warning("Please select two different documents.")

        elif not comparison_question.strip():
            st.warning("Enter a comparison question.")

        else:
            try:
                with st.spinner("Comparing retrieved document evidence..."):
                    comparison = compare_documents(
                        document_a,
                        document_b,
                        comparison_question.strip(),
                    )

                st.markdown("### Comparison Result")
                st.markdown(comparison["answer"])

                st.markdown("### Sources")

                for index, source in enumerate(
                    comparison.get("sources", []),
                    start=1,
                ):
                    with st.expander(
                        f"{index}. {source['document']} "
                        f"— Page {source['page']}"
                    ):
                        st.write(source["excerpt"])

            except Exception as exc:
                st.error(f"Document comparison failed: {exc}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by LangChain, ChromaDB, Hugging Face embeddings, "
    "and the Gemini API."
)

