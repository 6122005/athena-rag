"""Streamlit Web UI for Athena RAG — PDF Upload & Self-Correcting Q&A System."""

import streamlit as st
from athena.graph import build_graph
from athena.ingestion import extract_pdf_pages, ingest_documents


def main():
    # Page configuration
    try:
        st.set_page_config(
            page_title="Athena RAG — PDF Agentic Q&A",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        pass

    # Custom CSS
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .source-badge {
            background-color: #EEF2FF;
            color: #4338CA;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-right: 6px;
            display: inline-block;
            margin-bottom: 6px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph()

    # --- SIDEBAR: Document & PDF Ingestion Only ---
    with st.sidebar:
        st.header("📄 Knowledge Ingestion")
        st.markdown("Upload any document or multi-page PDF (supports **100+ pages**).")

        uploaded_file = st.file_uploader(
            "Choose a PDF or Document",
            type=["pdf", "txt", "md"],
            help="Upload large reports, manuals, policies, or books.",
        )

        with st.expander("🛠️ Advanced Settings", expanded=False):
            chunk_size = st.slider("Chunk Size (characters)", 500, 2500, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 400, 150, 25)
            max_rewrites = st.slider("Max Query Rewrites", 0, 5, 2, 1)

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_name = uploaded_file.name

            # Check if this file has been indexed
            is_already_indexed = any(f["name"] == file_name for f in st.session_state.ingested_files)
            if not is_already_indexed:
                st.info("📌 Click the button below to index this document into the AI knowledge base.")

            if st.button("🚀 Ingest & Index Document", use_container_width=True, type="primary"):
                progress_bar = st.progress(0, text=f"Analyzing {file_name}...")

                try:
                    progress_bar.progress(20, text=f"Extracting text from {file_name}...")
                    if file_name.lower().endswith(".pdf"):
                        docs = extract_pdf_pages(
                            file_bytes,
                            filename=file_name,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                    else:
                        text_content = file_bytes.decode("utf-8", errors="ignore")
                        from athena.ingestion import chunk_text
                        chunks = chunk_text(text_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        docs = [
                            {"source": file_name, "content": c, "metadata": {"filename": file_name, "chunk_index": i}}
                            for i, c in enumerate(chunks)
                        ]

                    if not docs:
                        st.error("No extractable text found in this file.")
                    else:
                        total_docs = len(docs)
                        def on_batch_progress(curr, tot):
                            pct = min(100, int(25 + (curr / tot) * 75))
                            progress_bar.progress(pct, text=f"Indexing chunk {curr} of {tot} ({pct}%)...")

                        ingested_count = ingest_documents(docs, callback=on_batch_progress)
                        progress_bar.progress(100, text="✅ Indexing Complete!")

                        if not any(f["name"] == file_name for f in st.session_state.ingested_files):
                            st.session_state.ingested_files.append({
                                "name": file_name,
                                "chunks": ingested_count,
                            })

                        # Rebuild graph to bind immediately to the updated vector store
                        st.session_state.graph = build_graph()
                        st.success(f"✅ Ready! Ingested **{ingested_count} chunks** from `{file_name}`.")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error ingesting document: {e}")

        st.markdown("---")
        st.subheader("📊 Ingested Library")
        if st.session_state.ingested_files:
            for f in st.session_state.ingested_files:
                st.markdown(f"• **{f['name']}** ({f['chunks']} chunks)")
        else:
            st.info("No custom documents uploaded yet.")

        st.markdown("---")
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN CHAT & Q&A AREA ---
    st.markdown(
        '<div class="main-title">Athena RAG — Self-Correcting PDF Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Ask questions across your 100+ page PDFs. Athena grades retrieval relevance and rewrites failed queries before answering.</div>',
        unsafe_allow_html=True,
    )

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                st.markdown("**Sources & Page References:**")
                for src in msg["sources"]:
                    st.markdown(f'<span class="source-badge">📄 {src}</span>', unsafe_allow_html=True)

            if "trace" in msg and msg["trace"]:
                with st.expander("🔍 View LangGraph Self-Correction Trace"):
                    for step in msg["trace"]:
                        node = step.get("node", "step")
                        if node == "retrieve":
                            st.write(f"• **Retrieve**: Fetched `{step.get('count')}` candidate documents for query: *'{step.get('query')}'*")
                        elif node == "grade":
                            st.write(f"• **Grade Relevance**: Scores = `{step.get('scores')}` | All Relevant = `{step.get('all_relevant')}`")
                        elif node == "rewrite":
                            st.write(f"• **Rewrite Query (Attempt {step.get('attempt')})**: Reformulated query to *'{step.get('new_query')}'*")
                        elif node == "generate":
                            st.write(f"• **Generate**: Produced grounded answer citing `{step.get('sources')}`")

    # User Query Input
    user_question = st.chat_input("Ask any question from your uploaded document...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving, grading context, and verifying answer..."):
                initial_state = {
                    "question": user_question,
                    "retrieval_query": user_question,
                    "rewrite_count": 0,
                    "max_rewrites": max_rewrites,
                    "trace": [],
                }

                result = st.session_state.graph.invoke(initial_state)

                answer = result.get("answer", "No answer could be generated.")
                sources = result.get("sources", [])
                trace = result.get("trace", [])

                st.markdown(answer)

                if sources:
                    st.markdown("**Sources & Page References:**")
                    for src in sources:
                        st.markdown(f'<span class="source-badge">📄 {src}</span>', unsafe_allow_html=True)

                if trace:
                    with st.expander("🔍 View LangGraph Self-Correction Trace"):
                        for step in trace:
                            node = step.get("node", "step")
                            if node == "retrieve":
                                st.write(f"• **Retrieve**: Fetched `{step.get('count')}` candidate documents for query: *'{step.get('query')}'*")
                            elif node == "grade":
                                st.write(f"• **Grade Relevance**: Scores = `{step.get('scores')}` | All Relevant = `{step.get('all_relevant')}`")
                            elif node == "rewrite":
                                st.write(f"• **Rewrite Query (Attempt {step.get('attempt')})**: Reformulated query to *'{step.get('new_query')}'*")
                            elif node == "generate":
                                st.write(f"• **Generate**: Produced grounded answer citing `{step.get('sources')}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "trace": trace,
                })


if __name__ == "__main__":
    main()
