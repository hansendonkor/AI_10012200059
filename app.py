"""Academic City Manual RAG Chatbot

Name: Hansen Donkor
Index Number: 10012200059

Streamlit UI that demonstrates a fully manual RAG pipeline:
Query -> Chunking -> Retrieval -> Reranking -> Prompt -> LLM -> Answer (+ logs)

Constraints: No LangChain / LlamaIndex.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List

import streamlit as st
from dotenv import load_dotenv

from src.config import AppConfig
from src.data_loader import load_sources
from src.cleaner import clean_documents
from src.chunker import chunk_documents
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.keyword_search import KeywordSearch
from src.retriever import hybrid_retrieve
from src.reranker import apply_domain_scores, rerank
from src.prompt_builder import build_prompt
from src.generator import generate_answer
from src.logger import JsonlLogger


def _set_app_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'DM Sans', sans-serif;
            background: #0e0e14;
            color: #e8e6f0;
        }

        [data-testid="stAppViewContainer"] > .main {
            background: #0e0e14;
        }

        [data-testid="stSidebar"] {
            background: #13131c;
            border-right: 1px solid rgba(160,153,247,0.10);
        }

        [data-testid="stSidebar"] * {
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Topbar ── */
        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2.6rem;
            padding-bottom: 1.2rem;
            border-bottom: 1px solid rgba(160,153,247,0.10);
        }
        .logo-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-mark {
            width: 38px;
            height: 38px;
            border-radius: 11px;
            background: linear-gradient(135deg, #7c74f5, #a78bfa);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 17px;
        }
        .logo-name {
            font-size: 15px;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: #f0eeff;
        }
        .logo-sub {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6b6880;
            margin-top: 1px;
        }
        .model-badge {
            background: rgba(124,116,245,0.12);
            color: #a99ff5;
            font-size: 11px;
            font-weight: 500;
            padding: 5px 13px;
            border-radius: 100px;
            border: 1px solid rgba(124,116,245,0.22);
            letter-spacing: 0.04em;
            font-family: 'DM Mono', monospace;
        }

        /* ── Hero ── */
        .hero {
            margin-bottom: 2.4rem;
        }
        .hero__headline {
            font-size: 3rem;
            font-weight: 300;
            letter-spacing: -0.05em;
            line-height: 1.04;
            color: #f0eeff;
            margin: 0 0 0.7rem;
        }
        .hero__headline strong {
            font-weight: 600;
            background: linear-gradient(90deg, #a99ff5, #c4b5fd);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero__subtext {
            font-size: 0.94rem;
            color: #6b6880;
            max-width: 500px;
            line-height: 1.7;
            margin: 0;
            font-weight: 300;
        }

        /* ── Section labels ── */
        .section-header {
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #4e4b63;
            margin-bottom: 0.7rem;
            margin-top: 1.8rem;
        }

        /* ── Cards ── */
        .panel {
            background: #13131c;
            border: 1px solid rgba(160,153,247,0.10);
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1rem;
        }
        .panel--accent {
            border-color: rgba(160,153,247,0.28);
        }

        /* ── Inputs ── */
        .stTextArea textarea,
        .stTextInput input {
            border-radius: 13px !important;
            border: 1px solid rgba(160,153,247,0.14) !important;
            background: #13131c !important;
            color: #e8e6f0 !important;
            font-size: 0.92rem !important;
            font-family: 'DM Sans', sans-serif !important;
        }
        .stTextArea textarea:focus,
        .stTextInput input:focus {
            border-color: rgba(160,153,247,0.45) !important;
            box-shadow: 0 0 0 3px rgba(124,116,245,0.10) !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: #3e3c52 !important;
        }

        /* ── Buttons ── */
        .stButton > button {
            border-radius: 13px;
            font-weight: 500;
            font-size: 0.88rem;
            letter-spacing: 0.01em;
            padding: 0.65rem 1.5rem;
            transition: all 0.15s;
            font-family: 'DM Sans', sans-serif !important;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #7c74f5, #9d8ff7) !important;
            border: none !important;
            color: #fff !important;
            box-shadow: 0 4px 16px rgba(124,116,245,0.25) !important;
        }
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 6px 22px rgba(124,116,245,0.38) !important;
            transform: translateY(-1px);
        }
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid rgba(160,153,247,0.14) !important;
            color: #6b6880 !important;
        }
        .stButton > button[kind="secondary"]:hover {
            border-color: rgba(160,153,247,0.28) !important;
            color: #a99ff5 !important;
        }

        /* ── Metrics ── */
        [data-testid="stMetric"] {
            background: #13131c;
            border: 1px solid rgba(160,153,247,0.10);
            border-radius: 16px;
            padding: 1rem 1.2rem;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 10px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            color: #4e4b63 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 300 !important;
            letter-spacing: -0.04em !important;
            color: #f0eeff !important;
        }

        /* ── Expanders ── */
        [data-testid="stExpander"] {
            background: #13131c !important;
            border: 1px solid rgba(160,153,247,0.10) !important;
            border-radius: 14px !important;
            margin-bottom: 8px;
        }
        [data-testid="stExpander"] summary {
            font-size: 0.83rem !important;
            font-weight: 500 !important;
            color: #a99ff5 !important;
            font-family: 'DM Mono', monospace !important;
        }

        /* ── Code block ── */
        .stCode, pre, code {
            border-radius: 14px !important;
            font-family: 'DM Mono', monospace !important;
            font-size: 0.78rem !important;
            background: #0a0a10 !important;
        }

        /* ── Multiselect ── */
        [data-testid="stMultiSelect"] > div {
            border-radius: 13px !important;
            border: 1px solid rgba(160,153,247,0.14) !important;
            background: #13131c !important;
        }

        /* ── Selectbox / Slider ── */
        [data-testid="stSelectbox"] > div > div,
        .stSlider > div {
            font-family: 'DM Sans', sans-serif !important;
        }

        /* ── Sidebar labels ── */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stCheckbox label {
            font-size: 10px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            color: #4e4b63 !important;
        }

        /* ── Dividers ── */
        hr {
            border-color: rgba(160,153,247,0.08) !important;
            margin: 1.4rem 0 !important;
        }

        /* ── Alerts / Info ── */
        [data-testid="stAlert"] {
            border-radius: 13px !important;
            border: 1px solid rgba(160,153,247,0.16) !important;
            background: rgba(124,116,245,0.07) !important;
        }

        /* ── Spinner ── */
        [data-testid="stSpinner"] {
            color: #a99ff5 !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(124,116,245,0.22);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(124,116,245,0.40);
        }

        /* ── JSON viewer ── */
        .stJson {
            border-radius: 14px !important;
            background: #0a0a10 !important;
        }

        /* ── Warning ── */
        [data-testid="stWarning"] {
            border-radius: 13px !important;
            background: rgba(234,179,8,0.07) !important;
            border: 1px solid rgba(234,179,8,0.18) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def build_pipeline(config: AppConfig):
    docs = load_sources(config)
    docs = clean_documents(docs)

    embedder = Embedder(model_name=config.embedding_model)

    def build_indexes(chunking_strategy: str):
        chunks = chunk_documents(
            docs,
            strategy=chunking_strategy,
            chunk_size_chars=config.chunk_size_chars,
            overlap_chars=config.chunk_overlap_chars,
        )

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts, normalize=True)
        vstore = VectorStore()
        vstore.build(embeddings=embeddings, chunks=chunks)

        kstore = KeywordSearch()
        kstore.fit(texts=texts)

        return chunks, vstore, kstore

    return docs, embedder, build_indexes


def main() -> None:
    load_dotenv()

    config = AppConfig.from_env()
    logger = JsonlLogger(path=config.log_path)

    st.set_page_config(
        page_title="ACity RAG · Terence Anquandah",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _set_app_style()

    st.markdown(
        """
        <div class="topbar">
            <div class="logo-wrap">
                <div class="logo-mark">⬡</div>
                <div>
                    <div class="logo-name">ACity RAG</div>
                    <div class="logo-sub">Manual Pipeline</div>
                </div>
            </div>
            <span class="model-badge">No LangChain · No LlamaIndex</span>
        </div>
        <div class="hero">
            <div class="hero__headline">Ghana election &amp;<br><strong>budget intelligence</strong></div>
            <p class="hero__subtext">
                Manual retrieval-augmented generation. Every component hand-built — chunking, embeddings, retrieval, reranking, and prompt construction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        chunking_strategy = st.selectbox(
            "Chunking strategy",
            options=["fixed", "section"],
            index=0,
            help="Use fixed-size chunking or section-aware chunking.",
        )
        top_k = st.slider("Top K", min_value=3, max_value=20, value=config.top_k, step=1)
        max_context_tokens = st.slider(
            "Max context tokens",
            min_value=256,
            max_value=4096,
            value=config.max_context_tokens,
            step=128,
        )

        st.subheader("Rerank Weights")
        w_vec = st.slider("Vector weight", 0.0, 1.0, float(config.w_vector), 0.05)
        w_kw = st.slider("Keyword weight", 0.0, 1.0, float(config.w_keyword), 0.05)
        w_dom = st.slider("Domain weight", 0.0, 1.0, float(config.w_domain), 0.05)

        st.divider()
        st.subheader("LLM")
        providers = ["vertex"]
        default_provider_idx = (
            providers.index(config.llm_provider)
            if config.llm_provider in providers
            else 0
        )
        provider = st.selectbox("Provider", options=providers, index=default_provider_idx)
        default_model = config.gemini_model
        model = st.text_input("Model", value=default_model)
        show_pure_llm = st.checkbox("Also show pure LLM answer", value=True)

    if abs((w_vec + w_kw + w_dom) - 1.0) > 1e-6:
        st.warning("Weights should sum to 1.0 for clean interpretation.")

    missing = []
    if not config.csv_path.exists():
        missing.append("data/Ghana_Election_Result.csv")
    if not config.pdf_path.exists():
        missing.append("data/2025_Budget_Statement.pdf")

    if missing:
        st.error("Missing required data file(s): " + ", ".join(missing))
        st.info("Place the provided files into the data/ folder, then refresh.")
        st.stop()

    docs, embedder, build_indexes = build_pipeline(config)

    st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
    overview_cols = st.columns([1, 1, 1])
    overview_cols[0].metric("Sources", len(docs))
    overview_cols[1].metric("Chunking", chunking_strategy.title())
    overview_cols[2].metric("Top K", top_k)

    with st.expander("Loaded sources", expanded=False):
        st.write(f"Loaded {len(docs)} documents")
        for d in docs[:10]:
            st.write({"source": d.source, "meta": d.metadata})
        if len(docs) > 10:
            st.write("(showing first 10)")

    st.markdown('<div class="section-header">Query</div>', unsafe_allow_html=True)
    query = st.text_area(
        "",
        placeholder="Ask something like: What are the top budget priorities in the 2025 statement?",
        height=120,
    )
    action_col, clear_col = st.columns([1, 1])
    with action_col:
        retrieve = st.button("Retrieve", type="primary", disabled=not query)
    with clear_col:
        clear = st.button("Clear")

    if "retrieval" not in st.session_state:
        st.session_state.retrieval = None

    if retrieve:
        with st.spinner("Indexing + retrieving..."):
            chunks, vstore, kstore = build_indexes(chunking_strategy)

            q_emb = embedder.embed_texts([query], normalize=True)[0]
            retrieved = hybrid_retrieve(
                query=query,
                query_embedding=q_emb,
                vector_store=vstore,
                keyword_store=kstore,
                top_k=top_k,
            )

            reranked = rerank(
                query=query,
                candidates=retrieved,
                w_vector=w_vec,
                w_keyword=w_kw,
                w_domain=w_dom,
            )

            chunk_sources = [ch.source for ch in chunks]
            reranked = apply_domain_scores(
                query=query,
                candidates=reranked,
                chunk_sources=chunk_sources,
                w_vector=w_vec,
                w_keyword=w_kw,
                w_domain=w_dom,
            )

            st.session_state.retrieval = {
                "chunks": chunks,
                "candidates": reranked,
            }

    if clear:
        st.session_state.retrieval = None

    if st.session_state.retrieval:
        chunks: List = st.session_state.retrieval["chunks"]
        candidates: List = st.session_state.retrieval["candidates"]

        st.markdown('<div class="section-header">Retrieved chunks</div>', unsafe_allow_html=True)
        labels_by_idx = {}
        for c in candidates:
            chunk = chunks[c.chunk_idx]
            labels_by_idx[c.chunk_idx] = (
                f"{c.final_score:.3f} | {chunk.source} | {chunk.metadata.get('label', '')}"
            )

        option_idxs = list(labels_by_idx.keys())
        selected_chunk_idxs = st.multiselect(
            "Select chunks to include as context",
            options=option_idxs,
            default=option_idxs[: min(5, len(option_idxs))],
            format_func=lambda idx: labels_by_idx.get(idx, str(idx)),
        )

        for c in candidates[: min(10, len(candidates))]:
            chunk = chunks[c.chunk_idx]
            with st.expander(
                f"Score {c.final_score:.3f} · vec={c.vector_score:.3f} kw={c.keyword_score:.3f} dom={c.domain_score:.3f} · {chunk.source}",
                expanded=False,
            ):
                st.write(chunk.metadata)
                st.write(chunk.text)

        selected_chunks = [chunks[i] for i in selected_chunk_idxs]
        prompt = build_prompt(
            question=query,
            chunks=selected_chunks,
            max_context_tokens=max_context_tokens,
        )

        st.markdown('<div class="section-header">Final prompt</div>', unsafe_allow_html=True)
        st.code(prompt, language="markdown")

        if st.button("Generate answer", disabled=not selected_chunks):
            with st.spinner("Calling LLM..."):
                answer = generate_answer(provider=provider, model=model, prompt=prompt)

            st.markdown('<div class="section-header">Answer · RAG</div>', unsafe_allow_html=True)
            st.write(answer)

            pure_llm_answer = None
            if show_pure_llm:
                pure_prompt = f"Answer the question as best you can.\n\nQuestion: {query}"
                pure_llm_answer = generate_answer(
                    provider=provider, model=model, prompt=pure_prompt
                )
                st.markdown(
                    '<div class="section-header">Answer · Pure LLM baseline</div>',
                    unsafe_allow_html=True,
                )
                st.write(pure_llm_answer)

            event = {
                "query": query,
                "chunking_strategy": chunking_strategy,
                "top_k": top_k,
                "weights": {"vector": w_vec, "keyword": w_kw, "domain": w_dom},
                "retrieved": [
                    {
                        "chunk_idx": c.chunk_idx,
                        "chunk_id": chunks[c.chunk_idx].chunk_id,
                        "source": chunks[c.chunk_idx].source,
                        "metadata": chunks[c.chunk_idx].metadata,
                        "text": chunks[c.chunk_idx].text,
                        "scores": asdict(c),
                    }
                    for c in candidates[:top_k]
                ],
                "selected_chunk_idxs": selected_chunk_idxs,
                "selected_chunks": [c.to_dict() for c in selected_chunks],
                "final_prompt": prompt,
                "answer_rag": answer,
                "answer_pure_llm": pure_llm_answer,
            }
            logger.log(event)
            st.success("Logged to logs/rag_logs.jsonl")

        st.divider()
        st.markdown('<div class="section-header">Recent logs</div>', unsafe_allow_html=True)
        tail = logger.tail(n=3)
        if tail:
            st.json(tail)
        else:
            st.info("No logs yet.")


if __name__ == "__main__":
    main()