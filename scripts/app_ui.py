"""
ScholarLens Web UI - Streamlit-based academic paper QA interface.

Run from repo root:
  streamlit run scripts/app_ui.py
"""

import sys
import re
from csv import DictReader
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="ScholarLens AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_index(persist_dir: str):
    """Load and cache the vector store index."""
    from llama_index.core import StorageContext, load_index_from_storage
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)


@st.cache_resource
def apply_settings(llm_model: str, embed_model: str):
    """Apply Ollama settings once."""
    from scholarlens.ollama_config import apply_ollama_settings
    apply_ollama_settings(llm_model=llm_model, embed_model=embed_model, chunk_size=512, chunk_overlap=50)
    return True


def get_chat_engine(index, top_k: int):
    """Create a chat engine from the index."""
    from llama_index.core.memory import ChatMemoryBuffer
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        similarity_top_k=top_k,
        system_prompt=(
            "You are ScholarLens, an expert academic AI assistant specialized in NLP research papers. "
            "Answer based on indexed papers and course materials only. "
            "If asked about RAG, interpret it as Retrieval-Augmented Generation unless user says otherwise. "
            "If evidence is insufficient, say so clearly and avoid guessing. "
            "Cite paper names or paper_id when relevant."
        ),
    )


def _list_paper_ids() -> list[str]:
    manifest_path = ROOT / "data" / "papers" / "manifest.csv"
    if not manifest_path.is_file():
        return []
    paper_ids: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        for row in DictReader(file):
            paper_id = (row.get("paper_id") or "").strip()
            if paper_id:
                paper_ids.append(paper_id)
    return sorted(set(paper_ids))


def _build_filters(focus_paper_id: str | None):
    if not focus_paper_id:
        return None
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

    return MetadataFilters(
        filters=[ExactMatchFilter(key="paper_id", value=focus_paper_id)]
    )


def get_filtered_chat_engine(index, top_k: int, focus_paper_id: str | None):
    from llama_index.core.memory import ChatMemoryBuffer

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    filters = _build_filters(focus_paper_id)
    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        similarity_top_k=top_k,
        filters=filters,
        system_prompt=(
            "You are ScholarLens, an expert academic AI assistant specialized in NLP research papers. "
            "Answer based on indexed papers and course materials only. "
            "If asked about RAG, interpret it as Retrieval-Augmented Generation unless user says otherwise. "
            "If evidence is insufficient, say so clearly and avoid guessing. "
            "Cite paper names or paper_id when relevant."
        ),
    )


def _normalize_prompt(prompt: str) -> str:
    if re.search(r"\brag\b", prompt, flags=re.IGNORECASE) and not re.search(
        r"retrieval[- ]augmented generation", prompt, flags=re.IGNORECASE
    ):
        return f"{prompt} (RAG means Retrieval-Augmented Generation in NLP.)"
    return prompt


def _infer_focus_paper_from_prompt(prompt: str) -> str | None:
    """Infer a default paper focus for ambiguous acronym questions."""
    lowered = prompt.lower()
    if "self-rag" in lowered:
        return "asai2024_selfrag"
    if re.search(r"\brag\b", lowered) and "lewis" not in lowered:
        return "lewis2020_rag"
    return None


def _is_unsure_answer(answer: str) -> bool:
    patterns = (
        r"no information",
        r"does not mention",
        r"not possible to determine",
        r"cannot determine",
        r"insufficient information",
        r"hypothetical",
    )
    joined_patterns = "|".join(patterns)
    return bool(re.search(joined_patterns, answer, flags=re.IGNORECASE))


def _get_source_metadata(source_node) -> dict:
    if hasattr(source_node, "metadata") and isinstance(source_node.metadata, dict):
        return source_node.metadata or {}
    if hasattr(source_node, "node") and hasattr(source_node.node, "metadata"):
        node_metadata = source_node.node.metadata
        if isinstance(node_metadata, dict):
            return node_metadata
    return {}


def _source_label(metadata: dict) -> str:
    return (
        metadata.get("paper_id")
        or metadata.get("file_name")
        or metadata.get("source_url")
        or metadata.get("source_type")
        or "unknown source"
    )


_PDF_BINARY_PATTERNS = (
    r"/Filter",
    r"/FlateDecode",
    r"\bstream\b",
    r"\bendstream\b",
    r"\bxref\b",
    r"\btrailer\b",
    r"%PDF-",
)


def _clean_source_preview(text: str, max_len: int = 500) -> tuple[str, bool]:
    if not text:
        return "", False

    text = text.replace("\uFFFD", " ").replace("\x00", " ")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"[\uD800-\uDFFF]", " ", text)
    text = re.sub(r"[^\w\s]{5,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    joined_patterns = "|".join(_PDF_BINARY_PATTERNS)
    if re.search(joined_patterns, text, flags=re.IGNORECASE):
        return "", False

    if len(text) < 30:
        return text, False

    non_ascii = sum(1 for ch in text if ord(ch) > 126)
    non_ascii_ratio = non_ascii / max(len(text), 1)
    if non_ascii_ratio > 0.25:
        ascii_text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
        ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
        if len(ascii_text) < 30:
            return "", False
        return ascii_text[:max_len], True

    return text[:max_len], True


def _is_quality_source(metadata: dict, text: str) -> int:
    score = 0
    if metadata.get("paper_id"):
        score += 100
    if metadata.get("file_name"):
        score += 50
    if metadata.get("source_url"):
        score += 30
    if len(text) < 50:
        score -= 50
    return score


def _extract_sources(response) -> list[dict]:
    sources: list[dict] = []
    if not hasattr(response, "source_nodes"):
        return sources

    raw_sources: list[dict] = []
    for node in response.source_nodes[:8]:
        metadata = _get_source_metadata(node)
        raw_text = node.text if hasattr(node, "text") else str(node)
        clean_text, usable = _clean_source_preview(raw_text)
        quality = _is_quality_source(metadata, clean_text)
        raw_sources.append(
            {
                "score": node.score if hasattr(node, "score") else 0.0,
                "clean_text": clean_text,
                "label": _source_label(metadata),
                "usable": usable,
                "quality": quality,
            }
        )

    raw_sources.sort(key=lambda item: (-item["quality"], -item["score"]))
    return [item for item in raw_sources if item["usable"]][:3]


def main():
    st.title("📚 ScholarLens")
    st.caption("Academic Paper QA with RAG - CS6493 Project")

    with st.sidebar:
        st.header("⚙️ Settings")

        available_indices = []
        storage_path = ROOT / "storage"
        if storage_path.exists():
            available_indices = [d.name for d in storage_path.iterdir() if d.is_dir() and (d / "docstore.json").exists()]

        if not available_indices:
            st.error("No index found. Run `build_index.py` first.")
            st.stop()

        selected_index = st.selectbox(
            "Select Index",
            available_indices,
            index=available_indices.index("index_512") if "index_512" in available_indices else 0,
        )

        top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=5)

        llm_model = st.selectbox(
            "LLM Model",
            ["mistral", "llama3.2", "qwen2.5:0.5b"],
            index=0,
        )
        paper_ids = _list_paper_ids()
        focus_option = st.selectbox(
            "Focus Paper (optional)",
            ["All papers"] + paper_ids,
            index=0,
        )
        focus_paper_id = None if focus_option == "All papers" else focus_option

        st.divider()
        st.subheader("📊 Experiment Results")
        st.markdown("""
        | Config | Score |
        |--------|-------|
        | Chunk 256 | 3.03 |
        | **Chunk 512** | **3.47** |
        """)

        st.divider()
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_engine = None
            st.session_state.current_index = None
            st.session_state.current_llm = None
            st.session_state.current_top_k = None
            st.session_state.current_focus_paper = None
            st.rerun()

    persist_dir = str(ROOT / "storage" / selected_index)

    with st.spinner("Loading model settings..."):
        apply_settings(llm_model, "nomic-embed-text")

    with st.spinner(f"Loading index: {selected_index}..."):
        index = load_index(persist_dir)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    need_new_engine = (
        "chat_engine" not in st.session_state
        or st.session_state.get("chat_engine") is None
        or st.session_state.get("current_index") != selected_index
        or st.session_state.get("current_llm") != llm_model
        or st.session_state.get("current_top_k") != top_k
        or st.session_state.get("current_focus_paper") != focus_paper_id
    )
    if need_new_engine:
        st.session_state.chat_engine = get_filtered_chat_engine(index, top_k, focus_paper_id)
        st.session_state.current_index = selected_index
        st.session_state.current_llm = llm_model
        st.session_state.current_top_k = top_k
        st.session_state.current_focus_paper = focus_paper_id

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                usable_sources = [item for item in message["sources"] if item.get("usable", True)]
                if usable_sources:
                    with st.expander("📄 View Sources", expanded=False):
                        for i, src in enumerate(usable_sources, 1):
                            st.info(
                                f"**Source {i}** (Score: {src['score']:.3f})\n"
                                f"From: `{src.get('label', 'unknown source')}`\n\n"
                                f"{src.get('clean_text', '')}..."
                            )

    if prompt := st.chat_input("Ask about RAG, Self-RAG, or NLP papers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    normalized_prompt = _normalize_prompt(prompt)
                    inferred_focus = None
                    if focus_paper_id is None:
                        inferred_focus = _infer_focus_paper_from_prompt(prompt)

                    if inferred_focus:
                        primary_engine = get_filtered_chat_engine(index, top_k, inferred_focus)
                        response = primary_engine.chat(normalized_prompt)
                    else:
                        response = st.session_state.chat_engine.chat(normalized_prompt)
                    answer = str(response)
                    # Retry once with a paper-focused filter when the query is about RAG
                    # and the first attempt fails to retrieve useful evidence.
                    if (
                        re.search(r"\brag\b", prompt, flags=re.IGNORECASE)
                        and focus_paper_id is None
                        and _is_unsure_answer(answer)
                    ):
                        retry_engine = get_filtered_chat_engine(index, top_k, "lewis2020_rag")
                        retry_response = retry_engine.chat(normalized_prompt)
                        retry_answer = str(retry_response)
                        if not _is_unsure_answer(retry_answer):
                            response = retry_response
                            answer = retry_answer

                    sources = _extract_sources(response)

                except Exception as e:
                    answer = f"Error: {str(e)}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("📄 View Sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.info(
                            f"**Source {i}** (Score: {src['score']:.3f})\n"
                            f"From: `{src.get('label', 'unknown source')}`\n\n"
                            f"{src.get('clean_text', '')}..."
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

    with st.sidebar:
        st.divider()
        st.caption("ScholarLens v1.0 | CS6493 NLP Project")


if __name__ == "__main__":
    main()
