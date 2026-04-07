"""
ScholarLens Web UI - Streamlit-based academic paper QA interface.

Run from repo root:
  streamlit run scripts/app_ui.py
"""

import sys
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
            "Answer questions based on the indexed papers and course materials. "
            "Be concise but thorough. Cite specific papers when relevant."
        ),
    )


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

        llm_model = st.selectbox("LLM Model", ["mistral", "llama3.2", "qwen2.5"], index=0)

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
            st.rerun()

    persist_dir = str(ROOT / "storage" / selected_index)

    with st.spinner("Loading model settings..."):
        apply_settings(llm_model, "nomic-embed-text")

    with st.spinner(f"Loading index: {selected_index}..."):
        index = load_index(persist_dir)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_engine" not in st.session_state or st.session_state.get("current_index") != selected_index:
        st.session_state.chat_engine = get_chat_engine(index, top_k)
        st.session_state.current_index = selected_index

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Sources", expanded=False):
                    for i, src in enumerate(message["sources"], 1):
                        st.info(f"**Source {i}** (Score: {src['score']:.3f})\n\n{src['text'][:500]}...")

    if prompt := st.chat_input("Ask about RAG, Self-RAG, or NLP papers..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_engine.chat(prompt)
                    answer = str(response)

                    sources = []
                    if hasattr(response, "source_nodes"):
                        for node in response.source_nodes[:3]:
                            sources.append({
                                "score": node.score if hasattr(node, "score") else 0.0,
                                "text": node.text if hasattr(node, "text") else str(node),
                            })

                except Exception as e:
                    answer = f"Error: {str(e)}"
                    sources = []

            st.markdown(answer)

            if sources:
                with st.expander("📄 View Sources", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.info(f"**Source {i}** (Score: {src['score']:.3f})\n\n{src['text'][:500]}...")

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
