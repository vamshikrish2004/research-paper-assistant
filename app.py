import streamlit as st

from src.answer import Assistant
from src.retrieval import Retriever, MODES

st.set_page_config(page_title="Research Paper Assistant", page_icon="📚", layout="wide")
st.title("📚 Research Paper Assistant")
st.caption("Ask questions about your PDFs. Every answer cites its sources, and the code checks the citations.")

MODE_LABELS = {
    "vector": "Vector (by meaning)",
    "keyword": "Keyword (BM25)",
    "hybrid": "Hybrid (meaning + keyword)",
    "hybrid_rerank": "Hybrid + re-rank (recommended)",
}


@st.cache_resource
def load_retriever():
    """Load the search models once, not on every message."""
    return Retriever()


retriever = load_retriever()

# Each browser session gets its own chat memory
if "assistant" not in st.session_state:
    st.session_state.assistant = Assistant(retriever)
    st.session_state.messages = []
assistant = st.session_state.assistant

doc_names = sorted({c["doc"] for c in retriever.chunks})

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox(
        "Search method",
        MODES,
        index=MODES.index("hybrid_rerank"),
        format_func=lambda m: MODE_LABELS[m],
    )
    doc_choice = st.selectbox("Search in", ["All PDFs"] + doc_names)
    st.divider()
    st.caption(f"{len(doc_names)} PDFs, {len(retriever.chunks)} chunks loaded")
    if st.button("Clear chat"):
        st.session_state.messages = []
        assistant.history = []
        st.rerun()


def show_details(msg):
    """Show how the answer was produced: rewrite, citation check, source text."""
    if msg.get("searched_for"):
        st.caption(f"Searched for: {msg['searched_for']}")

    result = msg["result"]
    if result["refused"]:
        return

    if result["problems"]:
        st.warning("Citation check: " + "; ".join(result["problems"]))
    else:
        st.caption("✅ Citation check passed")

    for n in result["cited"]:
        hit = result["hits"][n - 1]
        with st.expander(f"[S{n}] {hit['doc']}, page {hit['page']}"):
            st.write(hit["text"])


# ---- Earlier messages ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_details(msg)

# ---- New question ----
question = st.chat_input("Ask a question about your papers")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            doc = None if doc_choice == "All PDFs" else doc_choice
            result = assistant.ask(question, mode=mode, doc=doc)
        msg = {
            "role": "assistant",
            "content": result["answer"],
            "result": result,
            "searched_for": result["search_query"] if result["search_query"] != question else None,
        }
        st.markdown(msg["content"])
        show_details(msg)

    st.session_state.messages.append(msg)