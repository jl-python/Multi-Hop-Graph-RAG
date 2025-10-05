
# Run this in Terminal
# pip install streamlit langchain langchain-community langchain-huggingface chromadb rank-bm25 sentence-transformers spacy pypdf transformers accelerate networkx
# python -m spacy download en_core_web_sm

# Test question:
# "What were the accuracy improvements in move prediction and perplexity achieved by Maia-2 compared to original Maia?"

import os, re, tempfile, networkx as nx, spacy
import streamlit as st
from typing import List
import matplotlib.pyplot as plt


from sentence_transformers import CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="Multi-Hop Graph RAG", layout="wide")
st.title("📚 Multi-Hop Graph RAG — Streamlit Demo")
st.caption("Upload a PDF ➜ Build entity graph ➜ Ask a multi-hop question ➜ See hop breakdown + final answer")

st.markdown(
    """
    <style>
    /* Black slider dot */
    .stSlider [role="slider"] {
        background-color: black !important;
        border: none !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
    }

    /* Show slider value (number) larger and bold */
    .stSlider label + div[data-baseweb="slider"] > div > div > div > div {
        font-size: 16px !important;
        font-weight: bold !important;
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Settings
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_trf = st.checkbox("Use spaCy transformer model (GPU recommended)", value=False)
    use_reranker = st.checkbox("Use Cross-Encoder reranker (slower on CPU)", value=False)

    # Top-K (dropdown 3–10)
    top_k = st.selectbox("Top-K retrieved docs", options=list(range(3, 11)), index=2)
    st.write(f"Currently selected Top-K: **{top_k}**")

    # Hop limit (slider 1–4)
    hop_limit = st.selectbox("Max hops to traverse", options=list(range(1,5)), index=0)
    st.write(f"Currently selected Hop Limit: **{hop_limit}**")


    # Temperature (dropdown 0.20 → 1.00 with 0.05 step)
    temperature = st.selectbox(
        "LLM temperature",
        options=[round(x, 2) for x in [0.20, 0.25, 0.30, 0.35, 0.40,
                                       0.45, 0.50, 0.55, 0.60, 0.65,
                                       0.70, 0.75, 0.80, 0.85, 0.90,
                                       0.95, 1.00]],
        index=0
    )
    st.write(f"Currently selected Temperature: **{temperature}**")

    model_name = st.selectbox(
        "LLM (HF model id)",
        ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "gpt2"],
        index=0
    )
    st.write(f"🤖 Using model: **{model_name}**")


# ----------------------------
# Load Models (cached in session)
# ----------------------------
@st.cache_resource
def load_spacy(use_trf: bool):
    try:
        return spacy.load("en_core_web_trf") if use_trf else spacy.load("en_core_web_sm")
    except:
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="intfloat/e5-large")

@st.cache_resource
def load_reranker():
    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        st.warning(f"CrossEncoder failed: {e}")
        return None

@st.cache_resource
def load_generator(model_name: str, temperature: float):
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name)
        return pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256, do_sample=True, temperature=temperature)
    except:
        st.error(f"Failed to load {model_name}, falling back to GPT-2.")
        tok = AutoTokenizer.from_pretrained("gpt2")
        mdl = AutoModelForCausalLM.from_pretrained("gpt2")
        return pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=128, do_sample=True, temperature=temperature)

nlp = load_spacy(use_trf)
emb = load_embeddings()
reranker = load_reranker() if use_reranker else None
generator = load_generator(model_name, temperature)

# ----------------------------
# Helpers
# ----------------------------
def read_uploaded_pdf(file) -> List[str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp.flush()
        path = tmp.name
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    return texts, chunks

def build_in_memory_chroma(chunks):
    return Chroma.from_documents(documents=chunks, embedding=emb, collection_name="tmp_collection")

def build_weighted_entity_graph(texts: List[str]) -> nx.Graph:
    G = nx.Graph()
    for i, text in enumerate(texts):
        doc = nlp(text)
        ents = [ent.text for ent in doc.ents]
        for ent in ents:
            G.add_node(ent)
        for e1 in ents:
            for e2 in ents:
                if e1 == e2:
                    continue
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1, chunk_id=i)
    return G

# ----------------------------
# Question Decomposition (Track B style)
# ----------------------------
def decompose(query: str) -> List[str]:
    """
    Decompose multi-hop questions into sub-questions.
    First try rule-based, then optionally fall back to LLM.
    """
    q = query.lower().strip()

    # --- Rule-based examples ---
    if "introduced" in q and "method" in q and "dataset" in q:
        return [
            "Which paper introduced the method?",
            "Which dataset did that paper use?"
        ]
    if "who" in q and "when" in q:
        return [
            "Who is the subject?",
            "When did the event occur?"
        ]

    # Generic split heuristics
    if " and " in q:
        return [p.strip() for p in re.split(r"\band\b", query, flags=re.I) if p.strip()]
    if " then " in q:
        return [p.strip() for p in query.split(" then ") if p.strip()]

    # --- Optional: fallback to LLM ---
    try:
        prompt = f"Break this question into smaller sub-questions:\n\n{query}\n\nSub-questions:"
        out = generator(prompt)
        text = out[0]["generated_text"]
        subs = [line.strip("-• ").strip() for line in text.split("\n") if line.strip()]
        if subs:
            return subs
    except Exception as e:
        st.warning(f"LLM decomposition failed, using full query. ({e})")

    return [query]



def generate_answer(query: str, context: str) -> str:
    # Build prompt (hidden from user output)
    prompt = (
        "Answer the question based only on the context. "
        "If the answer cannot be found, say you cannot answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    out = generator(prompt)
    raw_text = out[0]["generated_text"]

    # Strip away everything before "Answer:" (so user only sees the actual answer)
    if "Answer:" in raw_text:
        answer = raw_text.split("Answer:", 1)[-1].strip()
    else:
        answer = raw_text.strip()

    return answer



def graph_rag(query: str, chunk_texts: List[str], vectordb: Chroma, G: nx.Graph, top_k: int = 5, multi_hop: bool = True, hop_limit: int = 2):
    doc = nlp(query)
    q_ents = [ent.text for ent in doc.ents]
    candidate_entities = set()

    for ent in q_ents:
        if ent in G:
            neighbors = sorted(G.neighbors(ent), key=lambda n: G[ent][n]["weight"], reverse=True)
            candidate_entities.update(neighbors[:5])
            if multi_hop:
                for n in neighbors[:hop_limit]:
                    candidate_entities.update(sorted(G.neighbors(n), key=lambda x: G[n][x]["weight"], reverse=True)[:hop_limit])


    matched_texts = [t for t in chunk_texts if any(ent in t for ent in candidate_entities)]
    dense_docs = vectordb.similarity_search(query, k=top_k * 3)
    candidates = dense_docs if not candidate_entities else [d for d in dense_docs if any(ent in d.page_content for ent in candidate_entities)] or dense_docs

    if reranker:
        pairs = [(query, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)
        candidates = [doc for doc, _ in sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)]

    return candidates[:top_k], list(candidate_entities)

# ----------------------------
# UI
# ----------------------------
uploaded_pdf = st.file_uploader("📄 Upload a PDF", type=["pdf"])
if uploaded_pdf:
    st.info("Processing document...")
    texts, chunks = read_uploaded_pdf(uploaded_pdf)
    vectordb = build_in_memory_chroma(chunks)
    G = build_weighted_entity_graph(texts)
    st.success(f"Built index: {len(texts)} chunks | Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

    # ----------------------------
    # Graph Visualization
    # ----------------------------
    with st.expander("📊 View Entity Graph"):
        sub_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:30]
        H = G.subgraph([n for n, _ in sub_nodes])
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(H, seed=42, k=1.2)
        sizes = [300 + 50*H.degree(n) for n in H.nodes]
        nx.draw(H, pos, ax=ax,
                with_labels=True,
                node_size=sizes,
                font_size=8,
                font_color="black",
                node_color="skyblue",
                edge_color="gray")
        st.pyplot(fig)



    query = st.text_input("🔎 Ask a multi-hop question")
    multi_hop_enabled = st.checkbox("Enable multi-hop", value=True)
    show_context = st.checkbox("Show retrieved context", value=False)

    if st.button("Run") and query:
        subs = decompose(query)
        st.markdown("**Question decomposition:**")
        for i, s in enumerate(subs, 1):
            st.write(f"• Hop {i}: {s}")

        docs, entities = graph_rag(query, texts, vectordb, G, top_k=top_k, multi_hop=multi_hop_enabled, hop_limit=hop_limit)


        if entities:
            st.markdown("**Entities considered:** " + ", ".join(sorted(entities)))

        context = "\n\n".join([d.page_content for d in docs])
        answer = generate_answer(query, context)

        st.subheader("💡 Final Answer")
        st.write(answer)

        if show_context:
            with st.expander("Retrieved Context"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.write(d.page_content)
else:
    st.info("👆 Upload a PDF to begin.")