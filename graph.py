try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # not needed locally on Windows — only kicks in on Streamlit Cloud's Linux environment

import os
import time
from typing import TypedDict, Literal
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
]
SYNTHESIS_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
]
REFLECTION_MODELS = [
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "google/gemma-4-26b-a4b-it:free",
    "inclusionai/ling-3.0-flash:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

def classify(query: str) -> str:
    system_prompt = (
        "You are a routing classifier for a business assistant. "
        "Classify the user's query into exactly one category: "
        "'inventory' (stock levels, production, manufacturing) or "
        "'compliance' (HR policy, EPF, ETF, labour law). "
        "Respond with only the single word: inventory or compliance."
    )
    for model in FREE_MODELS:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    temperature=0,
                    max_tokens=20,
                )
                content = response.choices[0].message.content
                raw = (content or "").strip().lower()
                print(f"[classify:{model}] raw={raw!r}")
                has_inv, has_comp = "inventory" in raw, "compliance" in raw
                if has_inv and not has_comp:
                    return "inventory"
                elif has_comp and not has_inv:
                    return "compliance"
                break
            except Exception as e:
                print(f"[classify:{model}] attempt {attempt+1} FAILED: {type(e).__name__}: {e}")
                time.sleep(2)
    return "compliance"

def chunk_text(text: str, max_chars: int = 500) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) < max_chars:
            current += ("\n\n" + p if current else p)
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks

def load_and_chunk_domain(domain: str) -> list[dict]:
    import glob
    records = []
    for filepath in glob.glob(f"documents/{domain}/*.md") + glob.glob(f"documents/{domain}/*.txt"):
        with open(filepath, "r") as f:
            text = f.read()
        for i, chunk in enumerate(chunk_text(text)):
            records.append({"id": f"{filepath}-{i}", "text": chunk, "source": filepath})
    return records

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
inventory_collection = chroma_client.get_or_create_collection(name="inventory", embedding_function=embedding_fn)
compliance_collection = chroma_client.get_or_create_collection(name="compliance", embedding_function=embedding_fn)

def add_chunks_to_collection(collection, chunks):
    if chunks:
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
        )

add_chunks_to_collection(inventory_collection, load_and_chunk_domain("inventory"))
add_chunks_to_collection(compliance_collection, load_and_chunk_domain("compliance"))
print(f"[startup] inventory chunks loaded: {inventory_collection.count()}")
print(f"[startup] compliance chunks loaded: {compliance_collection.count()}")

def retrieve(query: str, domain: str, k: int = 3) -> list[str]:
    collection = inventory_collection if domain == "inventory" else compliance_collection
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]

def synthesize(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(retrieved_chunks)
    system_prompt = (
        "You are a business assistant answering questions using only "
        "the provided context. If the context doesn't contain enough "
        "information to answer confidently, say so explicitly rather than guessing."
    )
    for model in SYNTHESIS_MODELS:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                    ],
                    temperature=0.2,
                    max_tokens=400,
                )
                content = response.choices[0].message.content
                answer = (content or "").strip()
                print(f"[synthesize:{model}] got answer: {bool(answer)}")
                if answer:
                    return answer
            except Exception as e:
                print(f"[synthesize:{model}] attempt {attempt+1} FAILED: {type(e).__name__}: {e}")
                time.sleep(2)
    return "Unable to generate an answer right now — all free-tier models were unavailable. Please try again shortly."

def reflect(draft_answer: str, retrieved_chunks: list[str]) -> dict:
    context = "\n\n---\n\n".join(retrieved_chunks)
    system_prompt = (
        "You are a fact-checker. You will be given CONTEXT and a DRAFT ANSWER. "
        "Determine whether the draft answer is fully supported by the context. "
        "Respond with only one word: SUPPORTED if every claim in the draft "
        "answer is backed by the context, or UNSUPPORTED if the draft answer "
        "contains any claim, number, or detail not found in the context."
    )
    for model in REFLECTION_MODELS:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nDRAFT ANSWER:\n{draft_answer}"},
                    ],
                    temperature=0,
                    max_tokens=20,
                )
                content = response.choices[0].message.content
                verdict = (content or "").strip().upper()
                print(f"[reflect:{model}] verdict={verdict!r}")
                if "UNSUPPORTED" in verdict:
                    return {"needs_review": True, "verdict": "unsupported"}
                elif "SUPPORTED" in verdict:
                    return {"needs_review": False, "verdict": "supported"}
            except Exception as e:
                print(f"[reflect:{model}] attempt {attempt+1} FAILED: {type(e).__name__}: {e}")
                time.sleep(2)
    return {"needs_review": True, "verdict": "reflection_unavailable"}

class AgentState(TypedDict):
    query: str
    route: str
    retrieved_chunks: list
    draft_answer: str
    needs_review: bool
    verdict: str
    final_answer: str

def router_node(state: AgentState) -> AgentState:
    state["route"] = classify(state["query"])
    return state

def inventory_node(state: AgentState) -> AgentState:
    chunks = retrieve(state["query"], domain="inventory")
    state["retrieved_chunks"] = chunks
    state["draft_answer"] = synthesize(state["query"], chunks)
    return state

def compliance_node(state: AgentState) -> AgentState:
    chunks = retrieve(state["query"], domain="compliance")
    state["retrieved_chunks"] = chunks
    state["draft_answer"] = synthesize(state["query"], chunks)
    return state

def reflect_node(state: AgentState) -> AgentState:
    review = reflect(state["draft_answer"], state["retrieved_chunks"])
    state["needs_review"] = review["needs_review"]
    state["verdict"] = review["verdict"]
    if state["needs_review"]:
        state["final_answer"] = state["draft_answer"] + "\n\n⚠️ Note: this answer could not be fully verified against the source documents."
    else:
        state["final_answer"] = state["draft_answer"]
    return state

def route_decision(state: AgentState) -> Literal["inventory", "compliance"]:
    return state["route"]

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("inventory", inventory_node)
graph.add_node("compliance", compliance_node)
graph.add_node("reflect", reflect_node)
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_decision, {"inventory": "inventory", "compliance": "compliance"})
graph.add_edge("inventory", "reflect")
graph.add_edge("compliance", "reflect")
graph.add_edge("reflect", END)

app = graph.compile()