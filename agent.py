import os
import json
import chromadb
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ── 1. Setup Models & ChromaDB ──
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Dynamically load the documents from the JSON file
json_path = os.path.join("data", "legal_docs.json")
try:
    with open(json_path, "r", encoding="utf-8") as file:
        DOCUMENTS = json.load(file)
except FileNotFoundError:
    print(f"Error: Could not find {json_path}. Please ensure the file exists.")
    DOCUMENTS = []

client = chromadb.Client()
# Bumped to v3 to ensure it reads the new JSON structure properly
collection_name = "legal_kb_production_v3" 
try:
    client.delete_collection(collection_name)
except Exception:
    pass
collection = client.create_collection(collection_name)

if DOCUMENTS:
    # Mapping your JSON fields: "content", "id", "title", "category"
    texts = [d["content"] for d in DOCUMENTS]
    ids   = [d["id"]   for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=ids,
        metadatas=[{"title": d["title"], "category": d["category"]} for d in DOCUMENTS]
    )

# ── 2. State & Nodes ──
class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", []) + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6: msgs = msgs[-6:]
    return {"messages": msgs}

def router_node(state: CapstoneState) -> dict:
    q = state["question"]
    recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in state.get("messages", [])[-3:-1]) or "none"
    prompt = f"""You are a router for a legal document assistant.
Options:
- retrieve: search the knowledge base for specific contract clauses or lease terms.
- memory_only: answer based on conversation history.
- tool: use the web search tool ONLY for general legal definitions outside the contracts.
Recent: {recent}
Current: {q}
Reply ONE word: retrieve / memory_only / tool"""
    decision = llm.invoke(prompt).content.strip().lower()
    if "memory" in decision: decision = "memory_only"
    elif "tool" in decision: decision = "tool"
    else: decision = "retrieve"
    return {"route": decision}

def retrieval_node(state: CapstoneState) -> dict:
    q_emb = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks = results["documents"][0]
    
    # Updated to extract the "title" from your JSON structure
    topics = [m["title"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}

def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}

def tool_node(state: CapstoneState) -> dict:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(state["question"], max_results=3))
        res = "\n".join(f"{r['title']}: {r['body'][:200]}" for r in results)
    except Exception as e:
        res = f"Search error: {e}"
    return {"tool_result": res}

def answer_node(state: CapstoneState) -> dict:
    ctx_parts = []
    if state.get("retrieved"): ctx_parts.append(f"KB:\n{state['retrieved']}")
    if state.get("tool_result"): ctx_parts.append(f"TOOL:\n{state['tool_result']}")
    ctx = "\n\n".join(ctx_parts)
    
    sys_msg = f"You are a Legal Assistant. Answer ONLY from context. No legal advice.\n{ctx}" if ctx else "Answer from memory."
    if state.get("eval_retries", 0) > 0: sys_msg += "\nONLY use explicitly stated context."
    
    msgs = [SystemMessage(content=sys_msg)]
    for m in state.get("messages", [])[:-1]:
        msgs.append(HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]))
    msgs.append(HumanMessage(content=state["question"]))
    return {"answer": llm.invoke(msgs).content}

def eval_node(state: CapstoneState) -> dict:
    ans, ctx, retries = state.get("answer", ""), state.get("retrieved", "")[:2000], state.get("eval_retries", 0)
    if not ctx: return {"faithfulness": 1.0, "eval_retries": retries + 1}
    prompt = f"Is the Answer supported by the Context?\nReply with ONLY 1.0 for yes, or 0.0 for no.\nContext: {ctx}\nAnswer: {ans[:300]}"
    try:
        res = llm.invoke(prompt).content.strip()
        score = 1.0 if "1.0" in res or "1" in res else (0.0 if "0.0" in res or "0" in res else 0.5)
    except:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}

def save_node(state: CapstoneState) -> dict:
    return {"messages": state.get("messages", []) + [{"role": "assistant", "content": state["answer"]}]}

# ── 3. Graph Assembly ──
def route_dec(state: CapstoneState):
    r = state.get("route", "retrieve")
    return "tool" if r=="tool" else "skip" if r=="memory_only" else "retrieve"

def eval_dec(state: CapstoneState):
    return "save" if state.get("faithfulness", 1.0) >= 0.7 or state.get("eval_retries", 0) >= 2 else "answer"

graph = StateGraph(CapstoneState)
for n, f in [("memory", memory_node), ("router", router_node), ("retrieve", retrieval_node), 
             ("skip", skip_retrieval_node), ("tool", tool_node), ("answer", answer_node), 
             ("eval", eval_node), ("save", save_node)]:
    graph.add_node(n, f)
    
graph.set_entry_point("memory")
graph.add_edge("memory", "router")
graph.add_conditional_edges("router", route_dec, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
for n in ["retrieve", "skip", "tool"]: graph.add_edge(n, "answer")
graph.add_edge("answer", "eval")
graph.add_conditional_edges("eval", eval_dec, {"answer": "answer", "save": "save"})
graph.add_edge("save", END)

# Export the compiled app and topics for the UI
app = graph.compile(checkpointer=MemorySaver())

# Updated to export "title" instead of "topic" to capstone_streamlit.py
kb_topics = [d["title"] for d in DOCUMENTS]