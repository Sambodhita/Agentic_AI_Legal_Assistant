# ⚖️ Legal Document Assistant - Agentic AI Capstone

## 📌 Problem Statement

Reading large volumes of case documents is time-consuming for legal professionals. This project is an intelligent patient/legal assistant built for paralegals and junior lawyers. It answers queries exclusively from uploaded legal documents (NDAs, contracts, leases) and refuses to hallucinate binding legal advice or facts outside the context.

## ⚙️ Tech Stack

- **Framework:** LangGraph (StateGraph architecture)
- **LLM:** Groq
- **Vector Store:** ChromaDB
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)
- **UI:** Streamlit
- **Tools:** DuckDuckGo Web Search (`ddgs`)

## 🚀 Features

- **RAG-Grounded Answers:** Strictly answers from the 10 provided legal documents.
- **Self-Reflection Loop:** Evaluates its own answers for faithfulness; automatically retries if it hallucinates.
- **Multi-Tool Routing:** Routes standard queries to the legal text, and general definition requests (e.g., "What is a tort?") to a live web search.
- **Persistent Memory:** Remembers conversation history using LangGraph's `MemorySaver`.

## 🛠️ How to Run Locally

1. Clone this repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Create a `.env` file and add your Groq API key: `GROQ_API_KEY=gsk_your_key_here`
4. Run the Streamlit app: `streamlit run capstone_streamlit.py`

## Live App - **https://bettercallsamb.streamlit.app/**
## 🧪 Example Questions to Try

To test the system's routing and retrieval capabilities, try asking these questions in the Streamlit UI:

**1. Document Retrieval (RAG)**
* *"What are the payment terms for the independent contractor?"*
* *"How long does the non-compete clause last after termination?"*
*(Expected behavior: The agent routes to `retrieve`, pulls from the JSON knowledge base, and answers strictly from the text).*

**2. Web Search Tool (DuckDuckGo)**
* *"What is the legal definition of a tort?"*
* *"Explain the concept of habeas corpus."*
*(Expected behavior: The agent routes to `tool`, searches the live web, and provides a general definition, bypassing the contract knowledge base).*

**3. Contextual Memory**
* *Turn 1: "What is the non-compete radius for an employee?"*
* *Turn 2: "Does that apply to independent contractors too?"*
*(Expected behavior: The agent remembers the specific clause discussed in the previous turn to answer the follow-up question).*

## 📊 Evaluation and Quality Gating

This system incorporates a strict zero-hallucination policy using self-reflective evaluation:
* **Self-Correction Node:** Every generated answer is intercepted by an `eval_node` before reaching the user. The LLM acts as a judge, scoring the answer's faithfulness to the retrieved context as either `1.0` (Pass) or `0.0` (Fail).
* **Retry Mechanism:** If the answer contains hallucinations or outside facts, the graph rejects it and loops back to the generation node for a rewrite. It is capped at two retries to prevent infinite loops.
* **Baseline Testing:** The architecture was evaluated against an adversarial test suite covering out-of-scope queries, false premises, and specific contract clauses to ensure high faithfulness scores on valid queries and safe fallbacks ("I don't have that information") on missing data.
