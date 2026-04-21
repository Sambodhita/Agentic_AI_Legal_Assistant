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
