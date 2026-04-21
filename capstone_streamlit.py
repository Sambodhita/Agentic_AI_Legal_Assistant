import streamlit as st
import uuid
from agent import app, kb_topics # Importing your modularized graph!

st.set_page_config(page_title="Legal Document Assistant", page_icon="⚖️", layout="centered")
st.title("⚖️ Legal Document Assistant")
st.caption("An AI assistant for paralegals to quickly query case files and legal definitions.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.header("About")
    st.write("Analyzes contracts, NDAs, and leases safely without hallucination.")
    st.write(f"**Session ID:** {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics loaded in KB:**")
    for t in kb_topics:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.write(msg["content"])

if prompt := st.chat_input("Ask a legal question..."):
    with st.chat_message("user"): 
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing case files..."):
            # Execute the LangGraph from agent.py
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Error generating response.")
            
        st.write(answer)
        
        # Display agent self-reflection data
        if result.get("faithfulness", 0.0) > 0:
            st.caption(f"Route: {result.get('route')} | Faithfulness: {result.get('faithfulness'):.2f} | Sources: {result.get('sources', [])}") 
            
    st.session_state.messages.append({"role": "assistant", "content": answer})