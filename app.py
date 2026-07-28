import streamlit as st
from graph import app

st.set_page_config(page_title="SME Business-Ops Copilot")
st.title("SME Business-Ops Copilot")
st.caption("Ask about inventory/production or HR/compliance policy.")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        result = app.invoke({"query": query})
    st.markdown(f"**Routed to:** {result['route']}")
    st.markdown(result["final_answer"])
    with st.expander("Retrieved source chunks"):
        for chunk in result["retrieved_chunks"]:
            st.text(chunk)
