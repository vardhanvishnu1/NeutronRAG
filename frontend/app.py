import streamlit as st
import requests

st.set_page_config(page_title="NeutronRAG", page_icon="⚡")
st.title("NeutronRAG")

with st.sidebar:
    st.header("Upload Knowledge")
    up = st.file_uploader("Upload PDF", type="pdf")
    if up and st.button("Index Data"):
        res = requests.post("http://localhost:8000/upload", files={"file": up.getvalue()})
        st.success(f"Indexed {res.json()['chunks']} chunks!")

st.header("Ask Questions")
query = st.text_input("What would you like to know?")
if query:
    data = requests.get(f"http://localhost:8000/ask?q={query}").json()
    if "answer" in data:
        st.info(f"**Confidence: {data['confidence']}**")
        st.write(data["answer"])
        st.caption(f"Source: {data['source']}")