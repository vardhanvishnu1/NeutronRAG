import streamlit as st
import numpy as np
import google.generativeai as genai
from engine import RAGEngine
import sys
import os
import subprocess

# AUTOMATIC C++ COMPILATION
# This ensures the C++ engine is built the first time it runs on the server
if not os.path.exists("core/neutron_math.so"):
    with st.spinner("Compiling C++ Similarity Engine..."):
        try:
            subprocess.run([
                "c++", "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC",
                "-I/usr/include/python3.11", 
                "core/similarity.cpp",
                "-o", "core/neutron_math.so"
            ], check=True)
        except Exception as e:
            st.error(f"C++ Compilation failed: {e}")

# Add the core folder to sys.path so we can import the .so file
sys.path.append(os.path.join(os.getcwd(), "core"))

try:
    import neutron_math 
except ImportError:
    st.error("Could not import neutron_math. Check your C++ compilation.")

# CONFIGURATION
st.set_page_config(page_title="NeutronRAG", page_icon="⚡")
st.title("NeutronRAG")

# Setup Gemini using Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("GEMINI_API_KEY not found in Streamlit Secrets.")

# Initialize shared components once per session
if 'engine' not in st.session_state:
    st.session_state.engine = RAGEngine()
    st.session_state.llm = genai.GenerativeModel("gemini-2.5-flash")
    st.session_state.kb = {"chunks": [], "vecs": [], "name": ""}

# SIDEBAR: INDEXING
with st.sidebar:
    st.header("Upload Knowledge")
    up = st.file_uploader("Upload PDF", type="pdf")
    
    if up and st.button("Index Data"):
        with st.spinner("Processing PDF..."):
            content = up.getvalue()
            # Text Extraction and Chunking
            text = st.session_state.engine.get_text_from_pdf(content)
            chunks = st.session_state.engine.get_chunks(text)
            vecs = st.session_state.engine.create_embeddings(chunks)
            
            # Store in Session State 
            st.session_state.kb = {
                "chunks": chunks, 
                "vecs": vecs, 
                "name": up.name
            }
            st.success(f"Indexed {len(chunks)} chunks!")

# MAIN UI
st.header("Ask Questions")
query = st.text_input("What would you like to know?")

if query:
    if not st.session_state.kb["chunks"]:
        st.warning("Please upload and index a PDF first.")
    else:
        with st.spinner("C++ Engine searching..."):
            # RETRIEVAL (The C++ Part)
            q_vec = st.session_state.engine.model.encode([query])[0].tolist()
            scores = neutron_math.batch_similarity(q_vec, st.session_state.kb["vecs"])
            
            # AUGMENTATION (Selecting Top 3 context matches)
            top_indices = np.argsort(scores)[-3:][::-1]
            context_text = "\n\n".join([st.session_state.kb["chunks"][i] for i in top_indices if scores[i] > 0.2])

            if not context_text:
                st.error("No relevant information found in the PDF.")
            else:
                #GENERATION
                prompt = f"""
                You are NeutronRAG. Answer based ONLY on this context:
                {context_text}
                
                Question: {query}
                """
                try:
                    response = st.session_state.llm.generate_content(prompt)
                    st.info(f"**Confidence: {scores[top_indices[0]]*100:.1f}%**")
                    st.write(response.text)
                    st.caption(f"Source: {st.session_state.kb['name']}")
                except Exception as e:
                    st.error(f"LLM Error: {str(e)}")