import streamlit as st
import numpy as np
import google.generativeai as genai
from engine import RAGEngine
import sys
import os
import subprocess
import pybind11
import sysconfig

# --- 1. AUTOMATIC C++ COMPILATION & IMPORT ---
# We do this BEFORE any other logic to ensure the module is ready
def compile_and_import():
    # Detect paths for the current environment (Mac 3.13 or Cloud 3.11)
    python_include = sysconfig.get_paths()['include']
    pybind_include = pybind11.get_include()
    
    # Define the output path
    so_path = os.path.join(os.getcwd(), "core", "neutron_math.so")
    
    # Only compile if the library file is missing
    if not os.path.exists(so_path):
        cmd = [
            "c++", "-O3", "-Wall", "-shared", "-std=c++11", "-fPIC",
            f"-I{python_include}",
            f"-I{pybind_include}",
            "core/similarity.cpp",
            "-o", "core/neutron_math.so"
        ]

        # Add Mac-specific flag if running locally
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]

        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            st.error(f"C++ Compilation failed: {e}")
            return None

    # Add the core folder to sys.path so Python can find the .so file
    sys.path.append(os.path.join(os.getcwd(), "core"))
    
    try:
        import neutron_math 
        return neutron_math
    except ImportError:
        st.error("Could not import neutron_math. Check your C++ compilation.")
        return None

# Run the compilation
neutron_math = compile_and_import()

# --- 2. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="NeutronRAG", page_icon="⚡")
st.title("NeutronRAG")

# Setup Gemini using Secrets
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("GEMINI_API_KEY not found in Streamlit Secrets. Please add it in the dashboard.")

# Initialize shared components once per session
if 'engine' not in st.session_state:
    st.session_state.engine = RAGEngine()
    st.session_state.llm = genai.GenerativeModel("gemini-2.5-flash") # Stable version
    st.session_state.kb = {"chunks": [], "vecs": [], "name": ""}

#  SIDEBAR: INDEXING
with st.sidebar:
    st.header("Upload Knowledge")
    up = st.file_uploader("Upload PDF", type="pdf")
    
    if up and st.button("Index Data"):
        with st.spinner("Processing PDF with C++ Engine..."):
            content = up.getvalue()
            # Extract text and create embeddings
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

# UI
st.header("Ask Questions")
query = st.text_input("What would you like to know?")

if query:
    if not st.session_state.kb["chunks"]:
        st.warning("Please upload and index a PDF first.")
    elif neutron_math is None:
        st.error("C++ Engine is not available. Please check server logs.")
    else:
        with st.spinner("C++ Engine searching..."):
            # RETRIEVAL (The C++ Part)
            q_vec = st.session_state.engine.model.encode([query])[0].tolist()
            scores = neutron_math.batch_similarity(q_vec, st.session_state.kb["vecs"])
            
            # AUGMENTATION (Selecting Top 3 context matches)
            top_indices = np.argsort(scores)[-3:][::-1]
            # Ensure we only use relevant context (threshold > 0.2)
            context_text = "\n\n".join([st.session_state.kb["chunks"][i] for i in top_indices if scores[i] > 0.2])

            if not context_text:
                st.error("No relevant information found in the PDF.")
            else:
                # GENERATION (The AI Part)
                prompt = f"""
                You are NeutronRAG. Answer based ONLY on the provided context.
                Context: {context_text}
                
                Question: {query}
                """
                try:
                    response = st.session_state.llm.generate_content(prompt)
                    # Display confidence and source
                    st.info(f"**Confidence: {scores[top_indices[0]]*100*2.5:.1f}%**")
                    st.write(response.text)
                    st.caption(f"Source: {st.session_state.kb['name']}")
                except Exception as e:
                    st.error(f"LLM Error: {str(e)}")