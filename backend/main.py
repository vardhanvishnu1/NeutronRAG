from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import google.generativeai as genai
from engine import RAGEngine
import sys
import os

#C++ Path
sys.path.append(os.path.join(os.getcwd(), "core"))
import neutron_math 

genai.configure(api_key="AIzaSyD7TMrfj_Avy01o8U0bp-HH5YLua-2D2Ow")
llm_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()
engine = RAGEngine()
kb = {"chunks": [], "vecs": [], "name": ""}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = engine.get_text_from_pdf(content)
    chunks = engine.get_chunks(text)
    vecs = engine.create_embeddings(chunks)
    kb["chunks"], kb["vecs"], kb["name"] = chunks, vecs, file.filename
    return {"status": "success", "chunks": len(chunks)}

@app.get("/ask")
async def ask(q: str):
    if not kb["chunks"]: 
        return {"answer": "Please upload a PDF first.", "confidence": "0%", "source": "None"}
    
    #C++ SIMILARITY ENGINE
    q_vec = engine.model.encode([q])[0].tolist()
    scores = neutron_math.batch_similarity(q_vec, kb["vecs"])
    
    #CONTEXT RETRIEVAL
    top_indices = np.argsort(scores)[-3:][::-1]
    context_text = "\n\n".join([kb["chunks"][i] for i in top_indices if scores[i] > 0.2])

    if not context_text:
        return {"answer": "No relevant info found in PDF.", "confidence": "0%", "source": kb["name"]}

    #LLM INTELLIGENCE
    prompt = f"""
    You are NeutronRAG. Answer based ONLY on this context:
    {context_text}
    
    Question: {q}
    """
    try:
        response = llm_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        answer = f"LLM Error: {str(e)}"

    return {
        "answer": answer,
        "confidence": f"{scores[top_indices[0]]*100:.1f}%",
        "source": kb["name"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)