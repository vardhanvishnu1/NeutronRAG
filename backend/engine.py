import fitz 
import re
from sentence_transformers import SentenceTransformer

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_text_from_pdf(self, file_content):
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + " "
        # Clean up whitespace and newlines
        return re.sub(r'\s+', ' ', text).strip()

    def get_chunks(self, text, chunk_size=700, overlap=150):
        """Creates chunks with a sliding window (overlap)"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            current_chunk.append(sentence)
            current_length += len(sentence)

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                # Keep the last few sentences for the next chunk (Overlap)
                # This ensures context isn't lost at the boundaries
                overlap_count = max(1, len(current_chunk) // 3) 
                current_chunk = current_chunk[-overlap_count:]
                current_length = sum(len(s) for s in current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def create_embeddings(self, chunks):
        return self.model.encode(chunks).tolist()