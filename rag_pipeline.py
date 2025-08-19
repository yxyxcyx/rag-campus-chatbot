# rag_pipeline.py

"""
Core components of the Retrieval-Augmented Generation (RAG) pipeline.

This module contains functions for:
1.  Loading and extracting text from various document types (PDF, DOCX, TXT).
2.  Chunking text into smaller, manageable segments with metadata.
3.  Embedding text chunks into vector representations.
4.  A two-stage retrieval process:
    a. Fast initial retrieval from a vector database.
    b. Precise re-ranking using a Cross-Encoder model.
5.  Generating a final response using a Large Language Model (LLM), grounded
    on the retrieved context.
"""

import os
import io
import re
from typing import List, Dict
import groq
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from requests.exceptions import Timeout

# Configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# SECTION 1: DATA LOADING AND PREPROCESSING
def load_documents_from_folder(folder_path: str) -> Dict[str, str]:
    document_texts = {}
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_text = ""
        try:
            if filename.endswith(".pdf"):
                doc = fitz.open(file_path)
                for page_num, page in enumerate(doc):
                    # 1. Try to extract text directly
                    page_text = page.get_text("text")
                    
                    # 2. If direct extraction yields little text, fall back to OCR
                    # A threshold (50 char) helps identify.
                    if len(page_text.strip()) < 50:
                        print(f"  - Page {page_num + 1} of {filename} has little text. Falling back to OCR.")
                        pix = page.get_pixmap(dpi=300)
                        img_data = pix.tobytes("png")
                        image = Image.open(io.BytesIO(img_data))
                        ocr_text = pytesseract.image_to_string(image, lang='eng')
                        file_text += ocr_text + "\n"
                    else:
                        file_text += page_text + "\n"
                doc.close()
            
            elif filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
            
            elif filename.endswith(".docx"):
                doc = docx.Document(file_path)
                file_text = "\n".join([para.text for para in doc.paragraphs])

            if file_text.strip():
                document_texts[filename] = file_text
                print(f"  - Successfully loaded and extracted text from: {filename}")

        except Exception as e:
            print(f"  - Could not read file {filename}. Reason: {e}")

    return document_texts


def chunk_text(documents_dict: Dict[str, str]) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len
    )
    all_chunks = []
    for filename, text in documents_dict.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            # Prepend metadata to provide source context during retrieval.
            all_chunks.append(f"Source Document: {filename}\n\n{chunk}")
    return all_chunks


def embed_chunks(chunks: List[str], embedding_model: HuggingFaceEmbeddings):

    # Embeds a list of text chunks using the provided LangChain embedding model. 
    return embedding_model.embed_documents(chunks)



# SECTION 2: RETRIEVAL AND RE-RANKING
def retrieve_and_rerank(query: str, embedding_model: HuggingFaceEmbeddings, collection: chromadb.Collection, cross_encoder: CrossEncoder) -> List[str]:

    # Stage 1: Initial Retrieval
    query_embedding = embedding_model.embed_query(query)
    initial_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20  # Increased to 20 from 10
    )
    retrieved_chunks = initial_results['documents'][0]

    if not retrieved_chunks:
        return []

    # Stage 2: Precise Re-ranking
    # Cross-Encoder scores the relevance of the query against each retrieved chunk.
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    # Combine chunks with their relevance scores and sort for the best matches.
    scored_chunks = sorted(zip(scores, retrieved_chunks), reverse=True)

    # Return the top N chunks that are most relevant to the query.
    top_chunks = [chunk for score, chunk in scored_chunks[:3]]
    return top_chunks



# SECTION 3: LLM RESPONSE GENERATION


def custom_wait_for_rate_limit(retry_state):
    exception = retry_state.outcome.exception()
    
    if isinstance(exception, groq.RateLimitError):
        error_message = str(exception)
        # Find the suggested wait time
        match = re.search(r"Please try again in ([\d.]+m)?([\d.]+)s", error_message)
        if match:
            minutes_str, seconds_str = match.groups()
            minutes = float(minutes_str.replace('m', '')) if minutes_str else 0
            seconds = float(seconds_str) if seconds_str else 0
            wait_time = (minutes * 60) + seconds + 0.1 # Add a small buffer
            print(f"    - Rate limit hit. API suggested waiting {wait_time:.1f}s. Waiting...")
            return wait_time
        else:
            # Fallback if the error message format changes
            print("    - Rate limit hit. Using exponential backoff.")
            return wait_random_exponential(min=5, max=60)(retry_state)
            
    # For other errors use a standard backoff.
    return wait_random_exponential(min=2, max=60)(retry_state)


@retry(
    wait=custom_wait_for_rate_limit,
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((groq.RateLimitError, groq.APITimeoutError, groq.InternalServerError, Timeout))
)
def generate_response(context: str, query: str) -> str:

    load_dotenv()
    
    prompt_template = f"""
    CONTEXT:
    {context}

    USER QUESTION:
    {query}

    INSTRUCTIONS:
    - Answer the USER QUESTION using ONLY the information provided in the CONTEXT.
    - If the context does not contain the answer, state that you cannot answer based on the provided documents.
    - Be concise and directly answer the question.

    ANSWER:
    """
    
    client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_template,
            }
        ],
        model="llama3-8b-8192",
        temperature=0.2, # Lower temperature for more factual, less creative answers
    )
    return chat_completion.choices[0].message.content