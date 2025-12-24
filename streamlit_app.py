"""
Streamlit Cloud App - Self-contained RAG Chatbot

This is a standalone version of the XMUM AI Chatbot designed for Streamlit Community Cloud.
It embeds the RAG pipeline directly without requiring external services (FastAPI, Redis, Celery).

Features retained:
- Hybrid search (BM25 + Vector with RRF)
- Cross-encoder reranking
- Confidence scoring
- Conversation memory (session-based)
- Table-aware document retrieval
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables before imports
os.environ.setdefault("CHROMA_DB_PATH", "./chroma_db")
os.environ.setdefault("COLLECTION_NAME", "campus_docs")

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
import numpy as np
from rank_bm25 import BM25Okapi
import re
import uuid
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# =============================================================================
# MODEL LOADING (Cached)
# =============================================================================

@st.cache_resource
def load_embedding_model():
    """Load the embedding model (cached across sessions)."""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


@st.cache_resource
def load_cross_encoder():
    """Load the cross-encoder for reranking (cached across sessions)."""
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


@st.cache_resource
def load_vector_db():
    """Load ChromaDB collection (cached across sessions)."""
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection("campus_docs")
        return collection
    except Exception:
        # Try alternative collection name
        try:
            collection = client.get_collection("collection")
            return collection
        except Exception:
            return None


@st.cache_resource
def build_bm25_index(_collection):
    """Build BM25 index from collection documents."""
    if _collection is None:
        return None, []
    
    try:
        all_docs = _collection.get(include=["documents", "metadatas"])
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])
        ids = all_docs.get("ids", [])
        
        if not documents:
            return None, []
        
        # Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        return bm25, list(zip(ids, documents, metadatas))
    except Exception as e:
        st.warning(f"Could not build BM25 index: {e}")
        return None, []


# =============================================================================
# RAG PIPELINE FUNCTIONS
# =============================================================================

def reciprocal_rank_fusion(rankings: list, k: int = 60) -> dict:
    """Combine multiple rankings using Reciprocal Rank Fusion."""
    fused_scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
    return fused_scores


def hybrid_search(query: str, collection, bm25, doc_data, embedding_model, top_k: int = 10):
    """Perform hybrid search combining BM25 and vector similarity with RRF."""
    if collection is None:
        return []
    
    # Vector search
    query_embedding = embedding_model.encode([query])[0].tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, 20),
        include=["documents", "metadatas", "distances"]
    )
    
    # Create vector ranking
    vector_ranking = []
    if vector_results and vector_results.get("ids"):
        for i, doc_id in enumerate(vector_results["ids"][0]):
            vector_ranking.append((doc_id, 1 - vector_results["distances"][0][i]))
    
    # BM25 search
    bm25_ranking = []
    if bm25 is not None and doc_data:
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                doc_id, doc_text, metadata = doc_data[idx]
                bm25_ranking.append((doc_id, bm25_scores[idx]))
    
    # Combine with RRF
    if bm25_ranking:
        fused_scores = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
    else:
        fused_scores = {doc_id: score for doc_id, score in vector_ranking}
    
    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]
    
    # Retrieve full documents
    results = []
    if vector_results and vector_results.get("ids"):
        id_to_doc = {}
        for i, doc_id in enumerate(vector_results["ids"][0]):
            id_to_doc[doc_id] = {
                "id": doc_id,
                "document": vector_results["documents"][0][i],
                "metadata": vector_results["metadatas"][0][i] if vector_results.get("metadatas") else {},
                "score": fused_scores.get(doc_id, 0)
            }
        
        # Also add BM25-only results
        if doc_data:
            for doc_id, doc_text, metadata in doc_data:
                if doc_id in fused_scores and doc_id not in id_to_doc:
                    id_to_doc[doc_id] = {
                        "id": doc_id,
                        "document": doc_text,
                        "metadata": metadata or {},
                        "score": fused_scores[doc_id]
                    }
        
        for doc_id in sorted_ids:
            if doc_id in id_to_doc:
                results.append(id_to_doc[doc_id])
    
    return results


def rerank_results(query: str, results: list, cross_encoder, top_k: int = 5):
    """Rerank results using cross-encoder."""
    if not results or cross_encoder is None:
        return results[:top_k]
    
    # Prepare pairs for cross-encoder
    pairs = [(query, r["document"]) for r in results]
    
    # Get cross-encoder scores
    scores = cross_encoder.predict(pairs)
    
    # Add scores and sort
    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])
    
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def calculate_confidence(answer: str, results: list) -> float:
    """Calculate confidence score based on answer quality and retrieval."""
    confidence = 0.85  # Base confidence
    
    # Penalize for "no information" phrases
    no_info_phrases = [
        "i don't have", "i do not have", "no information",
        "not mentioned", "cannot find", "not available",
        "i'm not sure", "i am not sure", "unclear"
    ]
    answer_lower = answer.lower()
    for phrase in no_info_phrases:
        if phrase in answer_lower:
            confidence -= 0.3
            break
    
    # Penalize for hedging language
    hedging_phrases = ["might", "may", "possibly", "perhaps", "could be"]
    hedging_count = sum(1 for phrase in hedging_phrases if phrase in answer_lower)
    confidence -= hedging_count * 0.05
    
    # Boost for specific data (RM amounts, dates, etc.)
    if re.search(r'RM\s*[\d,]+', answer):
        confidence += 0.1
    if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', answer):
        confidence += 0.05
    
    # Consider retrieval quality
    if results:
        avg_score = np.mean([r.get("rerank_score", r.get("score", 0)) for r in results[:3]])
        if avg_score > 0.7:
            confidence += 0.1
        elif avg_score < 0.3:
            confidence -= 0.1
    else:
        confidence -= 0.2
    
    return max(0.1, min(1.0, confidence))


def generate_response(query: str, context: str, groq_client, conversation_history: list = None):
    """Generate response using Groq LLM."""
    
    # Build conversation context
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant for XMUM (Xiamen University Malaysia) campus.
Answer questions based on the provided context. Be accurate and helpful.
If the context doesn't contain the answer, say so honestly.
For fee-related questions, always mention the specific amounts if available.
Keep responses concise but complete."""
        }
    ]
    
    # Add conversation history for context
    if conversation_history:
        for msg in conversation_history[-4:]:  # Last 2 exchanges
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current query with context
    user_message = f"""Context from XMUM documents:
{context}

Question: {query}

Please answer based on the context above. If the information is not in the context, say so."""
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def extract_sources(results: list) -> list:
    """Extract source information from results."""
    sources = []
    seen = set()
    
    for r in results[:5]:
        metadata = r.get("metadata", {})
        source = metadata.get("source", metadata.get("file_name", "Unknown"))
        
        # Clean up source path
        if "/" in source:
            source = source.split("/")[-1]
        if "\\" in source:
            source = source.split("\\")[-1]
        
        if source not in seen:
            seen.add(source)
            sources.append(source)
    
    return sources


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="XMUM AI Chatbot",
        page_icon="🎓",
        layout="centered"
    )
    
    # Check for API key
    groq_api_key = None
    if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please add it to Streamlit secrets.")
        st.info("Go to your app settings → Secrets → Add: GROQ_API_KEY = 'your_key_here'")
        st.stop()
    
    # Initialize Groq client
    groq_client = Groq(api_key=groq_api_key)
    
    # Load models with progress
    with st.spinner("Loading AI models... (first load takes ~30 seconds)"):
        embedding_model = load_embedding_model()
        cross_encoder = load_cross_encoder()
        collection = load_vector_db()
        bm25, doc_data = build_bm25_index(collection)
    
    # Check if database is loaded
    if collection is None:
        st.error("⚠️ Vector database not found. Please ensure documents are ingested.")
        st.info("The database should be at: ./chroma_db")
        st.stop()
    
    # Session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Header
    st.title("🎓 XMUM AI Campus Chatbot")
    st.markdown("""
    Welcome! I'm an AI assistant trained on XMUM documents.
    Ask me about campus rules, academic policies, fees, or procedures.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Show database stats
        try:
            doc_count = collection.count()
            st.success(f"📚 {doc_count} documents loaded")
        except:
            st.warning("Could not get document count")
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        
        st.markdown("---")
        st.caption(f"Session: {st.session_state.session_id[:8]}...")
        
        # Feature info
        st.markdown("---")
        st.markdown("### 🔧 Features")
        st.markdown("""
        - ✅ Hybrid Search (BM25 + Vector)
        - ✅ Cross-Encoder Reranking
        - ✅ Confidence Scoring
        - ✅ Conversation Memory
        - ✅ Source Citations
        """)
    
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                if meta.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in meta["sources"]:
                            st.caption(f"• {source}")
                if meta.get("confidence"):
                    confidence = meta["confidence"]
                    color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                    st.caption(f"Confidence: :{color}[{confidence:.0%}]")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about XMUM..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Hybrid search
                results = hybrid_search(
                    prompt, collection, bm25, doc_data, 
                    embedding_model, top_k=10
                )
                
                # 2. Rerank
                reranked = rerank_results(prompt, results, cross_encoder, top_k=5)
                
                # 3. Build context
                context = "\n\n---\n\n".join([r["document"] for r in reranked])
                
                # 4. Generate response
                conversation_history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages[:-1]  # Exclude current
                ]
                answer = generate_response(prompt, context, groq_client, conversation_history)
                
                # 5. Calculate confidence
                confidence = calculate_confidence(answer, reranked)
                
                # 6. Extract sources
                sources = extract_sources(reranked)
                
                # Display
                st.markdown(answer)
                
                metadata = {
                    "sources": sources,
                    "confidence": confidence
                }
                
                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.caption(f"• {source}")
                
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.caption(f"Confidence: :{color}[{confidence:.0%}]")
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metadata": metadata
        })


if __name__ == "__main__":
    main()
