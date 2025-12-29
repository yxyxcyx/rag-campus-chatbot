"""
Streamlit Cloud App - Self-contained RAG Chatbot

This is a standalone version of the XMUM AI Chatbot designed for Streamlit Community Cloud.
It embeds the RAG pipeline directly without requiring external services (FastAPI, Redis, Celery).

Full Enhanced Features:
- Hybrid search (BM25 + Vector with RRF)
- Cross-encoder reranking
- Diversity filtering (word-overlap heuristic)
- Query analysis (multi-part, ambiguity detection, intent classification)
- Citation verification
- Confidence scoring
- Conversation memory (session-based)
- Enhanced anti-hallucination system prompt
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
from typing import List, Dict, Tuple, Optional

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


def diversity_filter(results: List[Dict], threshold: float = 0.85) -> List[Dict]:
    """
    Remove near-duplicate documents for diversity using word-overlap heuristic.
    
    Args:
        results: List of result dictionaries with 'document' key
        threshold: Similarity threshold (0.0-1.0), higher = more filtering
    
    Returns:
        Filtered list of results
    """
    if len(results) <= 1:
        return results
    
    filtered = [results[0]]
    
    for result in results[1:]:
        doc = result.get("document", "")
        is_duplicate = False
        
        for filtered_result in filtered:
            filtered_doc = filtered_result.get("document", "")
            
            # Word-overlap similarity
            words_a = set(doc.lower().split())
            words_b = set(filtered_doc.lower().split())
            
            if not words_a or not words_b:
                continue
            
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(result)
    
    return filtered


# =============================================================================
# QUERY ANALYSIS
# =============================================================================

class QueryAnalyzer:
    """Analyzes user queries for multi-part questions, ambiguity, and intent."""
    
    AMBIGUOUS_TERMS = {
        'fees': ['tuition fees', 'application fees', 'registration fees', 'examination fees'],
        'deadline': ['application deadline', 'registration deadline', 'payment deadline'],
        'requirements': ['admission requirements', 'graduation requirements', 'course requirements'],
        'program': ['undergraduate program', 'postgraduate program', 'diploma program'],
        'scholarship': ['merit scholarship', 'need-based scholarship', 'sports scholarship'],
        'student': ['local student', 'international student', 'part-time student'],
    }
    
    MULTI_PART_INDICATORS = [
        r'\band\b',
        r'\balso\b',
        r'\bas well as\b',
        r'\badditionally\b',
        r'\bwhat about\b',
        r'\bhow about\b',
        r'\?.*\?',
    ]
    
    def __init__(self):
        self.intent_patterns = {
            'fee_inquiry': [
                r'(?:how much|what is|what are).*(?:fee|cost|price|tuition)',
                r'(?:fee|tuition|cost).*(?:for|of)',
            ],
            'deadline_inquiry': [
                r'(?:when|what).*(?:deadline|due date|last date)',
            ],
            'requirement_inquiry': [
                r'(?:what|which).*(?:require|need|prerequisite)',
                r'(?:how to|how do).*(?:apply|register|enroll)',
            ],
            'general_info': [
                r'(?:tell me about|what is|explain|describe)',
            ],
        }
    
    def analyze(self, query: str) -> Dict:
        """Analyze a query and return analysis results."""
        query_lower = query.lower()
        
        # Detect multi-part
        is_multi_part = any(
            re.search(pattern, query_lower) 
            for pattern in self.MULTI_PART_INDICATORS
        )
        
        # Detect ambiguity
        ambiguous_terms = []
        for term, options in self.AMBIGUOUS_TERMS.items():
            if term in query_lower:
                specific_found = any(opt.lower() in query_lower for opt in options)
                if not specific_found:
                    ambiguous_terms.append({'term': term, 'options': options})
        
        # Classify intent
        intent = 'general_info'
        for intent_name, patterns in self.intent_patterns.items():
            if any(re.search(p, query_lower) for p in patterns):
                intent = intent_name
                break
        
        # Calculate confidence
        confidence = 1.0
        if ambiguous_terms:
            confidence -= 0.2 * len(ambiguous_terms)
        if is_multi_part:
            confidence -= 0.1
        if len(query.split()) < 3:
            confidence -= 0.2
        
        return {
            'is_multi_part': is_multi_part,
            'is_ambiguous': len(ambiguous_terms) > 0,
            'ambiguous_terms': ambiguous_terms,
            'intent': intent,
            'confidence': max(0.0, min(1.0, confidence)),
        }


# =============================================================================
# CITATION VERIFICATION
# =============================================================================

def verify_citations(query: str, results: List[Dict], query_analysis: Dict = None) -> Dict:
    """
    Verify and filter citations for quality and relevance.
    
    Returns dict with verified_results, rejected_results, and sources.
    """
    if not results:
        return {'verified_results': [], 'rejected_results': [], 'sources': []}
    
    query_lower = query.lower()
    query_keywords = set(query_lower.split())
    
    verified = []
    rejected = []
    sources = []
    
    for result in results:
        doc = result.get("document", "")
        doc_lower = doc.lower()
        doc_words = set(doc_lower.split())
        
        # Calculate relevance score
        overlap = len(query_keywords & doc_words)
        keyword_score = overlap / max(len(query_keywords), 1)
        
        # Partial match bonus
        partial_matches = sum(1 for kw in query_keywords if len(kw) > 3 and kw in doc_lower)
        partial_score = partial_matches / max(len(query_keywords), 1)
        
        # Intent-based scoring
        intent_score = 0.0
        if query_analysis:
            intent = query_analysis.get('intent', '')
            if intent == 'fee_inquiry' and re.search(r'RM\s*[\d,]+', doc):
                intent_score = 0.3
            elif intent == 'deadline_inquiry' and re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', doc):
                intent_score = 0.3
        
        # Length score
        length_score = min(1.0, len(doc) / 200)
        
        # Final score
        final_score = max(keyword_score, partial_score * 0.8) * 0.5 + intent_score * 0.3 + length_score * 0.2
        
        if final_score >= 0.15:
            result['relevance_score'] = final_score
            verified.append(result)
            
            # Extract source
            metadata = result.get("metadata", {})
            source = metadata.get("source", metadata.get("file_name", ""))
            if source:
                if "/" in source:
                    source = source.split("/")[-1]
                if source not in sources:
                    sources.append(source)
        else:
            rejected.append(result)
    
    return {
        'verified_results': verified,
        'rejected_results': rejected,
        'sources': sources,
    }


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


def generate_response(query: str, context: str, groq_client, conversation_history: list = None, query_analysis: Dict = None):
    """Generate response using Groq LLM with enhanced anti-hallucination prompt."""
    
    # Enhanced system message matching the FastAPI enhanced engine
    system_message = """You are an expert university information assistant for Xiamen University Malaysia (XMUM). Your role is to provide accurate, helpful information about the university based on official documents.

## Core Principles:
1. **Accuracy First**: Only provide information that is explicitly stated in the given context. Never invent or assume information.
2. **Cite Sources**: When mentioning specific facts (fees, dates, requirements), indicate which document they come from.
3. **Be Precise**: Use exact numbers, dates, and requirements from the documents.
4. **Be Helpful**: If information is incomplete, acknowledge what you know and what's missing.
5. **Handle Uncertainty**: If information is ambiguous or contradictory, explain the different possibilities.

## Response Format:
- Start with a direct answer to the question
- Include specific details (amounts, dates, requirements) when available
- Keep responses concise but complete
- For fee queries, always include the exact amount in RM format

## When You Don't Know:
- Say "Based on the available documents, I don't have information about [topic]."
- Don't guess or provide generic information
- Suggest what information might help answer the question"""

    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history for context
    if conversation_history:
        for msg in conversation_history[-4:]:  # Last 2 exchanges
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Build context-aware prompt
    special_instructions = ""
    if query_analysis and query_analysis.get('intent') == 'fee_inquiry':
        special_instructions = """
## Special Instructions for Fee Query:
- Look for exact fee amounts in the format "RM XX,XXX"
- Match the specific programme mentioned in the question
- Include duration (e.g., "4 years") if available
- Specify if the fee is annual or total
- Note if it's for local or international students"""
    
    user_message = f"""## Relevant Documents:
{context}
{special_instructions}

## Question: {query}

## Instructions:
1. Answer based ONLY on the provided documents above
2. Be specific with numbers, dates, and requirements
3. If the documents contain the answer, provide it clearly
4. If the answer is not in the documents, say so explicitly
5. For fees, always include the exact amount and any conditions

Answer:"""
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,
            max_tokens=1536
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
    st.title("XMUM AI Campus Chatbot")
    st.markdown("""
    Welcome! I'm an AI assistant trained on XMUM documents.
    Ask me about campus rules, academic policies, fees, or procedures.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Show database stats
        try:
            doc_count = collection.count()
            st.success(f"{doc_count} documents loaded")
        except:
            st.warning("Could not get document count")
        
        st.markdown("---")
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        
        st.markdown("---")
        st.caption(f"Session: {st.session_state.session_id[:8]}...")
        
            
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                if meta.get("sources"):
                    with st.expander("Sources"):
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
                # 1. Query analysis
                analyzer = QueryAnalyzer()
                query_analysis = analyzer.analyze(prompt)
                
                # 2. Hybrid search
                results = hybrid_search(
                    prompt, collection, bm25, doc_data, 
                    embedding_model, top_k=15
                )
                
                # 3. Rerank
                reranked = rerank_results(prompt, results, cross_encoder, top_k=8)
                
                # 4. Diversity filter
                diverse_results = diversity_filter(reranked, threshold=0.85)
                
                # 5. Citation verification
                citation_result = verify_citations(prompt, diverse_results, query_analysis)
                verified_results = citation_result['verified_results']
                
                # Use verified results if available, otherwise fall back to diverse results
                final_results = verified_results if verified_results else diverse_results[:5]
                
                # 6. Build context
                context = "\n\n---\n\n".join([r["document"] for r in final_results])
                
                # 7. Generate response with query analysis
                conversation_history = [
                    {"role": m["role"], "content": m["content"]} 
                    for m in st.session_state.messages[:-1]  # Exclude current
                ]
                answer = generate_response(prompt, context, groq_client, conversation_history, query_analysis)
                
                # 8. Calculate confidence
                confidence = calculate_confidence(answer, final_results)
                
                # 9. Extract sources (from citation verification or fallback)
                sources = citation_result['sources'] if citation_result['sources'] else extract_sources(final_results)
                
                # Display
                st.markdown(answer)
                
                metadata = {
                    "sources": sources,
                    "confidence": confidence,
                    "query_analysis": query_analysis
                }
                
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.caption(f"• {source}")
                
                # Show query analysis info if relevant
                if query_analysis.get('is_multi_part') or query_analysis.get('is_ambiguous'):
                    with st.expander("Query Analysis"):
                        if query_analysis.get('is_multi_part'):
                            st.caption("ℹ️ Multi-part question detected")
                        if query_analysis.get('is_ambiguous'):
                            st.caption(f"⚠️ Ambiguous terms: {', '.join([t['term'] for t in query_analysis.get('ambiguous_terms', [])])}")
                        st.caption(f"Intent: {query_analysis.get('intent', 'general')}")
                
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
