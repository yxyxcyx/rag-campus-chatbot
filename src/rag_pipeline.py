# rag_pipeline.py

"""
Features:
1. OCR for image-based PDFs
2. Hybrid Search (BM25 + Vector + Reranking)
3. Query Preprocessing & Expansion
4. Semantic Caching
5. Multi-strategy Reranking
6. Comprehensive Evaluation Metrics

Architecture:
- Document Loading: OCR with preprocessing
- Chunking: Sentence-window retrieval
- Retrieval: Hybrid BM25 + Vector search
- Reranking: Cross-encoder + diversity filter
- Generation: Groq LLM with optimized prompts
- Caching: Semantic similarity-based caching
"""

import os
import hashlib
import pickle
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import groq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from requests.exceptions import Timeout

from logging_config import get_logger

# Module logger
logger = get_logger(__name__)

# Import enhanced loader (conditional for container compatibility)
try:
    from enhanced_document_loader import EnhancedDocumentLoader
    ENHANCED_LOADER_AVAILABLE = True
except ImportError:
    ENHANCED_LOADER_AVAILABLE = False

try:
    from sentence_window_retrieval import chunk_text_with_sentence_windows
    SENTENCE_WINDOW_AVAILABLE = True
except ImportError:
    SENTENCE_WINDOW_AVAILABLE = False


# Config is loaded via centralized config module
# Settings are passed as parameters to avoid direct os.getenv calls


# ============================================================================
# SECTION 1: DOCUMENT LOADING WITH ENHANCED OCR
# ============================================================================

def load_documents_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Load documents with enhanced OCR support (worker containers only)
    """
    if not ENHANCED_LOADER_AVAILABLE:
        raise ImportError("Enhanced document loading not available in this container. Use worker container for ingestion.")
    
    loader = EnhancedDocumentLoader(dpi=2.0, parallel_workers=4)
    return loader.load_folder(folder_path)


# ============================================================================
# SECTION 2: CHUNKING WITH SENTENCE-WINDOW RETRIEVAL
# ============================================================================

def chunk_text(documents_dict: Dict[str, str], window_size: int = 3) -> Tuple[List[str], List[str]]:
    """
    Sentence-window chunking for precise retrieval (worker containers only)
    
    Returns:
        (central_sentences, windows)
    """
    if not SENTENCE_WINDOW_AVAILABLE:
        raise ImportError("Sentence-window retrieval not available in this container. Use worker container for ingestion.")
    
    return chunk_text_with_sentence_windows(documents_dict, window_size=window_size)


def embed_chunks(chunks: List[str], embedding_model: HuggingFaceEmbeddings) -> List[List[float]]:
    """
    Generate embeddings for chunks
    """
    return embedding_model.embed_documents(chunks)


# ============================================================================
# SECTION 3: QUERY PREPROCESSING & EXPANSION
# ============================================================================

class QueryProcessor:
    """Advanced query preprocessing and expansion"""
    
    @staticmethod
    def clean_query(query: str) -> str:
        """Clean and normalize query"""
        query = query.strip()
        # Remove multiple spaces
        query = " ".join(query.split())
        return query
    
    @staticmethod
    def expand_query(query: str) -> List[str]:
        """
        Generate query variations for better retrieval
        
        Returns:
            [original, expanded_version_1, expanded_version_2, ...]
        """
        queries = [query]
        
        # Add question variations
        if not query.endswith('?'):
            queries.append(query + '?')
        
        # Add keyword extraction (simple version)
        words = query.lower().split()
        keywords = [w for w in words if len(w) > 4 and w not in [
            'what', 'when', 'where', 'which', 'about', 'there', 'their'
        ]]
        if keywords:
            queries.append(' '.join(keywords))
        
        return queries


# ============================================================================
# SECTION 4: HYBRID SEARCH (BM25 + VECTOR)
# ============================================================================

class HybridRetriever:
    """Hybrid retrieval combining BM25 and vector search"""
    
    def __init__(
        self, 
        documents: List[str],
        embeddings: List[List[float]],
        embedding_model: HuggingFaceEmbeddings,
        collection: chromadb.Collection
    ):
        """
        Initialize hybrid retriever
        
        Args:
            documents: List of text chunks
            embeddings: Precomputed embeddings
            embedding_model: Model for query embedding
            collection: ChromaDB collection
        """
        self.documents = documents
        self.embeddings = embeddings
        self.embedding_model = embedding_model
        self.collection = collection
        
        # Initialize BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info("Hybrid Retriever initialized", document_count=len(documents))
    
    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        BM25 keyword search
        
        Returns:
            List of (doc_index, score) tuples
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
    
    def retrieve_vector(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Vector similarity search
        
        Returns:
            List of (doc_index, score) tuples
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results['ids'] or not results['ids'][0]:
            return []
        
        # Get indices and distances
        doc_indices = []
        for doc_id in results['ids'][0]:
            # Parse chunk ID to get index
            try:
                idx = int(doc_id.split('_')[-1])
                doc_indices.append(idx)
            except:
                pass
        
        # Convert distances to similarity scores (1 - distance)
        similarities = [1 - d for d in results['distances'][0]]
        
        return list(zip(doc_indices, similarities))
    
    def hybrid_retrieve(
        self, 
        query: str, 
        top_k: int = 20,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7
    ) -> List[str]:
        """
        Hybrid retrieval combining BM25 and vector search
        
        Args:
            query: Search query
            top_k: Number of results
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            List of retrieved documents
        """
        # Get BM25 results
        bm25_results = self.retrieve_bm25(query, top_k=top_k * 2)
        
        # Get vector results
        vector_results = self.retrieve_vector(query, top_k=top_k * 2)
        
        # Combine scores
        combined_scores = {}
        
        # Add BM25 scores
        for idx, score in bm25_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + bm25_weight * score
        
        # Add vector scores
        for idx, score in vector_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + vector_weight * score
        
        # Sort by combined score
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Return documents
        return [self.documents[idx] for idx, _ in sorted_indices if idx < len(self.documents)]


# ============================================================================
# SECTION 5: MULTI-STRATEGY RERANKING
# ============================================================================

def rerank_with_cross_encoder(
    query: str,
    documents: List[str],
    cross_encoder: CrossEncoder,
    top_k: int = 5
) -> List[str]:
    """
    Rerank documents using cross-encoder
    """
    if not documents:
        return []
    
    # Create pairs
    pairs = [[query, doc] for doc in documents]
    
    # Get scores
    scores = cross_encoder.predict(pairs)
    
    # Sort by score
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    
    return [documents[idx] for idx in ranked_indices]


def diversity_filter(documents: List[str], threshold: float = 0.8) -> List[str]:
    """
    Remove near-duplicate documents for diversity
    
    Simple version: Check string similarity
    """
    if len(documents) <= 1:
        return documents
    
    filtered = [documents[0]]
    
    for doc in documents[1:]:
        # Check if too similar to any filtered doc
        is_duplicate = False
        for filtered_doc in filtered:
            # Simple overlap ratio
            words_a = set(doc.lower().split())
            words_b = set(filtered_doc.lower().split())
            if not words_a or not words_b:
                continue
            overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
            if overlap > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(doc)
    
    return filtered


# Enhanced retrieval with diversity filtering
def retrieve_and_rerank(
    query: str,
    embedding_model: HuggingFaceEmbeddings,
    collection: chromadb.Collection,
    cross_encoder: CrossEncoder,
    n_initial: int = 20,
    n_final: int = 5,  # Increased to allow diversity filtering
    use_diversity_filter: bool = True,
    diversity_threshold: float = 0.85
) -> List[str]:
    """Enhanced retrieval used by the FastAPI API and evaluation scripts.

    Features:
    1. Query preprocessing and expansion
    2. Vector search in ChromaDB
    3. Cross-encoder reranking
    4. Diversity filtering to remove redundant results
    """
    # Query preprocessing
    processor = QueryProcessor()
    clean_query = processor.clean_query(query)
    
    # Stage 1: initial vector retrieval from ChromaDB
    query_embedding = embedding_model.embed_query(clean_query)
    initial_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_initial,
    )

    documents = initial_results.get("documents", [[]])
    if not documents or not documents[0]:
        return []

    retrieved_windows = documents[0]

    # Stage 2: precise reranking with cross-encoder
    pairs = [[clean_query, window] for window in retrieved_windows]
    scores = cross_encoder.predict(pairs)

    # Get more results initially to allow for diversity filtering
    rerank_top_k = n_final * 2 if use_diversity_filter else n_final
    scored_windows = sorted(zip(scores, retrieved_windows), reverse=True)
    top_windows = [window for score, window in scored_windows[:rerank_top_k]]
    
    # Stage 3: Apply diversity filter to remove redundancy
    if use_diversity_filter and len(top_windows) > 1:
        top_windows = diversity_filter(top_windows, threshold=diversity_threshold)
    
    # Return final top results
    return top_windows[:n_final]


def retrieve_and_rerank_hybrid(
    query: str,
    embedding_model: HuggingFaceEmbeddings,
    collection: chromadb.Collection,
    cross_encoder: CrossEncoder,
    bm25_index: Optional[BM25Okapi] = None,
    documents: Optional[List[str]] = None,
    n_initial: int = 30,
    n_final: int = 5,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
    use_diversity_filter: bool = True,
    diversity_threshold: float = 0.85
) -> List[str]:
    """Hybrid retrieval with BM25 + Vector search when BM25 index is available.
    
    Falls back to enhanced vector-only retrieval if BM25 is not available.
    """
    # If BM25 index not available, fall back to enhanced vector retrieval
    if bm25_index is None or documents is None:
        return retrieve_and_rerank(
            query, embedding_model, collection, cross_encoder,
            n_initial, n_final, use_diversity_filter, diversity_threshold
        )
    
    # Query preprocessing
    processor = QueryProcessor()
    clean_query = processor.clean_query(query)
    
    # Stage 1a: BM25 retrieval
    tokenized_query = clean_query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:n_initial]
    bm25_results = {idx: bm25_scores[idx] for idx in bm25_top_indices if bm25_scores[idx] > 0}
    
    # Stage 1b: Vector retrieval
    query_embedding = embedding_model.embed_query(clean_query)
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_initial,
    )
    
    # Combine scores
    combined_scores = {}
    
    # Add BM25 scores
    for idx, score in bm25_results.items():
        if idx < len(documents):
            combined_scores[documents[idx]] = bm25_weight * score
    
    # Add vector scores  
    if vector_results.get("documents") and vector_results["documents"][0]:
        retrieved_docs = vector_results["documents"][0]
        distances = vector_results.get("distances", [[]])[0]
        for doc, dist in zip(retrieved_docs, distances):
            similarity = 1 - dist  # Convert distance to similarity
            combined_scores[doc] = combined_scores.get(doc, 0) + vector_weight * similarity
    
    # Sort by combined score
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in sorted_docs[:n_initial]]
    
    if not top_docs:
        return []
    
    # Stage 2: Cross-encoder reranking
    pairs = [[clean_query, doc] for doc in top_docs]
    scores = cross_encoder.predict(pairs)
    
    # Get more results initially to allow for diversity filtering
    rerank_top_k = n_final * 2 if use_diversity_filter else n_final
    scored_docs = sorted(zip(scores, top_docs), reverse=True)
    reranked_docs = [doc for score, doc in scored_docs[:rerank_top_k]]
    
    # Stage 3: Apply diversity filter
    if use_diversity_filter and len(reranked_docs) > 1:
        reranked_docs = diversity_filter(reranked_docs, threshold=diversity_threshold)
    
    # Return final top results
    return reranked_docs[:n_final]


# ============================================================================
# SECTION 6: SEMANTIC CACHING
# ============================================================================

class SemanticCache:
    """
    Semantic caching for similar queries
    """
    
    def __init__(
        self, 
        embedding_model: HuggingFaceEmbeddings,
        cache_file: str = ".cache/semantic_cache.pkl",
        similarity_threshold: float = 0.95,
        max_age_hours: int = 24
    ):
        """
        Args:
            embedding_model: Model for query embeddings
            cache_file: Path to cache file
            similarity_threshold: Minimum similarity for cache hit
            max_age_hours: Maximum age of cache entries
        """
        self.embedding_model = embedding_model
        self.cache_file = cache_file
        self.similarity_threshold = similarity_threshold
        self.max_age_hours = max_age_hours
        
        # Load cache
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def _compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity"""
        a = np.array(emb1)
        b = np.array(emb2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for query
        
        Returns:
            Cached response or None
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        # Check cache
        for cache_key, cache_entry in self.cache.items():
            # Check age
            age = datetime.now() - cache_entry['timestamp']
            if age > timedelta(hours=self.max_age_hours):
                continue
            
            # Check similarity
            similarity = self._compute_similarity(query_embedding, cache_entry['embedding'])
            
            if similarity >= self.similarity_threshold:
                logger.info("Cache hit", similarity=round(similarity, 3))
                return cache_entry['response']
        
        return None
    
    def set(self, query: str, response: str):
        """
        Cache query-response pair
        """
        query_embedding = self.embedding_model.embed_query(query)
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        self.cache[cache_key] = {
            'query': query,
            'embedding': query_embedding,
            'response': response,
            'timestamp': datetime.now()
        }
        
        self._save_cache()
    
    def clear_old(self):
        """Remove old cache entries"""
        cutoff = datetime.now() - timedelta(hours=self.max_age_hours)
        self.cache = {
            k: v for k, v in self.cache.items()
            if v['timestamp'] > cutoff
        }
        self._save_cache()


# ============================================================================
# SECTION 7: RETRIEVAL & GENERATION
# ============================================================================

@retry(
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((groq.RateLimitError, groq.APITimeoutError, groq.InternalServerError, Timeout))
)
def generate_response(context: str, query: str) -> str:
    """
    Generate response using Groq LLM with optimized prompt
    """
    prompt_template = f"""You are an expert assistant for a university campus. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer directly and concisely
2. Use ONLY information from the context
3. If the context doesn't contain the answer, say "I don't have enough information to answer this question based on the available documents."
4. Cite specific details when possible (e.g., fees, dates, requirements)
5. Be precise with numbers and requirements

Answer:"""
    
    # Import config here to get API key from centralized config
    from config import get_settings
    settings = get_settings()
    
    client = groq.Groq(api_key=settings.groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt_template,
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.1,  # Lower for more factual responses
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def advanced_rag_query(
    query: str,
    hybrid_retriever: HybridRetriever,
    cross_encoder: CrossEncoder,
    semantic_cache: Optional[SemanticCache] = None,
    use_cache: bool = True
) -> Tuple[str, Dict[str, any]]:
    """
    Complete advanced RAG pipeline
    
    Returns:
        (answer, metadata)
    """
    metadata = {
        'query': query,
        'cache_hit': False,
        'num_retrieved': 0,
        'num_reranked': 0
    }
    
    # 1. Check cache
    if use_cache and semantic_cache:
        cached_response = semantic_cache.get(query)
        if cached_response:
            metadata['cache_hit'] = True
            return cached_response, metadata
    
    # 2. Query preprocessing
    processor = QueryProcessor()
    clean_query = processor.clean_query(query)
    
    # 3. Hybrid retrieval
    retrieved_docs = hybrid_retriever.hybrid_retrieve(clean_query, top_k=20)
    metadata['num_retrieved'] = len(retrieved_docs)
    
    if not retrieved_docs:
        response = "I don't have enough information to answer this question based on the available documents."
        return response, metadata
    
    # 4. Reranking
    reranked_docs = rerank_with_cross_encoder(clean_query, retrieved_docs, cross_encoder, top_k=5)
    metadata['num_reranked'] = len(reranked_docs)
    
    # 5. Diversity filter
    final_docs = diversity_filter(reranked_docs, threshold=0.85)
    
    # 6. Generate response
    context = "\n\n".join(final_docs)
    response = generate_response(context, clean_query)
    
    # 7. Cache result
    if use_cache and semantic_cache:
        semantic_cache.set(query, response)
    
    return response, metadata
