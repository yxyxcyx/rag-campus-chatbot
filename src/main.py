# main.py

"""
FastAPI Application - Stateless API Server

This script serves a RESTful API to interact with the RAG pipeline.

ARCHITECTURE:
This API server is STATELESS and follows the read-only pattern.
It serves queries via the `/ask` endpoint with full enhanced features:
- Multi-part question handling
- Ambiguous query detection  
- Conversation memory and follow-up support
- Citation verification
- Confidence scoring

Data ingestion is handled separately by Celery workers (see ingestion_worker.py).
To ingest documents, use trigger_ingestion.py script.

This separation follows best practices:
- Read path (query): Handled by this API
- Write path (ingestion): Handled by async workers
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
from typing import Optional, List, Dict, Any
import groq
import uuid
from requests.exceptions import Timeout, ConnectionError

from config import validate_startup_config, get_settings
from logging_config import setup_logging, get_logger, set_request_id
from rag_pipeline import (
    retrieve_and_rerank,
    retrieve_and_rerank_hybrid,
)
from enhanced_rag_engine import (
    EnhancedRAGEngine,
    QueryAnalyzer,
    ConversationMemory,
    CitationVerifier,
)

# SECTION 1: INITIALIZATION (Read-only resources)
# Validate configuration first - fail fast if missing critical config
settings = validate_startup_config()

# Setup structured logging
logger = setup_logging(
    level=settings.log_level,
    json_output=False,  # Set True for production/cloud environments
    app_name="rag-chatbot"
)

logger.info("Initializing RAG API", mode="stateless")
logger.info("Loading embedding model", model=settings.embedding_model_name)
embedding_model = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

logger.info("Loading Cross-Encoder model", model=settings.cross_encoder_model_name)
cross_encoder = CrossEncoder(settings.cross_encoder_model_name)

logger.info("Connecting to Vector Database", path=settings.chroma_db_path)
client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = client.get_or_create_collection(name=settings.collection_name)

# Load configuration from validated settings
ENABLE_HYBRID_SEARCH = settings.enable_hybrid_search
BM25_WEIGHT = settings.bm25_weight
VECTOR_WEIGHT = settings.vector_weight
USE_DIVERSITY_FILTER = settings.use_diversity_filter
DIVERSITY_THRESHOLD = settings.diversity_threshold

# Initialize BM25 index if hybrid search is enabled
bm25_index: Optional[BM25Okapi] = None
all_documents: Optional[List[str]] = None

# Log the current state (but don't modify it)
chunk_count = collection.count()
if chunk_count == 0:
    logger.warning(
        "Database is empty - run ingestion first",
        hint="python trigger_ingestion.py data/"
    )
else:
    logger.info("Database ready", chunk_count=chunk_count)
    
    # Initialize BM25 if hybrid search is enabled and database has content
    if ENABLE_HYBRID_SEARCH and chunk_count > 0:
        logger.info("Initializing BM25 index for hybrid search")
        try:
            # Fetch all documents from ChromaDB for BM25
            all_results = collection.get(limit=chunk_count)
            all_documents = all_results.get("documents", [])
            
            if all_documents:
                # Tokenize documents for BM25
                tokenized_docs = [doc.lower().split() for doc in all_documents]
                bm25_index = BM25Okapi(tokenized_docs)
                logger.info(
                    "BM25 index initialized",
                    document_count=len(all_documents),
                    bm25_weight=BM25_WEIGHT,
                    vector_weight=VECTOR_WEIGHT
                )
            else:
                logger.warning("Could not fetch documents for BM25 index")
                ENABLE_HYBRID_SEARCH = False
        except Exception as e:
            logger.error(
                "BM25 initialization failed, falling back to vector-only",
                exc_info=True,
                error=str(e)
            )
            ENABLE_HYBRID_SEARCH = False
            bm25_index = None
            all_documents = None
    
    if USE_DIVERSITY_FILTER:
        logger.info("Diversity filter enabled", threshold=DIVERSITY_THRESHOLD)

# Initialize Enhanced RAG Engine with all improvements
logger.info("Initializing Enhanced RAG Engine...")
enhanced_rag_engine = EnhancedRAGEngine(
    groq_api_key=settings.groq_api_key,
    embedding_model=embedding_model,
    collection=collection,
    cross_encoder=cross_encoder,
    model=settings.llm_model_name,
)
logger.info("Enhanced RAG Engine initialized with conversation memory and query analysis")

logger.info("API server ready", mode="read-only")


# SECTION 2: FASTAPI APPLICATION
app = FastAPI(
    title="RAG Chatbot API",
    description="An API for asking questions powered by a Retrieval-Augmented Generation pipeline.",
    version=settings.api_version
)


# Custom exception classes for granular error handling
class LLMServiceError(Exception):
    """Raised when LLM service is unavailable or times out."""
    pass


class DatabaseError(Exception):
    """Raised when vector database operations fail."""
    pass


# Global exception handler for unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "Unhandled exception",
        exc_info=True,
        request_id=request_id,
        path=request.url.path,
        error_type=type(exc).__name__
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred. Please try again later.",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id
        }
    )


# Middleware to add request ID for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to each request for tracing."""
    request_id = set_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Pydantic Models for Data Validation
class Question(BaseModel):
    query: str
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation memory")

class Answer(BaseModel):
    """
    Unified response model with all enhanced features.
    """
    response: str
    sources: List[str] = Field(default_factory=list, description="Source documents used")
    session_id: str = Field(description="Session ID for follow-up questions")
    confidence: float = Field(default=1.0, description="Response confidence score (0.0-1.0)")
    needs_clarification: bool = Field(default=False, description="Whether clarification is needed")
    clarification_prompt: Optional[str] = Field(default=None, description="Clarification question if needed")
    query_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Query analysis details")
    answered_parts: Optional[List[str]] = Field(default=None, description="Answered parts for multi-part questions")

class SessionInfo(BaseModel):
    session_id: str
    history_count: int
    topics: List[str]
    entities: Dict[str, Any]

# API Endpoints
@app.get("/", summary="Check API Status")
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {
        "status": "RAG Chatbot API is online and ready.",
        "version": settings.api_version,
        "database_chunks": collection.count()
    }

@app.post("/ask", response_model=Answer, summary="Ask a Question")
def ask_question(question: Question, request: Request):
    """
    Ask a question with full enhanced RAG features:
    - Multi-part question handling
    - Ambiguous query detection
    - Conversation memory and follow-up support
    - Citation verification
    - Confidence scoring
    
    Uses hybrid search (BM25 + Vector) if enabled, with cross-encoder reranking
    and diversity filtering.
    
    Pass a session_id to enable conversation memory for follow-up questions.
    
    Returns:
        200: Successful response with answer, sources, confidence, and analysis
        503: Service unavailable (LLM timeout or overload)
        500: Internal server error
    """
    user_query = question.query
    session_id = question.session_id or str(uuid.uuid4())
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(
        "Query received",
        query_length=len(user_query),
        session_id=session_id,
        request_id=request_id,
        hybrid_search=ENABLE_HYBRID_SEARCH
    )
    
    try:
        # 1. Retrieve chunks using hybrid or vector search
        if ENABLE_HYBRID_SEARCH and bm25_index is not None:
            logger.info("Using hybrid retrieval", method="bm25+vector")
            retrieved_chunks = retrieve_and_rerank_hybrid(
                user_query, 
                embedding_model, 
                collection, 
                cross_encoder,
                bm25_index=bm25_index,
                documents=all_documents,
                n_initial=30,
                n_final=settings.n_final_results,
                bm25_weight=BM25_WEIGHT,
                vector_weight=VECTOR_WEIGHT,
                use_diversity_filter=USE_DIVERSITY_FILTER,
                diversity_threshold=DIVERSITY_THRESHOLD
            )
        else:
            logger.info("Using vector retrieval", method="vector-only")
            retrieved_chunks = retrieve_and_rerank(
                user_query, 
                embedding_model, 
                collection, 
                cross_encoder,
                n_initial=settings.n_initial_retrieval,
                n_final=settings.n_final_results,
                use_diversity_filter=USE_DIVERSITY_FILTER,
                diversity_threshold=DIVERSITY_THRESHOLD
            )
        
        logger.info("Retrieval complete", chunks_retrieved=len(retrieved_chunks))
        
        # 2. Process through enhanced RAG engine (with all guardrails)
        result = enhanced_rag_engine.query(
            user_query=user_query,
            session_id=session_id,
            retrieved_chunks=retrieved_chunks,
        )
        
        logger.info(
            "Query processed",
            confidence=result.get('confidence', 0),
            sources_count=len(result.get('sources', [])),
            request_id=request_id
        )
        
        return Answer(
            response=result['response'],
            sources=result.get('sources', []),
            session_id=session_id,
            confidence=result.get('confidence', 1.0),
            needs_clarification=result.get('needs_clarification', False),
            clarification_prompt=result.get('clarification_prompt'),
            query_analysis={
                'is_multi_part': result.get('query_analysis', {}).get('is_multi_part', False),
                'is_ambiguous': result.get('query_analysis', {}).get('is_ambiguous', False),
                'intent': result.get('query_analysis', {}).get('intent', 'general'),
                'is_followup': result.get('query_analysis', {}).get('is_followup', False),
            },
            answered_parts=result.get('answered_parts', []),
        )
        
    except groq.APITimeoutError:
        logger.error("LLM timeout", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail={
                "message": "AI service is temporarily unavailable. Please try again.",
                "error_code": "LLM_TIMEOUT",
                "request_id": request_id
            }
        )
    except groq.RateLimitError:
        logger.warning("LLM rate limit exceeded")
        raise HTTPException(
            status_code=503,
            detail={
                "message": "AI service is busy. Please try again in a few seconds.",
                "error_code": "RATE_LIMITED",
                "request_id": request_id
            }
        )
    except chromadb.errors.ChromaError as e:
        logger.error(
            "Database error",
            exc_info=True,
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Database error occurred. Please contact support.",
                "error_code": "DATABASE_ERROR",
                "request_id": request_id
            }
        )
    except (Timeout, ConnectionError) as e:
        logger.error(
            "Network error during processing",
            exc_info=True,
            error=str(e)
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Service temporarily unavailable due to network issues.",
                "error_code": "NETWORK_ERROR",
                "request_id": request_id
            }
        )
    except Exception as e:
        logger.error(
            "Error in query processing",
            exc_info=True,
            error_type=type(e).__name__,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "An unexpected error occurred. Please try again.",
                "error_code": "INTERNAL_ERROR",
                "request_id": request_id
            }
        )


@app.get("/session/{session_id}", response_model=SessionInfo, summary="Get Session Info")
def get_session_info(session_id: str):
    """
    Get information about a conversation session.
    
    Returns the session's history count, discussed topics, and tracked entities.
    """
    context = enhanced_rag_engine.get_session_context(session_id)
    return SessionInfo(
        session_id=session_id,
        history_count=context.get('history_count', 0),
        topics=context.get('topics', []),
        entities=context.get('entities', {})
    )


@app.delete("/session/{session_id}", summary="Clear Session")
def clear_session(session_id: str):
    """
    Clear a conversation session's history.
    
    Use this to start a fresh conversation.
    """
    enhanced_rag_engine.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}


@app.post("/analyze", summary="Analyze Query")
def analyze_query(question: Question):
    """
    Analyze a query without generating a response.
    
    Useful for debugging or understanding how the system interprets queries.
    Returns information about multi-part detection, ambiguity, intent, etc.
    """
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(question.query)
    
    return {
        "query": question.query,
        "is_multi_part": analysis['is_multi_part'],
        "sub_questions": analysis['sub_questions'],
        "is_ambiguous": analysis['is_ambiguous'],
        "ambiguous_terms": analysis['ambiguous_terms'],
        "intent": analysis['intent'],
        "confidence": analysis['confidence'],
    }
