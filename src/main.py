# main.py

"""
FastAPI Application - Stateless API Server

This script serves a RESTful API to interact with the RAG pipeline.

ARCHITECTURE CHANGE:
This API server is now STATELESS and follows the read-only pattern.
It ONLY serves queries via the `/ask` endpoint.

Data ingestion is handled separately by Celery workers (see ingestion_worker.py).
To ingest documents, use trigger_ingestion.py script.

This separation follows best practices:
- Read path (query): Handled by this API
- Write path (ingestion): Handled by async workers
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
from typing import Optional, List
import groq
from requests.exceptions import Timeout, ConnectionError

from config import validate_startup_config, get_settings
from logging_config import setup_logging, get_logger, set_request_id
from rag_pipeline import (
    retrieve_and_rerank,
    retrieve_and_rerank_hybrid,
    generate_response
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

class Answer(BaseModel):
    response: str

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
    Receives a user query, runs it through the full RAG pipeline,
    and returns the generated answer.
    
    Uses hybrid search (BM25 + Vector) if enabled, otherwise uses enhanced vector search.
    Both methods include diversity filtering to reduce redundancy.
    
    Returns:
        200: Successful response with answer
        503: Service unavailable (LLM timeout or overload)
        500: Internal server error
    """
    user_query = question.query
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log with structured metadata
    logger.info(
        "Query received",
        query_length=len(user_query),
        request_id=request_id,
        hybrid_search=ENABLE_HYBRID_SEARCH
    )
    
    try:
        # 1. Retrieve and rerank with appropriate method
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
        retrieved_context = "\n\n".join(retrieved_chunks)
        
        # 2. Generate response with error handling
        logger.info("Generating LLM response")
        try:
            final_answer = generate_response(retrieved_context, user_query)
        except groq.APITimeoutError as e:
            logger.error(
                "LLM timeout",
                exc_info=True,
                timeout_seconds=settings.llm_timeout_seconds
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "AI service is temporarily unavailable. Please try again in a moment.",
                    "error_code": "LLM_TIMEOUT",
                    "request_id": request_id,
                    "retry_after": 5
                }
            )
        except groq.RateLimitError as e:
            logger.warning(
                "LLM rate limit exceeded",
                error=str(e)
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "AI service is busy. Please try again in a few seconds.",
                    "error_code": "RATE_LIMITED",
                    "request_id": request_id,
                    "retry_after": 10
                }
            )
        except groq.APIConnectionError as e:
            logger.error(
                "LLM connection failed",
                exc_info=True,
                error=str(e)
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Cannot connect to AI service. Please try again later.",
                    "error_code": "LLM_CONNECTION_ERROR",
                    "request_id": request_id
                }
            )
        except groq.InternalServerError as e:
            logger.error(
                "LLM internal error",
                exc_info=True,
                error=str(e)
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "AI service encountered an error. Please try again.",
                    "error_code": "LLM_INTERNAL_ERROR",
                    "request_id": request_id
                }
            )
        
        logger.info(
            "Query processed successfully",
            response_length=len(final_answer),
            request_id=request_id
        )
        
        return {"response": final_answer}
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
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
        # Log full stack trace internally, return sanitized message
        logger.error(
            "Unexpected error during query processing",
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
