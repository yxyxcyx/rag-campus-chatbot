# evaluate.py

"""
RAG Pipeline Evaluation

This script runs a comprehensive evaluation of the RAG pipeline using the Ragas
framework. It measures key performance metrics to guide iterative improvements

The evaluation process is as follows:
1.  Load an evaluation dataset (question and ground_truth pairs)
2.  For each question, run the full RAG pipeline (retrieve, rerank, generate)
    to get an answer and the context used
3.  Use the Ragas library to score the generated answer against the ground truth
    and context based on four key metrics:
    -   Faithfulness: How factually consistent is the answer with the context?
    -   Answer Relevancy: How relevant is the answer to the question?
    -   Context Recall: Was all necessary information retrieved from the context?
    -   Context Precision: Was the retrieved context free of irrelevant information?
4.  The evaluation is run iteratively to avoid API rate limits
5.  Results are saved to a timestamped CSV file for analysis
"""


import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from rag_pipeline import (
    load_documents_from_folder,
    chunk_text,
    embed_chunks,
    retrieve_and_rerank,
    retrieve_and_rerank_hybrid,
    generate_response
)
from rank_bm25 import BM25Okapi

# SECTION 1: SETUP
print("Starting RAG evaluation pipeline...")
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")

print("  - Initializing models...")
# The "Judge" LLM for Ragas
judge_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # Same family as the main RAG pipeline
    groq_api_key=groq_api_key,
    timeout=60.0,
    max_retries=5,
    max_tokens=1024
)
# Calculate semantic similarity
ragas_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

print("  - Loading Cross-Encoder re-ranking model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("  - Connecting to vector database...")
chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
collection_name = os.getenv("COLLECTION_NAME", "collection")
client = chromadb.PersistentClient(path=chroma_db_path)
collection = client.get_or_create_collection(name=collection_name)

current_count = collection.count()
if current_count == 0:
    print("  - Database is empty. Running one-time data ingestion pipeline...")
    documents_dict = load_documents_from_folder("data")
    if documents_dict:
        # Sentence-window chunking returns (central_sentences, windows)
        central_sentences, windows = chunk_text(documents_dict)

        print(f"  - Created {len(windows)} sentence windows for evaluation store.")
        vector_embeddings = embed_chunks(central_sentences, ragas_embeddings)
        ids = [f"sw_eval_{i}" for i in range(len(windows))]

        collection.add(embeddings=vector_embeddings, documents=windows, ids=ids)
        print("  - PIPELINE COMPLETE. Database is now populated.")
        print(f"  - Total chunks/windows in collection '{collection_name}': {collection.count()}.")
    else:
        raise ValueError("No text could be extracted from the 'data' folder.")
else:
    print(f"  - Database already populated with {current_count} chunks/windows in collection '{collection_name}'.")

# Load retrieval configuration
ENABLE_HYBRID_SEARCH = os.getenv('ENABLE_HYBRID_SEARCH', 'false').lower() == 'true'
BM25_WEIGHT = float(os.getenv('BM25_WEIGHT', '0.3'))
VECTOR_WEIGHT = float(os.getenv('VECTOR_WEIGHT', '0.7'))
USE_DIVERSITY_FILTER = os.getenv('USE_DIVERSITY_FILTER', 'true').lower() == 'true'
DIVERSITY_THRESHOLD = float(os.getenv('DIVERSITY_THRESHOLD', '0.85'))

print("\n  - Retrieval configuration:")
print(f"    - Hybrid search: {'ENABLED' if ENABLE_HYBRID_SEARCH else 'DISABLED'}")
print(f"    - Diversity filter: {'ENABLED' if USE_DIVERSITY_FILTER else 'DISABLED'} (threshold: {DIVERSITY_THRESHOLD})")

# Initialize BM25 index if hybrid search is enabled
bm25_index = None
all_documents = None

if ENABLE_HYBRID_SEARCH and current_count > 0:
    print("  - Initializing BM25 index for hybrid search...")
    try:
        # Fetch all documents from ChromaDB for BM25
        all_results = collection.get(limit=current_count)
        all_documents = all_results.get("documents", [])
        
        if all_documents:
            # Tokenize documents for BM25
            tokenized_docs = [doc.lower().split() for doc in all_documents]
            bm25_index = BM25Okapi(tokenized_docs)
            print(f"    - BM25 index initialized with {len(all_documents)} documents")
            print(f"    - Weights: BM25={BM25_WEIGHT}, Vector={VECTOR_WEIGHT}")
        else:
            print("    - WARNING: Could not fetch documents for BM25 index")
            ENABLE_HYBRID_SEARCH = False
    except Exception as e:
        print(f"    - ERROR initializing BM25: {e}")
        print("    - Falling back to vector-only search")
        ENABLE_HYBRID_SEARCH = False

# SECTION 2: DATA PREPARATION
print("Preparing evaluation data...")
print("  - Loading evaluation questions and ground truths...")
eval_dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests', 'eval_dataset.json')
with open(eval_dataset_path, 'r') as f:
    eval_data = json.load(f)
questions = [item['question'] for item in eval_data]
ground_truths = [item['ground_truth'] for item in eval_data]

# Generate answer
print("  - Generating answers for evaluation questions...")
answers = []
contexts = []
for i, query in enumerate(questions):
    print(f"    - Processing question {i+1}/{len(questions)}: '{query[:50]}...'")
    
    # Use appropriate retrieval method based on configuration
    if ENABLE_HYBRID_SEARCH and bm25_index is not None:
        retrieved_chunks = retrieve_and_rerank_hybrid(
            query, 
            ragas_embeddings, 
            collection, 
            cross_encoder,
            bm25_index=bm25_index,
            documents=all_documents,
            n_initial=30,
            n_final=5,
            bm25_weight=BM25_WEIGHT,
            vector_weight=VECTOR_WEIGHT,
            use_diversity_filter=USE_DIVERSITY_FILTER,
            diversity_threshold=DIVERSITY_THRESHOLD
        )
    else:
        retrieved_chunks = retrieve_and_rerank(
            query, 
            ragas_embeddings, 
            collection, 
            cross_encoder,
            n_initial=20,
            n_final=5,
            use_diversity_filter=USE_DIVERSITY_FILTER,
            diversity_threshold=DIVERSITY_THRESHOLD
        )
    
    contexts.append(retrieved_chunks)
    context_str = "\n\n".join(retrieved_chunks)
    
    try:
        generated_answer = generate_response(context_str, query)
        answers.append(generated_answer)
        print("      - Success.")
    except Exception as e:
        print(f"      - FAILED to generate answer. Error: {e}")
        answers.append("Error: Could not generate answer.")
    time.sleep(2) # Manage API rate limits

response_dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

# SECTION 3: RAGAS EVALUATION
print("Running RAGAS evaluation...")
metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
all_results_data = []


def _extract_score(metric_value):
    """Extract a plain float score from a RAGAS metric value.

    Handles values that may be lists/arrays of length 1, numpy scalars,
    or objects with a `.score` attribute. Returns NaN if a numeric
    value cannot be obtained.
    """

    import math

    try:
        value = metric_value

        # If it's a list/tuple/array, use the first element
        if isinstance(value, (list, tuple)):
            if not value:
                return float("nan")
            value = value[0]
        else:
            # Some RAGAS types may be indexable like arrays
            try:
                value = value[0]
            except Exception:
                pass

        # If it has a `.score` attribute, use that
        if hasattr(value, "score"):
            value = value.score

        # If it's a dict with 'score', use that
        if isinstance(value, dict) and "score" in value:
            value = value["score"]

        # Convert numpy or Python numeric types to float
        return float(value)

    except Exception:
        return float("nan")

# Evaluate one question at a time (avoid rate limits and handle errors)
for i, row in enumerate(response_dataset):
    print(f"  - Evaluating question {i+1}/{len(response_dataset)}: '{row['question'][:50]}...'")
    
    # Ragas expects a Dataset object for evaluation
    row_dataset = Dataset.from_dict({k: [v] for k, v in row.items()})
    
    try:
        result = evaluate(
            dataset=row_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=True
        )

        # Store the scores as plain floats
        result_data = {
            'question': row['question'],
            'faithfulness': _extract_score(result['faithfulness']),
            'answer_relevancy': _extract_score(result['answer_relevancy']),
            'context_recall': _extract_score(result['context_recall']),
            'context_precision': _extract_score(result['context_precision'])
        }
        all_results_data.append(result_data)
        print("    - Success.")

    except Exception as e:
        print(f"    - FAILED to evaluate. Error: {e}")
        failed_result_data = {
            'question': row['question'],
            'faithfulness': 'ERROR', 'answer_relevancy': 'ERROR',
            'context_recall': 'ERROR', 'context_precision': 'ERROR'
        }
        all_results_data.append(failed_result_data)

    time.sleep(10) # Proactive sleep to manage API rate limits for the judge LLM

# SECTION 4: SAVE RESULTS
print("Saving evaluation results...")
df_results = pd.DataFrame(all_results_data)
results_folder = "evaluation_results"
os.makedirs(results_folder, exist_ok=True)

# Timestamped
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"ragas_results_{timestamp}.csv"
filepath = os.path.join(results_folder, filename)

# Save
df_results.to_csv(filepath, index=False, na_rep="NaN")
print("\n--- RAGAS EVALUATION COMPLETE ---")
print("Results:")
print(df_results)
print(f"\nResults successfully saved to: {filepath}")