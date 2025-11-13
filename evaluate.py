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

from rag_pipeline import (
    load_documents_from_folder,
    chunk_text,
    embed_chunks,
    retrieve_and_rerank,
    generate_response
)

# SECTION 1: SETUP
print("Starting RAG evaluation pipeline...")
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file. Please set it.")

print("  - Initializing models...")
# The "Judge" LLM for Ragas
judge_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # Updated from decommissioned llama3-8b-8192
    groq_api_key=groq_api_key,
    timeout=60.0,
    max_retries=5
)
# Calculate semantic similarity
ragas_embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

print("  - Loading Cross-Encoder re-ranking model...")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print("  - Connecting to vector database...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="xmum_handbook")

if collection.count() == 0:
    print("  - Database is empty. Running one-time data ingestion pipeline...")
    documents_dict = load_documents_from_folder("data")
    if documents_dict:
        text_chunks_with_metadata = chunk_text(documents_dict)
        vector_embeddings = embed_chunks(text_chunks_with_metadata, ragas_embeddings)
        ids = [f"chunk_{i}" for i in range(len(text_chunks_with_metadata))]
        collection.add(embeddings=vector_embeddings, documents=text_chunks_with_metadata, ids=ids)
        print("  - PIPELINE COMPLETE. Database is now populated.")
    else:
        raise ValueError("No text could be extracted from the 'data' folder.")
else:
    print(f"  - Database already populated with {collection.count()} chunks.")

# SECTION 2: DATA PREPARATION
print("Preparing evaluation data...")
print("  - Loading evaluation questions and ground truths...")
with open('eval_dataset.json', 'r') as f:
    eval_data = json.load(f)
questions = [item['question'] for item in eval_data]
ground_truths = [item['ground_truth'] for item in eval_data]

# Generate answer
print("  - Generating answers for evaluation questions...")
answers = []
contexts = []
for i, query in enumerate(questions):
    print(f"    - Processing question {i+1}/{len(questions)}: '{query[:50]}...'")
    retrieved_chunks = retrieve_and_rerank(query, ragas_embeddings, collection, cross_encoder)
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
        
        # Store the scores
        result_data = {
            'question': row['question'],
            'faithfulness': result['faithfulness'],
            'answer_relevancy': result['answer_relevancy'],
            'context_recall': result['context_recall'],
            'context_precision': result['context_precision']
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