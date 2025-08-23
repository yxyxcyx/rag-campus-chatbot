# A Retreival-Augmented Generation (RAG) Chatbot

This project is a complete Retrieval-Augmented Generation (RAG) system designed to function as an AI chatbot for Xiamen University Malaysia (XMUM). It answers user questions based on a collection of documents, such as the XMUM student handbook and related materials.

The system is broken down into four key parts:
1.  **Core RAG Logic (`rag_pipeline.py`):** The engine that performs the data processing, retrieval, and answer generation.
2.  **API Server (`main.py`):** A backend service built with FastAPI that exposes the RAG engine to the internet, allowing other applications to "ask" it questions.
3.  **User Interface (`app.py`):** A user-friendly web-based chatbot interface built with Streamlit that communicates with the API server.
4.  **Evaluation (`evaluate.py`):** A pipeline to rigorously test and measure the performance and accuracy of the RAG system using the RAGAS framework.

![Chatbot Interface](UI.png)

## Table of Contents
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Technical Stack](#technical-stack)
- [The Evaluation Journey](#the-evaluation-journey)
  - [Run 1: Baseline Performance](#run-1-baseline-performance)
  - [Run 2: Advanced Chunking](#run-2-advanced-chunking)
  - [Run 3: Final Optimized Pipeline](#run-3-final-optimized-pipeline)

## The Problem

Students and staff often have specific questions about university policies, academic rules, and application procedures. This critical information is typically scattered across a multitude of documents, such as student handbooks, academic calendars, and policy. Finding a precise answer requires navigating this disorganized collection of files, where a simple keyword search is often insufficient for complex or nuanced queries. This project solves that problem by creating a centralized, intelligent interface that can understand and answer questions using this entire collection of documents as its single source of truth.

## The Solution

A Retrieval-Augmented Generation (RAG) pipeline was chosen as the ideal architecture. This approach grounds the Large Language Model (LLM) in a specific set of documents, preventing hallucinations and ensuring that the answers are factual and based solely on the provided context.

The system operates in two main phases: a one-time data ingestion and the real-time query cycle.

#### Phase 1: Data Ingestion & Indexing (One-time Setup)
This process runs automatically when the API server starts if the vector database is empty.
1.  **Load Documents:** The system scans a `data/` folder and extracts text from `.pdf`, `.docx`, and `.txt` files. It uses a hybrid approach: direct text extraction is tried first, with OCR (Tesseract) as a fallback for image-based PDFs.
2.  **Chunk Text:** The extracted text is segmented into small, overlapping chunks of 500 characters. Each chunk is enriched with metadata identifying its source document.
3.  **Embed & Store:** Each text chunk is converted into a numerical vector (embedding) using the `all-MiniLM-L6-v2` model. These embeddings are then stored in a ChromaDB database, creating a searchable knowledge library.

#### Phase 2: Real-time Querying (Answering a Question)
1.  **API Call:** The Streamlit UI sends the user's question to the `/ask` endpoint of the FastAPI backend.
2.  **Two-Stage Retrieval:**
    * **Stage 1 (Broad Retrieval):** The system queries ChromaDB to find the **top 20** text chunks whose embeddings are most semantically similar to the user's question. This is a fast but broad search.
    * **Stage 2 (Precise Re-ranking):** These 20 candidate chunks are passed to a Cross-Encoder model. This more powerful model directly compares the query against each chunk to calculate a precise relevance score, re-ranking them for accuracy.
3.  **Contextual Generation:**
    * The **top 3** most relevant chunks from the re-ranking stage are selected and combined to form the `CONTEXT`.
    * This context is injected into a prompt template along with the user's original question. The prompt instructs the Llama 3 LLM to answer *only* using the provided information.
    * The complete prompt is sent to the Llama 3 model via the Groq API, which generates the final, grounded answer.
4.  **Return Answer:** The response is sent back through the API to the Streamlit UI and displayed to the user.

#### Evaluation Phase
The `evaluate.py` script is a separate, offline tool to measure how well the RAG system is performing.

1.  **Prepare Data:** It loads a predefined set of questions and corresponding ideal "ground truth" answers from an `eval_dataset.json` file.
2.  **Generate Answers:** For each question, it runs the entire RAG pipeline (retrieve, re-rank, generate) to get the system's actual answer and the context it used.
3.  **Evaluate with RAGAs:** It uses the **RAGAs** library to score the system's performance. RAGAs uses another LLM (the "judge") to measure key metrics:
    * **Faithfulness:** Does the answer stick to the provided context?
    * **Answer Relevancy:** Is the answer relevant to the user's question?
    * **Context Precision & Recall:** Was the retrieved context relevant and sufficient to answer the question?
4.  **Save Results:** The evaluation scores for every question are compiled and saved to a timestamped `.csv` file in the `evaluation_results` folder for analysis.


## Technical Stack

- **Backend API:** FastAPI
- **Frontend UI:** Streamlit
- **LLM:** `Llama3-8B` (via Groq API, with `tenacity` for rate limit handling)
- **Vector Database:** ChromaDB
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Re-ranking Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Evaluation Framework:** Ragas (using Llama3 as the judge)
- **Document Processing:** PyMuPDF, `python-docx`, Pytesseract (for OCR)

## The Evaluation Journey

A key focus of this project was using quantitative data to drive improvements. The Ragas framework was used to evaluate the pipeline after each major change, focusing on Context Recall (finding the right info) and Context Precision (ignoring the wrong info).

### Run 1: Baseline Performance

- **Strategy:** The pipeline featured a two-stage retrieval process: a fast vector search followed by CrossEncoder re-ranker. However, a naive chunking strategy is used (chunk_size=1000).
- **Result:** **Poor Context Recall (0.61) and Context Precision (0.556)**. The system frequently failed to find the necessary information, even if it was present in the documents.

### Run 2: Advanced Chunking

- **Hypothesis:** Smaller, more contextually-aware chunks would fix the retrieval failures.
- **Strategy:** Reduced `chunk_size` to 500 and enriched each chunk with source document metadata.
- **Result:** **Excellent Context Recall (0.711)**, but **poor Context Precision (0.561)**. The system found the right information but also retrieved a lot of irrelevant "noise."

### Run 3: Final Optimized Pipeline

- **Hypothesis:** The re-ranker could filter out the noise if it had more options to choose from.
- **Strategy:** Increased the initial retrieval net by increasing the candidate pool from 10 to 20 documents, giving the Cross-Encoder more to analyze and discard.
- **Result:** **Balanced Performance**. Maintained high **Context Recall (0.711)** while significantly improving **Context Precision (0.744)**, resulting in the best-performing version of the pipeline.



    