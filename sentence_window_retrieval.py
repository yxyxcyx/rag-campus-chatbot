# sentence_window_retrieval.py

"""
State-of-the-Art Sentence-Window Retrieval

This module implements an advanced RAG technique where:
1. Documents are split into individual sentences
2. For each sentence, a "window" is created containing k sentences before and after
3. Only the central sentence is embedded (for precise matching)
4. The full window is stored (for rich context)
5. Retrieval returns the sentence that matches, but provides the full window as context

This approach provides more precise retrieval while maintaining rich context for the LLM.

Reference: "Sentence-Window Retrieval" - A SOTA RAG technique for improved accuracy
"""

import re
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class SentenceWindowRetriever:
    """
    Implements sentence-window retrieval for enhanced RAG performance.
    """
    
    def __init__(self, window_size: int = 3):
        """
        Initialize the sentence-window retriever.
        
        Args:
            window_size: Number of sentences before and after the central sentence
                        to include in the window (default: 3)
        """
        self.window_size = window_size
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize a sentence."""
        # Remove extra whitespace
        sentence = ' '.join(sentence.split())
        # Remove special characters that might interfere
        sentence = sentence.strip()
        return sentence
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK's sentence tokenizer.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Use NLTK's sentence tokenizer (handles edge cases better than simple splitting)
        sentences = sent_tokenize(text)
        
        # Clean sentences
        sentences = [self._clean_sentence(s) for s in sentences if s.strip()]
        
        return sentences
    
    def create_sentence_windows(
        self,
        documents_dict: Dict[str, str]
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Create sentence windows from documents.
        
        For each sentence in each document:
        - Create a window containing k sentences before and after
        - Store the central sentence (for embedding)
        - Store the full window (for context)
        - Store metadata about the sentence position
        
        Args:
            documents_dict: Dictionary of {filename: text}
            
        Returns:
            Tuple of:
            - central_sentences: List of central sentences (to embed)
            - windows: List of window texts (to store)
            - metadata: List of metadata dicts for each window
        """
        central_sentences = []
        windows = []
        metadata = []
        
        for filename, text in documents_dict.items():
            # Split document into sentences
            sentences = self.split_into_sentences(text)
            
            if not sentences:
                continue
            
            # Create windows for each sentence
            for i, sentence in enumerate(sentences):
                # Calculate window boundaries
                start_idx = max(0, i - self.window_size)
                end_idx = min(len(sentences), i + self.window_size + 1)
                
                # Extract window
                window_sentences = sentences[start_idx:end_idx]
                window_text = ' '.join(window_sentences)
                
                # Store central sentence for embedding
                central_sentences.append(sentence)
                
                # Store full window for context
                windows.append(window_text)
                
                # Store metadata
                meta = {
                    'source_document': filename,
                    'sentence_index': i,
                    'total_sentences': len(sentences),
                    'window_start': start_idx,
                    'window_end': end_idx,
                    'central_sentence': sentence,
                    'window_size': self.window_size
                }
                metadata.append(meta)
        
        return central_sentences, windows, metadata
    
    def format_window_for_storage(
        self,
        window: str,
        metadata: Dict,
        include_metadata: bool = True
    ) -> str:
        """
        Format a window with its metadata for storage in the vector database.
        
        Args:
            window: The window text
            metadata: Metadata dictionary
            include_metadata: Whether to include metadata in the stored text
            
        Returns:
            Formatted string for storage
        """
        if not include_metadata:
            return window
        
        # Format with metadata
        formatted = f"""Source Document: {metadata['source_document']}
Sentence {metadata['sentence_index'] + 1} of {metadata['total_sentences']}
Window: [{metadata['window_start']}-{metadata['window_end']}]

{window}"""
        
        return formatted


def chunk_text_with_sentence_windows(
    documents_dict: Dict[str, str],
    window_size: int = 3
) -> Tuple[List[str], List[str]]:
    """
    Enhanced chunking function using sentence-window retrieval.
    
    This replaces the traditional RecursiveCharacterTextSplitter with
    a more sophisticated sentence-window approach.
    
    Args:
        documents_dict: Dictionary of {filename: text}
        window_size: Number of sentences before/after to include in window
        
    Returns:
        Tuple of:
        - sentences_to_embed: List of central sentences (for embedding)
        - windows_to_store: List of formatted windows (for storage)
    """
    retriever = SentenceWindowRetriever(window_size=window_size)
    
    # Create sentence windows
    central_sentences, windows, metadata_list = retriever.create_sentence_windows(
        documents_dict
    )
    
    # Format windows for storage
    formatted_windows = [
        retriever.format_window_for_storage(window, meta)
        for window, meta in zip(windows, metadata_list)
    ]
    
    print(f"📊 Created {len(central_sentences)} sentence windows")
    print(f"   - Window size: ±{window_size} sentences")
    print(f"   - Total sentences to embed: {len(central_sentences)}")
    
    return central_sentences, formatted_windows


# Example usage and comparison
if __name__ == '__main__':
    # Test the sentence-window retrieval
    test_doc = """
    The university offers a wide range of academic programs. Students can choose from 
    undergraduate and graduate degrees. The admission process is competitive and selective.
    All applicants must submit their transcripts and test scores. International students
    need to provide proof of English proficiency. The deadline for applications is March 1st.
    Late applications may be considered on a case-by-case basis.
    """
    
    retriever = SentenceWindowRetriever(window_size=2)
    documents = {'test.txt': test_doc}
    
    central_sentences, windows, metadata = retriever.create_sentence_windows(documents)
    
    print("\n" + "="*70)
    print("SENTENCE-WINDOW RETRIEVAL EXAMPLE")
    print("="*70)
    
    for i, (sentence, window, meta) in enumerate(zip(central_sentences[:3], windows[:3], metadata[:3])):
        print(f"\n--- Window {i+1} ---")
        print(f"Central Sentence: {sentence}")
        print(f"Window Size: {meta['window_end'] - meta['window_start']} sentences")
        print(f"Full Window: {window[:100]}...")
    
    print("\n" + "="*70)
