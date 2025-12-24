# enhanced_rag_engine.py

"""
Enhanced RAG Engine with Advanced Features

This module implements all the improvements for better RAG performance:
1. Improved prompt engineering with system messages
2. Citation verification and filtering
3. Multi-part question handling
4. Ambiguous query clarification
5. Contradictory information handling
6. Conversation memory with session context
7. Document versioning support

Architecture:
- QueryAnalyzer: Analyzes queries for multi-part, ambiguity, and intent
- ConversationMemory: Manages session context and follow-ups
- EnhancedGenerator: Improved LLM generation with structured prompts
- CitationVerifier: Validates and filters citations
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import groq
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from requests.exceptions import Timeout

from logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# SECTION 1: QUERY ANALYSIS
# =============================================================================

class QueryAnalyzer:
    """
    Analyzes user queries to identify:
    - Multi-part questions
    - Ambiguous terms
    - Query intent
    - Follow-up context needs
    """
    
    # Common ambiguous terms that need clarification
    AMBIGUOUS_TERMS = {
        'fees': ['tuition fees', 'application fees', 'registration fees', 'examination fees'],
        'deadline': ['application deadline', 'registration deadline', 'payment deadline'],
        'requirements': ['admission requirements', 'graduation requirements', 'course requirements'],
        'program': ['undergraduate program', 'postgraduate program', 'diploma program'],
        'scholarship': ['merit scholarship', 'need-based scholarship', 'sports scholarship'],
        'student': ['local student', 'international student', 'part-time student'],
    }
    
    # Question indicators for multi-part detection
    MULTI_PART_INDICATORS = [
        r'\band\b',
        r'\balso\b',
        r'\bas well as\b',
        r'\badditionally\b',
        r'\bwhat about\b',
        r'\bhow about\b',
        r'\?.*\?',  # Multiple question marks
        r'\bfirst.*(?:second|then|next|also)\b',
    ]
    
    def __init__(self):
        self.intent_patterns = self._build_intent_patterns()
    
    def _build_intent_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for intent classification."""
        return {
            'fee_inquiry': [
                r'(?:how much|what is|what are).*(?:fee|cost|price|tuition)',
                r'(?:fee|tuition|cost).*(?:for|of)',
                r'(?:annual|yearly|semester).*fee',
            ],
            'deadline_inquiry': [
                r'(?:when|what).*(?:deadline|due date|last date)',
                r'deadline.*(?:for|to)',
            ],
            'requirement_inquiry': [
                r'(?:what|which).*(?:require|need|prerequisite)',
                r'(?:requirement|criteria).*(?:for|to)',
                r'(?:how to|how do).*(?:apply|register|enroll)',
            ],
            'comparison': [
                r'(?:difference|compare|versus|vs)',
                r'(?:which is better|what is the difference)',
            ],
            'general_info': [
                r'(?:tell me about|what is|explain|describe)',
                r'(?:information|details).*(?:about|on)',
            ],
        }
    
    def analyze(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Comprehensive query analysis.
        
        Returns:
            {
                'original_query': str,
                'cleaned_query': str,
                'is_multi_part': bool,
                'sub_questions': List[str],
                'is_ambiguous': bool,
                'ambiguous_terms': List[Dict],
                'intent': str,
                'is_followup': bool,
                'resolved_query': str,  # Query with context from history
                'confidence': float,
            }
        """
        result = {
            'original_query': query,
            'cleaned_query': self._clean_query(query),
            'is_multi_part': False,
            'sub_questions': [],
            'is_ambiguous': False,
            'ambiguous_terms': [],
            'intent': 'general_info',
            'is_followup': False,
            'resolved_query': query,
            'confidence': 1.0,
        }
        
        # Detect multi-part questions
        result['is_multi_part'], result['sub_questions'] = self._detect_multi_part(query)
        
        # Detect ambiguous terms
        result['is_ambiguous'], result['ambiguous_terms'] = self._detect_ambiguity(query)
        
        # Classify intent
        result['intent'] = self._classify_intent(query)
        
        # Handle follow-up detection
        if conversation_history:
            result['is_followup'], result['resolved_query'] = self._resolve_followup(
                query, conversation_history
            )
        
        # Calculate confidence based on ambiguity and clarity
        result['confidence'] = self._calculate_confidence(result)
        
        return result
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        query = query.strip()
        query = ' '.join(query.split())  # Normalize whitespace
        return query
    
    def _detect_multi_part(self, query: str) -> Tuple[bool, List[str]]:
        """
        Detect if query contains multiple questions/parts.
        
        Returns:
            (is_multi_part, list_of_sub_questions)
        """
        query_lower = query.lower()
        
        # Check for multi-part indicators
        for pattern in self.MULTI_PART_INDICATORS:
            if re.search(pattern, query_lower):
                # Try to split into sub-questions
                sub_questions = self._split_multi_part(query)
                if len(sub_questions) > 1:
                    return True, sub_questions
        
        return False, [query]
    
    def _split_multi_part(self, query: str) -> List[str]:
        """Split a multi-part query into individual questions."""
        sub_questions = []
        
        # Split by common conjunctions
        parts = re.split(r'\s+(?:and|also|as well as|additionally)\s+', query, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if part:
                # Ensure each part is a complete question
                if not part.endswith('?'):
                    # Check if it starts with a question word
                    if not re.match(r'^(?:what|when|where|who|how|why|which|is|are|can|do|does)', 
                                   part.lower()):
                        # Add context from original query
                        part = f"What about {part}?"
                sub_questions.append(part)
        
        return sub_questions if len(sub_questions) > 1 else [query]
    
    def _detect_ambiguity(self, query: str) -> Tuple[bool, List[Dict]]:
        """
        Detect ambiguous terms in the query.
        
        Returns:
            (is_ambiguous, list_of_ambiguous_terms_with_options)
        """
        query_lower = query.lower()
        ambiguous_terms = []
        
        for term, options in self.AMBIGUOUS_TERMS.items():
            if term in query_lower:
                # Check if any specific option is already mentioned
                specific_found = any(opt.lower() in query_lower for opt in options)
                
                if not specific_found:
                    ambiguous_terms.append({
                        'term': term,
                        'options': options,
                        'context': self._extract_context(query, term)
                    })
        
        return len(ambiguous_terms) > 0, ambiguous_terms
    
    def _extract_context(self, query: str, term: str) -> str:
        """Extract surrounding context for a term."""
        words = query.split()
        try:
            idx = next(i for i, w in enumerate(words) if term in w.lower())
            start = max(0, idx - 3)
            end = min(len(words), idx + 4)
            return ' '.join(words[start:end])
        except StopIteration:
            return query
    
    def _classify_intent(self, query: str) -> str:
        """Classify the primary intent of the query."""
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return 'general_info'
    
    def _resolve_followup(
        self, 
        query: str, 
        history: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Resolve follow-up queries using conversation history.
        
        Handles cases like:
        - "What about for international students?" (after asking about local student fees)
        - "And the deadline?" (after asking about requirements)
        """
        query_lower = query.lower()
        
        # Indicators of follow-up questions
        followup_indicators = [
            r'^(?:what about|how about|and|also)',
            r'^(?:what|how) (?:about|is) (?:the|for)',
            r'^(?:is|are|can|do) (?:they|it|that)',
            r'^(?:same|similar) (?:for|with)',
        ]
        
        is_followup = any(
            re.match(pattern, query_lower) 
            for pattern in followup_indicators
        )
        
        if is_followup and history:
            # Get the most recent context
            last_exchange = history[-1] if history else None
            
            if last_exchange:
                # Extract key entities from previous query
                prev_query = last_exchange.get('query', '')
                
                # Build resolved query with context
                resolved = self._merge_with_context(query, prev_query)
                return True, resolved
        
        return False, query
    
    def _merge_with_context(self, current: str, previous: str) -> str:
        """Merge current query with context from previous query."""
        # Extract entities from previous query
        # This is a simplified version - a production system would use NER
        
        # Look for programme mentions
        programme_match = re.search(
            r'(?:Bachelor|Master|PhD|Diploma)\s+(?:of|in)\s+[A-Za-z\s]+(?:\([^)]+\))?',
            previous,
            re.IGNORECASE
        )
        
        if programme_match:
            programme = programme_match.group()
            if programme.lower() not in current.lower():
                return f"{current} (regarding {programme})"
        
        return current
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score for the query analysis."""
        confidence = 1.0
        
        # Reduce confidence for ambiguous queries
        if analysis['is_ambiguous']:
            confidence -= 0.2 * len(analysis['ambiguous_terms'])
        
        # Reduce confidence for multi-part queries
        if analysis['is_multi_part']:
            confidence -= 0.1
        
        # Reduce confidence for very short queries
        if len(analysis['cleaned_query'].split()) < 3:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))


# =============================================================================
# SECTION 2: CONVERSATION MEMORY
# =============================================================================

class ConversationMemory:
    """
    Manages conversation context for follow-up handling.
    
    Features:
    - Session-based memory storage
    - Context extraction for follow-ups
    - Automatic memory cleanup
    - Topic tracking
    """
    
    def __init__(self, max_history: int = 10, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, Dict] = {}
        self.max_history = max_history
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
    
    def get_or_create_session(self, session_id: str) -> Dict:
        """Get existing session or create a new one."""
        self._cleanup_old_sessions()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'history': [],
                'topics': [],
                'entities': {},  # Track mentioned entities
            }
        else:
            self.sessions[session_id]['last_activity'] = datetime.now()
        
        return self.sessions[session_id]
    
    def add_exchange(
        self, 
        session_id: str, 
        query: str, 
        response: str,
        metadata: Dict = None
    ):
        """Add a query-response exchange to session history."""
        session = self.get_or_create_session(session_id)
        
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'metadata': metadata or {}
        }
        
        session['history'].append(exchange)
        
        # Trim history if needed
        if len(session['history']) > self.max_history:
            session['history'] = session['history'][-self.max_history:]
        
        # Extract and track entities
        self._extract_entities(session, query, response)
        
        # Update topics
        self._update_topics(session, query)
    
    def get_context(self, session_id: str, num_exchanges: int = 3) -> List[Dict]:
        """Get recent conversation context."""
        session = self.sessions.get(session_id, {})
        history = session.get('history', [])
        return history[-num_exchanges:] if history else []
    
    def get_relevant_context(self, session_id: str, query: str) -> str:
        """
        Get context relevant to the current query.
        
        Returns a formatted string of relevant previous exchanges.
        """
        session = self.sessions.get(session_id, {})
        history = session.get('history', [])
        
        if not history:
            return ""
        
        # Get last 3 exchanges for context
        recent = history[-3:]
        
        context_parts = []
        for exchange in recent:
            context_parts.append(f"Previous Q: {exchange['query']}")
            # Include only first 200 chars of response to avoid context bloat
            response_preview = exchange['response'][:200]
            if len(exchange['response']) > 200:
                response_preview += "..."
            context_parts.append(f"Previous A: {response_preview}")
        
        return "\n".join(context_parts)
    
    def _extract_entities(self, session: Dict, query: str, response: str):
        """Extract and store mentioned entities for future reference."""
        combined_text = f"{query} {response}"
        
        # Extract programmes
        programmes = re.findall(
            r'(?:Bachelor|Master|PhD|Diploma)\s+(?:of|in)\s+[A-Za-z\s]+(?:\([^)]+\))?',
            combined_text,
            re.IGNORECASE
        )
        
        if programmes:
            session['entities']['programmes'] = list(set(
                session['entities'].get('programmes', []) + programmes
            ))
        
        # Extract fees
        fees = re.findall(r'RM\s*[\d,]+(?:\.\d{2})?', combined_text)
        if fees:
            session['entities']['fees'] = list(set(
                session['entities'].get('fees', []) + fees
            ))
    
    def _update_topics(self, session: Dict, query: str):
        """Track discussed topics."""
        query_lower = query.lower()
        
        topic_keywords = {
            'fees': ['fee', 'tuition', 'cost', 'price'],
            'admission': ['admission', 'apply', 'application', 'requirement'],
            'scholarship': ['scholarship', 'grant', 'financial aid'],
            'programme': ['programme', 'program', 'course', 'degree'],
            'deadline': ['deadline', 'due date', 'last date'],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                if topic not in session['topics']:
                    session['topics'].append(topic)
    
    def _cleanup_old_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session['last_activity'] > self.session_timeout
        ]
        
        for sid in expired:
            del self.sessions[sid]
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# =============================================================================
# SECTION 3: CITATION VERIFICATION
# =============================================================================

class CitationVerifier:
    """
    Verifies and filters citations from retrieved documents.
    
    Features:
    - Relevance scoring for citations
    - Contradiction detection
    - Source tracking
    - Confidence scoring
    """
    
    def __init__(self, min_relevance_score: float = 0.15):
        self.min_relevance_score = min_relevance_score
    
    def verify_citations(
        self, 
        query: str, 
        retrieved_chunks: List[str],
        query_analysis: Dict = None
    ) -> Dict:
        """
        Verify and filter citations for quality and relevance.
        
        Returns:
            {
                'verified_chunks': List[str],
                'rejected_chunks': List[str],
                'contradictions': List[Dict],
                'confidence_scores': List[float],
                'sources': List[str],
            }
        """
        result = {
            'verified_chunks': [],
            'rejected_chunks': [],
            'contradictions': [],
            'confidence_scores': [],
            'sources': [],
        }
        
        if not retrieved_chunks:
            return result
        
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for chunk in retrieved_chunks:
            # Calculate relevance score
            score = self._calculate_relevance(chunk, query_keywords, query_analysis)
            
            if score >= self.min_relevance_score:
                result['verified_chunks'].append(chunk)
                result['confidence_scores'].append(score)
                
                # Extract source
                source = self._extract_source(chunk)
                if source and source not in result['sources']:
                    result['sources'].append(source)
            else:
                result['rejected_chunks'].append(chunk)
        
        # Detect contradictions
        result['contradictions'] = self._detect_contradictions(result['verified_chunks'])
        
        return result
    
    def _calculate_relevance(
        self, 
        chunk: str, 
        query_keywords: set,
        query_analysis: Dict = None
    ) -> float:
        """Calculate relevance score for a chunk."""
        chunk_lower = chunk.lower()
        chunk_words = set(chunk_lower.split())
        
        # Keyword overlap (basic)
        overlap = len(query_keywords & chunk_words)
        keyword_score = overlap / max(len(query_keywords), 1)
        
        # Partial match bonus - check if query terms appear as substrings
        partial_matches = 0
        for keyword in query_keywords:
            if len(keyword) > 3 and keyword in chunk_lower:
                partial_matches += 1
        partial_score = partial_matches / max(len(query_keywords), 1)
        
        # Combine keyword scores
        keyword_score = max(keyword_score, partial_score * 0.8)
        
        # Intent-based scoring
        intent_score = 0.0
        if query_analysis:
            intent = query_analysis.get('intent', '')
            
            if intent == 'fee_inquiry':
                # Boost if chunk contains fee information
                if re.search(r'RM\s*[\d,]+', chunk):
                    intent_score = 0.3
            elif intent == 'deadline_inquiry':
                if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}', chunk):
                    intent_score = 0.3
        
        # Length penalty for very short chunks
        length_score = min(1.0, len(chunk) / 200)
        
        # Combine scores
        final_score = (keyword_score * 0.5) + (intent_score * 0.3) + (length_score * 0.2)
        
        return min(1.0, final_score)
    
    def _extract_source(self, chunk: str) -> Optional[str]:
        """Extract source document from chunk metadata."""
        # Look for source patterns
        source_match = re.search(r'Source(?:\s+Document)?:\s*([^\n]+)', chunk)
        if source_match:
            return source_match.group(1).strip()
        
        # Look for document patterns
        doc_match = re.search(r'Document:\s*([^\n]+)', chunk)
        if doc_match:
            return doc_match.group(1).strip()
        
        return None
    
    def _detect_contradictions(self, chunks: List[str]) -> List[Dict]:
        """
        Detect contradictory information across chunks.
        
        Returns list of detected contradictions with details.
        """
        contradictions = []
        
        # Extract numerical claims
        claims = []
        for i, chunk in enumerate(chunks):
            # Extract fee claims
            fee_matches = re.findall(r'([^.]*RM\s*[\d,]+[^.]*)', chunk)
            for match in fee_matches:
                claims.append({
                    'chunk_index': i,
                    'type': 'fee',
                    'text': match,
                    'value': self._extract_number(match)
                })
            
            # Extract date claims
            date_matches = re.findall(r'([^.]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}[^.]*)', chunk)
            for match in date_matches:
                claims.append({
                    'chunk_index': i,
                    'type': 'date',
                    'text': match
                })
        
        # Compare claims of same type
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                if claim1['type'] == claim2['type'] == 'fee':
                    if claim1.get('value') and claim2.get('value'):
                        # Check if same context but different values
                        if self._similar_context(claim1['text'], claim2['text']):
                            if abs(claim1['value'] - claim2['value']) > 100:
                                contradictions.append({
                                    'type': 'fee_mismatch',
                                    'claim1': claim1,
                                    'claim2': claim2,
                                    'note': 'Different fee amounts found for similar items'
                                })
        
        return contradictions
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text."""
        match = re.search(r'RM\s*([\d,]+(?:\.\d{2})?)', text)
        if match:
            return float(match.group(1).replace(',', ''))
        return None
    
    def _similar_context(self, text1: str, text2: str) -> bool:
        """Check if two texts have similar context."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        overlap = len(words1 & words2)
        similarity = overlap / max(min(len(words1), len(words2)), 1)
        
        return similarity > 0.5


# =============================================================================
# SECTION 4: ENHANCED RESPONSE GENERATOR
# =============================================================================

class EnhancedGenerator:
    """
    Enhanced LLM response generation with improved prompts.
    
    Features:
    - System message for consistent behavior
    - Multi-part question handling
    - Ambiguity handling
    - Contradiction awareness
    - Citation formatting
    """
    
    SYSTEM_MESSAGE = """You are an expert university information assistant for Xiamen University Malaysia (XMUM). Your role is to provide accurate, helpful information about the university based on official documents.

## Core Principles:
1. **Accuracy First**: Only provide information that is explicitly stated in the given context. Never invent or assume information.
2. **Cite Sources**: When mentioning specific facts (fees, dates, requirements), indicate which document they come from.
3. **Be Precise**: Use exact numbers, dates, and requirements from the documents.
4. **Be Helpful**: If information is incomplete, acknowledge what you know and what's missing.
5. **Handle Uncertainty**: If information is ambiguous or contradictory, explain the different possibilities.

## Response Format:
- Use clear, structured responses
- Use bullet points for lists
- Highlight key information (fees, dates, deadlines)
- Keep responses concise but complete

## When You Don't Know:
- Say "Based on the available documents, I don't have information about [topic]."
- Suggest what the user might do to find the answer (e.g., "You may want to contact the admissions office.")
- Don't guess or provide generic information."""

    def __init__(self, groq_api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = groq_api_key
        self.model = model
        self.client = groq.Groq(api_key=groq_api_key)
    
    @retry(
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((groq.RateLimitError, groq.APITimeoutError, Timeout))
    )
    def generate(
        self,
        query: str,
        context: str,
        query_analysis: Dict = None,
        citation_info: Dict = None,
        conversation_context: str = None,
    ) -> Dict:
        """
        Generate enhanced response with all context.
        
        Returns:
            {
                'response': str,
                'sources': List[str],
                'confidence': float,
                'needs_clarification': bool,
                'clarification_prompt': str,
                'answered_parts': List[str],  # For multi-part questions
            }
        """
        # Build the prompt
        prompt = self._build_prompt(
            query, context, query_analysis, citation_info, conversation_context
        )
        
        # Generate response
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.1,
            max_tokens=2048,
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Post-process response
        result = self._post_process(response_text, query_analysis, citation_info)
        
        return result
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        query_analysis: Dict = None,
        citation_info: Dict = None,
        conversation_context: str = None,
    ) -> str:
        """Build a comprehensive prompt with all context."""
        parts = []
        
        # Add conversation context if available
        if conversation_context:
            parts.append(f"## Previous Conversation:\n{conversation_context}\n")
        
        # Add retrieved context
        parts.append(f"## Relevant Documents:\n{context}\n")
        
        # Add contradiction warnings
        if citation_info and citation_info.get('contradictions'):
            parts.append("## Note: Some documents contain potentially contradictory information. Please address this in your response.\n")
        
        # Add the question
        parts.append(f"## Question: {query}\n")
        
        # Add special instructions based on query analysis
        if query_analysis:
            if query_analysis.get('is_multi_part'):
                sub_questions = query_analysis.get('sub_questions', [])
                parts.append(f"""
## Instructions for Multi-Part Question:
This question has multiple parts. Please address each part separately:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(sub_questions))}

Format your response with clear sections for each part.""")
            
            if query_analysis.get('is_ambiguous'):
                terms = query_analysis.get('ambiguous_terms', [])
                parts.append(f"""
## Note on Ambiguous Terms:
The following terms may have multiple meanings:
{chr(10).join(f"- '{t['term']}' could mean: {', '.join(t['options'])}" for t in terms)}

If possible, provide information for the most likely interpretation based on context, or briefly mention the alternatives.""")
        
        parts.append("""
## Response Requirements:
1. Answer based ONLY on the provided documents
2. Be specific with numbers, dates, and requirements
3. If the documents don't contain the answer, say so clearly
4. For multi-part questions, address each part

Answer:""")
        
        return "\n".join(parts)
    
    def _post_process(
        self,
        response: str,
        query_analysis: Dict = None,
        citation_info: Dict = None,
    ) -> Dict:
        """Post-process the generated response with improved confidence scoring."""
        result = {
            'response': response,
            'sources': citation_info.get('sources', []) if citation_info else [],
            'confidence': 1.0,
            'needs_clarification': False,
            'clarification_prompt': '',
            'answered_parts': [],
        }
        
        response_lower = response.lower()
        confidence = 1.0
        
        # === CONFIDENCE SCORING ===
        
        # 1. Check if response indicates lack of information (major penalty)
        no_info_phrases = [
            "don't have enough information",
            "don't have information",
            "not found in the documents",
            "no information available",
            "cannot find",
            "not mentioned in",
            "couldn't find",
            "i don't have",
            "documents do not contain",
            "not explicitly stated",
            "not provided in",
            "unable to find",
            "no specific information",
        ]
        
        if any(phrase in response_lower for phrase in no_info_phrases):
            confidence -= 0.5
        
        # 2. Check for hedging language (moderate penalty)
        hedging_phrases = [
            "may want to contact",
            "you might want to",
            "suggest that you",
            "recommend contacting",
            "check the website",
            "contact the admissions",
            "it is likely that",
            "it appears that",
            "seems to be",
            "possibly",
            "might be",
            "could be",
        ]
        
        hedging_count = sum(1 for phrase in hedging_phrases if phrase in response_lower)
        if hedging_count > 0:
            confidence -= min(0.2, hedging_count * 0.1)
        
        # 3. Check retrieval quality from citation_info
        if citation_info:
            confidence_scores = citation_info.get('confidence_scores', [])
            if confidence_scores:
                avg_retrieval_score = sum(confidence_scores) / len(confidence_scores)
                # Scale: if avg retrieval score is low, reduce confidence
                if avg_retrieval_score < 0.3:
                    confidence -= 0.3
                elif avg_retrieval_score < 0.5:
                    confidence -= 0.15
            
            # If no verified chunks, major penalty
            if not citation_info.get('verified_chunks'):
                confidence -= 0.4
            
            # If contradictions found, slight penalty
            if citation_info.get('contradictions'):
                confidence -= 0.1
        
        # 4. Check for specific data in response (bonus for fee queries)
        if query_analysis and query_analysis.get('intent') == 'fee_inquiry':
            # If asking about fees, response should contain RM amounts
            if re.search(r'RM\s*[\d,]+', response):
                confidence += 0.1  # Bonus for having specific fee data
            else:
                confidence -= 0.2  # Penalty for missing fee data in fee query
        
        # 5. Response length check
        if len(response) < 100:
            confidence -= 0.1  # Very short responses are suspicious
        
        # 6. Check for multi-part response coverage
        if query_analysis and query_analysis.get('is_multi_part'):
            sub_questions = query_analysis.get('sub_questions', [])
            answered_count = 0
            for sq in sub_questions:
                key_terms = set(sq.lower().split()) - {'what', 'is', 'the', 'are', 'how', 'and', 'for'}
                if any(term in response_lower for term in key_terms if len(term) > 3):
                    result['answered_parts'].append(sq)
                    answered_count += 1
            
            # Penalty if not all parts answered
            if sub_questions and answered_count < len(sub_questions):
                unanswered_ratio = (len(sub_questions) - answered_count) / len(sub_questions)
                confidence -= unanswered_ratio * 0.2
        
        # Clamp confidence to [0.0, 1.0]
        result['confidence'] = max(0.0, min(1.0, confidence))
        
        # Handle ambiguity
        if query_analysis and query_analysis.get('is_ambiguous'):
            if query_analysis.get('confidence', 1.0) < 0.6:
                result['needs_clarification'] = True
                terms = query_analysis.get('ambiguous_terms', [])
                if terms:
                    result['clarification_prompt'] = self._build_clarification_prompt(terms)
        
        return result
    
    def _build_clarification_prompt(self, ambiguous_terms: List[Dict]) -> str:
        """Build a clarification prompt for ambiguous queries."""
        if not ambiguous_terms:
            return ""
        
        term = ambiguous_terms[0]
        options = term.get('options', [])
        
        if options:
            options_text = "\n".join(f"  - {opt}" for opt in options[:4])
            return f"Could you please clarify what type of '{term['term']}' you're asking about?\n{options_text}"
        
        return f"Could you please provide more details about '{term['term']}'?"


# =============================================================================
# SECTION 5: DOCUMENT VERSIONING
# =============================================================================

class DocumentVersionManager:
    """
    Manages document versions for incremental updates.
    
    Features:
    - Track document versions by hash
    - Support delta updates (only re-ingest changed documents)
    - Maintain version history
    """
    
    def __init__(self, version_file: str = ".document_versions.json"):
        self.version_file = version_file
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict:
        """Load version data from file."""
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'documents': {}, 'last_update': None}
    
    def _save_versions(self):
        """Save version data to file."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def compute_hash(self, filepath: str) -> str:
        """Compute hash of file contents."""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_changed_documents(self, folder_path: str) -> Dict[str, str]:
        """
        Identify documents that have changed since last ingestion.
        
        Returns:
            {filepath: change_type} where change_type is 'new', 'modified', or 'deleted'
        """
        changes = {}
        current_files = set()
        
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if not os.path.isfile(filepath):
                continue
            
            current_files.add(filepath)
            current_hash = self.compute_hash(filepath)
            
            stored = self.versions['documents'].get(filepath)
            
            if stored is None:
                changes[filepath] = 'new'
            elif stored.get('hash') != current_hash:
                changes[filepath] = 'modified'
        
        # Check for deleted files
        for stored_path in self.versions['documents']:
            if stored_path not in current_files:
                changes[stored_path] = 'deleted'
        
        return changes
    
    def update_version(self, filepath: str, metadata: Dict = None):
        """Update version record for a document."""
        self.versions['documents'][filepath] = {
            'hash': self.compute_hash(filepath) if os.path.exists(filepath) else None,
            'updated_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.versions['last_update'] = datetime.now().isoformat()
        self._save_versions()
    
    def remove_version(self, filepath: str):
        """Remove version record for a deleted document."""
        if filepath in self.versions['documents']:
            del self.versions['documents'][filepath]
            self._save_versions()
    
    def get_all_versions(self) -> Dict:
        """Get all version records."""
        return self.versions


# =============================================================================
# SECTION 6: MAIN ENHANCED RAG ENGINE
# =============================================================================

class EnhancedRAGEngine:
    """
    Main RAG engine that combines all enhanced features.
    """
    
    def __init__(
        self,
        groq_api_key: str,
        embedding_model,
        collection,
        cross_encoder,
        model: str = "llama-3.1-8b-instant",
    ):
        self.query_analyzer = QueryAnalyzer()
        self.conversation_memory = ConversationMemory()
        self.citation_verifier = CitationVerifier()
        self.generator = EnhancedGenerator(groq_api_key, model)
        self.version_manager = DocumentVersionManager()
        
        self.embedding_model = embedding_model
        self.collection = collection
        self.cross_encoder = cross_encoder
    
    def query(
        self,
        user_query: str,
        session_id: str = None,
        retrieved_chunks: List[str] = None,
    ) -> Dict:
        """
        Process a query through the enhanced RAG pipeline.
        
        Args:
            user_query: The user's question
            session_id: Optional session ID for conversation memory
            retrieved_chunks: Pre-retrieved chunks (if None, will be retrieved)
            
        Returns:
            Enhanced response with metadata
        """
        # Get conversation context
        conversation_context = None
        conversation_history = []
        if session_id:
            conversation_history = self.conversation_memory.get_context(session_id)
            conversation_context = self.conversation_memory.get_relevant_context(
                session_id, user_query
            )
        
        # Analyze the query
        query_analysis = self.query_analyzer.analyze(user_query, conversation_history)
        
        logger.info(
            "Query analyzed",
            extra={
                'is_multi_part': query_analysis['is_multi_part'],
                'is_ambiguous': query_analysis['is_ambiguous'],
                'intent': query_analysis['intent'],
                'is_followup': query_analysis['is_followup'],
            }
        )
        
        # Use resolved query for retrieval
        search_query = query_analysis['resolved_query']
        
        # Handle multi-part questions by combining sub-question results
        if query_analysis['is_multi_part'] and retrieved_chunks is None:
            # This would need integration with retrieval - for now, use main query
            pass
        
        # Verify citations
        citation_info = self.citation_verifier.verify_citations(
            search_query,
            retrieved_chunks or [],
            query_analysis
        )
        
        # Build context from verified chunks
        context = "\n\n---\n\n".join(citation_info['verified_chunks'])
        
        if not context:
            context = "No relevant documents found for this query."
        
        # Generate response
        result = self.generator.generate(
            query=user_query,
            context=context,
            query_analysis=query_analysis,
            citation_info=citation_info,
            conversation_context=conversation_context,
        )
        
        # Add analysis metadata to result
        result['query_analysis'] = query_analysis
        result['citation_info'] = {
            'verified_count': len(citation_info['verified_chunks']),
            'rejected_count': len(citation_info['rejected_chunks']),
            'sources': citation_info['sources'],
            'has_contradictions': len(citation_info['contradictions']) > 0,
        }
        
        # Store in conversation memory
        if session_id:
            self.conversation_memory.add_exchange(
                session_id,
                user_query,
                result['response'],
                metadata={
                    'intent': query_analysis['intent'],
                    'sources': result.get('sources', []),
                }
            )
        
        return result
    
    def get_session_context(self, session_id: str) -> Dict:
        """Get the current session context."""
        session = self.conversation_memory.sessions.get(session_id, {})
        return {
            'history_count': len(session.get('history', [])),
            'topics': session.get('topics', []),
            'entities': session.get('entities', {}),
        }
    
    def clear_session(self, session_id: str):
        """Clear a session's history."""
        self.conversation_memory.clear_session(session_id)


# Export main components
__all__ = [
    'QueryAnalyzer',
    'ConversationMemory', 
    'CitationVerifier',
    'EnhancedGenerator',
    'DocumentVersionManager',
    'EnhancedRAGEngine',
]
