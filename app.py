import gradio as gr
import asyncio
import aiohttp
import logging
import math
import io
import numpy as np
from newspaper import Article
import PyPDF2
from collections import Counter
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers.util import pytorch_cos_sim
from enum import Enum
from groq import Groq
import os
from typing import List, Dict, Any, Set, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import cycle
from threading import Lock
import re
from urllib.parse import urlparse
import ollama
#from google.colab import userdata
import socket
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Starting application initialization")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

class GroqKeyManager:
    def __init__(self, api_keys: List[str]):
        """Initialize the key manager with multiple API keys."""
        if not api_keys:
            raise ValueError("At least one API key must be provided")

        self.api_keys = api_keys
        self.key_cycle = cycle(api_keys)
        self.clients = {key: Groq(api_key=key) for key in api_keys}
        self.lock = Lock()
        self.current_key = next(self.key_cycle)

    def get_next_client(self) -> Groq:
        """Get the next Groq client in rotation."""
        with self.lock:
            self.current_key = next(self.key_cycle)
            return self.clients[self.current_key]

    def get_current_client(self) -> Groq:
        """Get the current Groq client."""
        return self.clients[self.current_key]

    @property
    def current_api_key(self) -> str:
        """Get the current API key."""
        return self.current_key

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Initialize Groq key manager
groq_keys = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2")
]
groq_manager = GroqKeyManager(groq_keys)
logger.info("Groq key manager initialized")

def get_ollama_host():
    """
    Try to determine the correct host for Ollama in Colab
    """
    # Try localhost
    try:
        socket.gethostbyname('localhost')
        return 'localhost'
    except:
        pass
    
    # Try 127.0.0.1
    try:
        socket.gethostbyname('127.0.0.1')
        return '127.0.0.1'
    except:
        pass
    
    # Try getting the hostname
    try:
        hostname = socket.gethostname()
        socket.gethostbyname(hostname)
        return hostname
    except:
        pass
    
    return 'localhost'

def infer_with_ollama(prompt: str, LLMmodel: str = "llama3.2:latest", temperature: float = 0.1, max_tokens: int = 5000) -> str:
    try:
        # Use the explicitly set OLLAMA_HOST from environment
        ollama_host = os.environ.get('OLLAMA_HOST', 'localhost:11434')
        
        # Split host and port
        host, port = ollama_host.split(':')
        
        # Use ollama client with the specific host and port
        client = ollama.Client(host=f'http://{host}:{port}')
        
        # Generate response
        response = client.generate(prompt=prompt, model=LLMmodel)
        
        # Extract the response text directly
        # The response seems to have a 'response' attribute
        response_text = response.get('response', '')
        
        # If the response is empty or the default welcome message, 
        # you might want to handle it differently
        if not response_text.strip():
            logger.warning("Empty response received from Ollama")
            return "Error: No response from model"
        
        return response_text
    
    except Exception as e:
        logger.error(f"Ollama inference error: {e}")
        return f"Error: {e}"


class ScoringMethod(Enum):
    BM25 = "bm25"
    TFIDF = "tfidf"
    COMBINED = "combined"

class SafeSearch(Enum):
    STRICT = 2
    MODERATE = 1
    NONE = 0

class QueryType(Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"

SAFE_SEARCH_OPTIONS = [
    ("Strict (2)", SafeSearch.STRICT.value),
    ("Moderate (1)", SafeSearch.MODERATE.value),
    ("None (0)", SafeSearch.NONE.value)
]

def extract_urls(text: str) -> List[str]:
    """Extract URLs from text using an improved regex pattern."""
    # Updated regex pattern to better handle complex URLs with query parameters and paths
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^)\s]*)?'
    urls = re.findall(url_pattern, text)

    # Clean and validate found URLs
    valid_urls = []
    for url in urls:
        # Remove trailing punctuation or artifacts that might have been captured
        url = url.rstrip('.,;:)')
        if is_valid_url(url):
            valid_urls.append(url)

    return valid_urls

def is_valid_url(url: str) -> bool:
    """Check if the provided string is a valid URL with enhanced validation."""
    try:
        result = urlparse(url)
        # Check for both scheme and netloc (domain)
        has_valid_scheme = result.scheme in ('http', 'https')
        has_valid_domain = bool(result.netloc)
        # Additional validation to ensure complete URL structure
        is_complete = all([has_valid_scheme, has_valid_domain])
        return is_complete
    except Exception as e:
        logger.error(f'URL validation error: {e}')
        return False

async def determine_query_type(query: str, chat_history: List[List[str]], temperature: float = 0.1, LLMmodel: str = None, max_tokens: int = 10) -> QueryType:
    """
    Determine whether a query should be answered from knowledge base or require web search.
    Now with improved context handling and response validation.
    """
    logger.info(f'Determining query type for: {query}')
    try:
        # Assign default model early
        if LLMmodel is None:
            LLMmodel = "ollama"

        # Normalize model name to lowercase
        LLMmodel = LLMmodel.lower()

        # Format chat history into a more natural conversation format
        formatted_history = []
        for i, (user_msg, assistant_msg) in enumerate(chat_history[-5:], 1):  # Last 5 turns
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"User: {user_msg}")
            if assistant_msg:
                formatted_history.append(f"Assistant: {assistant_msg}")
        chat_context = "\n".join(formatted_history)
        system_prompt = """You are Sentinel, an intelligent AI agent tasked with determining whether a user query requires a web search or can be answered using your existing knowledge base. Your knowledge cutoff date is April 2024, and the current date is November 2024.
Rules for Classification:
IMPORTANT: You must ONLY respond with either "knowledge_base" or "web_search" - no other text or explanation is allowed.
Classify as "web_search" if the query:
- Explicitly asks for current/latest/recent information
- References events or data after April 2024
- Requires real-time information (prices, weather, news)
- Uses words like "current", "latest", "now", "today"
- Asks about ongoing events or situations
- Needs verification of recent claims
- Is a follow-up question about current events
- Previous context involves recent/ongoing events
Classify as "knowledge_base" if the query:
- Is about historical events or facts before April 2024
- Involves general knowledge, concepts, or theories
- Is casual conversation or greeting
- Asks for explanations of established topics
- Requires logical reasoning or analysis
- Is about personal opinions or hypotheticals
- Is a follow-up to a knowledge-base discussion
- Previous context is about historical or conceptual topics"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Previous conversation:\n{chat_context}\n\nQuery to classify: {query}\n\nRespond ONLY with 'knowledge_base' or 'web_search':"}
        ]
        
        if LLMmodel == "groq":
            response = groq_manager.get_next_client().chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                temperature=temperature,
                max_tokens=10,
                stream=False
            )
            result = response.choices[0].message.content.strip().lower()
        elif LLMmodel == "ollama":
            # Directly use the returned string and clean it
            result = infer_with_ollama("\n".join([m["content"] for m in messages]), LLMmodel="llama3.2:latest", temperature=temperature, max_tokens=max_tokens)
            result = result.strip().lower()
        else:
            raise ValueError(f"Invalid model: {LLMmodel}")

        # Validate the response
        if result not in ["knowledge_base", "web_search"]:
            logger.warning(f'Invalid response from LLM: {result}. Defaulting to knowledge_base')
            return QueryType.KNOWLEDGE_BASE
        
        logger.info(f'Query type determined as: {result}')
        return QueryType.WEB_SEARCH if result == "web_search" else QueryType.KNOWLEDGE_BASE
    except Exception as e:
        logger.error(f'Error determining query type: {e}. Defaulting to knowledge_base')
        return QueryType.KNOWLEDGE_BASE

async def process_knowledge_base_query(query: str, chat_history: List[List[str]], temperature: float = 0.7, LLMmodel: str = None, max_tokens: int = 5000) -> str:
    """Handle queries that can be answered from the knowledge base, with context."""
    logger.info(f'Processing knowledge base query: {query}')
    try:
        # Assign default model early
        if LLMmodel is None:
            LLMmodel = "ollama"

        # Normalize model name to lowercase
        LLMmodel = LLMmodel.lower()

        # Format recent conversation history
        formatted_history = []
        for i, (user_msg, assistant_msg) in enumerate(chat_history[-5:], 1):
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"User: {user_msg}")
            if assistant_msg:
                formatted_history.append(f"Assistant: {assistant_msg}")
        chat_context = "\n".join(formatted_history)
        system_prompt = """You are Sentinel, a highly knowledgeable AI assistant with expertise through April 2024. You provide accurate, informative responses based on your knowledge base while maintaining conversation context.
Guidelines:
1. Use the conversation history to provide contextually relevant responses
2. Reference previous turns when appropriate
3. Maintain consistency with previous responses
4. Use markdown formatting for better readability
5. Be clear about historical facts vs. analysis
6. Note if information might be outdated
7. Stay within knowledge cutoff date of April 2024
8. Be direct and conversational
9. Acknowledge and build upon previous context when relevant"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Previous conversation:\n{chat_context}\n\nCurrent query: {query}\n\nProvide a comprehensive response based on your knowledge base and the conversation context."}
        ]
        
        if LLMmodel == "groq":
            response = groq_manager.get_next_client().chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                temperature=temperature,
                stream=False
            )
            result = response.choices[0].message.content
        elif LLMmodel == "ollama":
            result = infer_with_ollama("\n".join([m["content"] for m in messages]), LLMmodel="llama3.2:latest", temperature=temperature, max_tokens=max_tokens)
        else:
            raise ValueError(f"Invalid model: {LLMmodel}")
        
        return result
    except Exception as e:
        logger.error(f'Error processing knowledge base query: {e}')
        return f"I apologize, but I encountered an error while processing your query: {str(e)}"


async def rephrase_query(chat_history, query, temperature=0.2, LLMmodel: str = None, max_tokens: int = 200) -> str:
    """Rephrase the query based on chat history and context."""
    logger.info(f'Rephrasing query: {query}')
    try:
        # Assign default LLM model early
        if LLMmodel is None:
            LLMmodel = "ollama"

        # Normalize LLM model name to lowercase
        LLMmodel = LLMmodel.lower()

        # Format recent conversation history (last 3 turns for context)
        formatted_history = []
        for i, (user_msg, assistant_msg) in enumerate(chat_history[-3:], 1):
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"User: {user_msg}")
            if assistant_msg:
                formatted_history.append(f"Assistant: {assistant_msg}")

        chat_context = "\n".join(formatted_history)
        current_year = datetime.now().year

        system_prompt = """You are a highly intelligent query rephrasing assistant. Your task is to analyze the conversation history and current query to generate a complete, contextual search query.

Key Rules:
1. For follow-up questions or queries referencing previous conversation:
   - Extract the main topic/subject from previous messages
   - Combine previous context with the current query
   - Example:
     Previous: "What is the structure of German banking industry?"
     Current: "can you do more latest web search on my previous query"
     Should become: "Latest structure and developments in German banking industry after: 2024"

2. Entity Handling:
   - Identify and preserve main entities from context
   - Enclose ONLY entity names in double quotes
   - Example: "Deutsche Bank" profits, not "Deutsche Bank profits"

3. Date and Time Context:
   - For queries about current/latest information:
     * Keep time-related words (latest, current, recent, now)
     * ALWAYS append "after: YYYY" (current year)
   - For specific time periods:
     * Preserve the original time reference
     * Add appropriate "after: YYYY" based on context
   - For queries without time reference:
     * Add "after: YYYY" if about current state/status

4. Query Formatting:
   - Capitalize first letter
   - No period at end
   - Include all relevant context
   - Maintain clear and searchable structure

Remember: Your goal is to create a complete, self-contained query that includes all necessary context from the conversation history."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Current year: {current_year}

Recent conversation history:
{chat_context}

Current query: {query}

Please rephrase this query into a complete, contextual search query following the rules above. The rephrased query should be clear and complete even without the conversation context."""}
        ]

        if LLMmodel == "groq":
            response = groq_manager.get_next_client().chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                temperature=temperature,
                max_tokens=200,
                stream=False
            )
            rephrased_query = response.choices[0].message.content.strip()
        elif LLMmodel == "ollama":
            response = infer_with_ollama("\n".join([m["content"] for m in messages]), LLMmodel="llama3.2:latest", temperature=temperature, max_tokens=max_tokens)
            rephrased_query = response.strip()
        else:
            raise ValueError(f"Invalid model: {LLMmodel}")

        logger.info(f'Query rephrased to: {rephrased_query}')
        return rephrased_query

    except Exception as e:
        logger.error(f'Error rephrasing query: {e}')
        # If rephrasing fails, construct a basic contextual query
        try:
            last_query = chat_history[-1][0] if chat_history else ""
            if any(word in query.lower() for word in ['latest', 'recent', 'current', 'now', 'update']):
                return f"{last_query} latest updates after: {datetime.now().year}"
            return query
        except:
            return query  # Return original query as last resort


class ParallelScraper:
    def __init__(self, max_workers: int = 5):
        logger.info(f"Initializing ParallelScraper with {max_workers} workers")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        logger.info("Creating aiohttp session")
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            logger.info("Closing aiohttp session")
            await self.session.close()

    def parse_article(self, article: Article) -> Dict[str, Any]:
        """Parse a newspaper Article object in a separate thread"""
        try:
            logger.info("Parsing article")
            article.parse()
            return {
                "content": article.text,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None
            }
        except Exception as e:
            logger.error(f'Error parsing article: {e}')
            return None

    async def download_and_parse_html(self, url: str, max_chars: int) -> Dict[str, Any]:
        """Download and parse HTML content asynchronously"""
        logger.info(f'Processing HTML URL: {url}')
        try:
            article = Article(url)
            await asyncio.get_event_loop().run_in_executor(self.executor, article.download)
            result = await asyncio.get_event_loop().run_in_executor(self.executor, self.parse_article, article)

            if result:
                result["content"] = result["content"][:max_chars]
                logger.info(f'Successfully processed HTML from {url}')
            return result
        except Exception as e:
            logger.error(f'Error processing HTML from {url}: {e}')
            return None

    async def download_and_parse_pdf(self, url: str, max_chars: int) -> Dict[str, Any]:
        """Download and parse PDF content asynchronously"""
        logger.info(f'Processing PDF URL: {url}')
        try:
            if not self.session:
                raise RuntimeError("Session not initialized")

            async with self.session.get(url) as response:
                pdf_bytes = await response.read()

            def process_pdf():
                logger.info("Processing PDF content")
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text[:max_chars]

            text = await asyncio.get_event_loop().run_in_executor(self.executor, process_pdf)
            logger.info(f'Successfully processed PDF from {url}')
            return {"content": text, "publish_date": None}
        except Exception as e:
            logger.error(f'Error processing PDF from {url}: {e}')
            return None

    async def scrape_url(self, url: str, max_chars: int) -> Dict[str, Any]:
        """Scrape content from a URL, handling both HTML and PDF formats"""
        logger.info(f'Starting to scrape URL: {url}')
        if url.endswith('.pdf'):
            return await self.download_and_parse_pdf(url, max_chars)
        else:
            return await self.download_and_parse_html(url, max_chars)

    async def scrape_urls(self, urls: list, max_chars: int) -> list:
        """Scrape multiple URLs in parallel"""
        logger.info(f'Starting parallel scraping of {len(urls)} URLs')
        tasks = [self.scrape_url(url, max_chars) for url in urls]
        return await asyncio.gather(*tasks)

async def scrape_urls_parallel(results: list, max_chars: int) -> list:
    """Scrape multiple URLs in parallel using the ParallelScraper"""
    logger.info(f'Initializing parallel scraping for {len(results)} results')
    async with ParallelScraper() as scraper:
        urls = [result["url"] for result in results]
        scraped_data = await scraper.scrape_urls(urls, max_chars)

        # Combine results with scraped data
        valid_results = []
        for result, article in zip(results, scraped_data):
            if article is not None:
                valid_results.append((result, article))

        logger.info(f'Successfully scraped {len(valid_results)} valid results')
        return valid_results

async def get_available_engines(session, base_url, headers):
    """Fetch available search engines from SearxNG instance."""
    logger.info("Fetching available search engines")
    try:
        params = {
            "q": "test",
            "format": "json",
            "engines": "all"
        }
        async with session.get(f"{base_url}/search", headers=headers, params=params) as response:
            data = await response.json()
            available_engines = set()
            if "search" in data:
                for engine_data in data["search"]:
                    if isinstance(engine_data, dict) and "engine" in engine_data:
                        available_engines.add(engine_data["engine"])

            if not available_engines:
                async with session.get(f"{base_url}/engines", headers=headers) as response:
                    engines_data = await response.json()
                    available_engines = set(engine["name"] for engine in engines_data if engine.get("enabled", True))

            logger.info(f'Found {len(available_engines)} available engines')
            return list(available_engines)
    except Exception as e:
        logger.error(f'Error fetching search engines: {e}')
        return ["google", "bing", "duckduckgo", "brave", "wikipedia"]

def normalize_scores(scores):
    """Normalize scores to [0, 1] range using min-max normalization"""
    if not isinstance(scores, np.ndarray):
        scores = np.array(scores)

    if len(scores) == 0:
        return []

    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score - min_score > 0:
        normalized = (scores - min_score) / (max_score - min_score)
    else:
        normalized = np.ones_like(scores)

    return normalized.tolist()

async def calculate_bm25(query, documents):
    """Calculate BM25 scores for documents."""
    logger.info("Calculating BM25 scores")
    try:
        if not documents:
            return []

        bm25 = BM25Okapi([doc.split() for doc in documents])
        scores = bm25.get_scores(query.split())
        normalized_scores = normalize_scores(scores)
        logger.info("BM25 scores calculated successfully")
        return normalized_scores

    except Exception as e:
        logger.error(f'Error calculating BM25 scores: {e}')
        return [0] * len(documents)

async def calculate_tfidf(query, documents, measure="cosine"):
    """Calculate TF-IDF based similarity scores."""
    logger.info("Calculating TF-IDF scores")
    try:
        if not documents:
            return []

        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Encoding query and documents")
        query_embedding = model.encode(query)
        document_embeddings = model.encode(documents)

        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1)[:, np.newaxis]

        if measure == "cosine":
            scores = np.dot(document_embeddings, query_embedding)
            normalized_scores = normalize_scores(scores)
            logger.info("TF-IDF scores calculated successfully")
            return normalized_scores
        else:
            raise ValueError("Unsupported similarity measure.")

    except Exception as e:
        logger.error(f'Error calculating TF-IDF scores: {e}')
        return [0] * len(documents)

def combine_scores(bm25_score, tfidf_score, weights=(0.5, 0.5)):
    """Combine scores using weighted average."""
    return weights[0] * bm25_score + weights[1] * tfidf_score

async def get_document_scores(query, documents, scoring_method: ScoringMethod):
    """Calculate document scores based on the chosen scoring method."""
    if not documents:
        return []

    if scoring_method == ScoringMethod.BM25:
        scores = await calculate_bm25(query, documents)
        return [(score, 0) for score in scores]
    elif scoring_method == ScoringMethod.TFIDF:
        scores = await calculate_tfidf(query, documents)
        return [(0, score) for score in scores]
    else:  # COMBINED
        bm25_scores = await calculate_bm25(query, documents)
        tfidf_scores = await calculate_tfidf(query, documents)
        return list(zip(bm25_scores, tfidf_scores))

def get_total_score(scores, scoring_method: ScoringMethod):
    """Calculate total score based on the scoring method."""
    bm25_score, tfidf_score = scores
    if scoring_method == ScoringMethod.BM25:
        return bm25_score
    elif scoring_method == ScoringMethod.TFIDF:
        return tfidf_score
    else:  # COMBINED
        return combine_scores(bm25_score, tfidf_score)

async def generate_summary(query: str, articles: List[Dict[str, Any]], temperature: float = 0.1, LLMmodel: str = None, max_tokens: int = 5000) -> str:
    """Generate a summary of the articles using Groq's LLama 3.1 model."""
    logger.info(f'Generating summary for query: {query}')
    try:
        # Assign default LLM model early
        if LLMmodel is None:
            LLMmodel = "ollama"

        # Normalize LLM model name to lowercase
        LLMmodel = LLMmodel.lower()

        json_input = json.dumps(articles, indent=2)

        system_prompt = """You are Sentinel, a world-class AI model who is expert at searching the web and answering user's queries. You are also an expert at summarizing web pages or documents and searching for content in them."""

        user_prompt = f"""
Please provide a comprehensive summary based on the following JSON input:
{json_input}

Original Query: {query}

Instructions:
1. Analyze the query and the provided documents.
2. Write a detailed, long, and complete research document that is informative and relevant to the user's query based on provided context.
3. Use this context to answer the user's query in the best way possible. Use an unbiased and journalistic tone.
4. Use an unbiased and professional tone in your response.
5. Do not repeat text verbatim from the input.
6. Provide the answer in the response itself.
7. Use markdown to format your response.
8. Use bullet points to list information where appropriate.
9. Cite the answer using [number] notation along with the appropriate source URL embedded in the notation.
10. Place these citations at the end of the relevant sentences.
11. You can cite the same sentence multiple times if it's relevant.
12. Make sure the answer is not short and is informative.
13. Your response should be detailed, informative, accurate, and directly relevant to the user's query."""

        logger.info("Sending request to LLM API")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if LLMmodel == "groq":
            response = groq_manager.get_next_client().chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                temperature=temperature,
                max_tokens=5000,
                stream=False
            )
            result = response.choices[0].message.content.strip()
        elif LLMmodel == "ollama":
            result = infer_with_ollama("\n".join([m["content"] for m in messages]), LLMmodel="llama3.2:latest", temperature=temperature, max_tokens=max_tokens )
            result = result.strip()
        else:
            raise ValueError(f"Invalid model: {LLMmodel}")

        logger.info(f'Summary generated successfully')
        return result

    except Exception as e:
        logger.error(f'Error generating summary: {e}')
        return f"Error generating summary: {str(e)}"

class ChatBot:
    def __init__(self):
        logger.info("Initializing ChatBot")
        self.scoring_method = ScoringMethod.COMBINED
        self.num_results = 10
        self.max_chars = 10000
        self.score_threshold = 0.8
        self.temperature = 0.1
        self.conversation_history = []
        self.base_url = "http://localhost:8888/"
        self.headers = {
            "X-Searx-API-Key": "f9f07f93b37b8483aadb5ba717f556f3a4ac507b281b4ca01e6c6288aa3e3ae5"
        }
        self.default_engines = ["google", "bing", "duckduckgo", "brave"]
        self.available_languages = {
            "all": "All Languages",
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean"
        }
        logger.info("ChatBot initialized successfully")

    def format_chat_history(self, history: List[List[str]]) -> str:
        """Format chat history into a readable string with clear turn markers."""
        formatted_history = []
        for i, (user_msg, assistant_msg) in enumerate(history, 1):
            formatted_history.append(f"Turn {i}:")
            formatted_history.append(f"User: {user_msg}")
            if assistant_msg:
                formatted_history.append(f"Assistant: {assistant_msg}")
        return "\n".join(formatted_history)

    async def generate_research_queries(self, query: str, temperature: float = 0.3, LLMmodel: str = None) -> List[str]:
        """Generate multiple research queries based on the initial query."""
        logger.info(f'Generating research queries for: {query}')
        try:
            # Assign default LLM model early
            if LLMmodel is None:
                LLMmodel = "ollama"

            # Normalize LLM model name to lowercase
            LLMmodel = LLMmodel.lower()

            system_prompt = """Advanced Research Query Generation System Prompt
    Core Objective
    You are a world-class research strategist and query optimization expert, tasked with generating precise, multi-dimensional research queries that unlock comprehensive insights on complex topics.
    Research Query Generation Principles
    Dimensionality of Inquiry
    Develop queries that explore the topic through multiple critical lenses:

    Historical Context: Origin, evolution, and foundational understanding
    Current Landscape: Present state, key players, and contemporary significance
    Analytical Depth: Underlying mechanisms, systemic impacts, and nuanced insights
    Future Projection: Emerging trends, potential developments, and strategic implications
    Interdisciplinary Connections: Intersections with related fields and broader ecosystem

    Query Sophistication Criteria

    Maximize semantic precision
    Target verifiable, substantive information sources
    Balance specificity with investigative breadth
    Anticipate potential research dead-ends
    Optimize for high-signal, low-noise information retrieval

    Tactical Query Construction

    Use advanced Boolean and semantic modifiers
    Incorporate domain-specific terminology
    Anticipate multiple knowledge domains
    Craft queries that reveal hidden relationships and insights

    Intellectual Rigor Standards

    Prioritize queries that reveal:

    Causal relationships
    Quantitative evidence
    Expert consensus
    Emerging paradigms
    Critical debates or controversies


    Output Specifications

    Generate EXACTLY 5 queries
    Queries must be:

    Grammatically precise
    Conceptually distinct
    Actionable and research-oriented
    Maximum 15 words each


    Format: Comma-separated list of queries

    Prohibited Approaches

    Avoid vague, overly broad queries
    Reject queries likely to produce superficial or uninformative results
    Eliminate redundant or semantically overlapping queries

    Intellectual Mandate
    Transform the initial query into a strategic research investigation that reveals profound, multi-dimensional understanding."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate research queries for the topic: {query}\n\nRespond with 5 comma-separated queries:"}
            ]

            if LLMmodel == "groq":
                response = groq_manager.get_next_client().chat.completions.create(
                    messages=messages,
                    model="llama-3.1-70b-versatile",
                    temperature=temperature,
                    max_tokens=5000,
                    stream=False
                )
                queries = response.choices[0].message.content.strip().split(',')
            elif LLMmodel == "ollama":
                response = infer_with_ollama("\n".join([m["content"] for m in messages]), LLMmodel="llama3.2:latest")
                queries = response.strip().split(',')
            else:
                raise ValueError(f"Invalid model: {LLMmodel}")

            queries = [q.strip() for q in queries][:5]  # Ensure exactly 5 queries

            logger.info(f'Generated research queries: {queries}')
            return queries

        except Exception as e:
            logger.error(f'Error generating research queries: {e}')
            # Fallback: Generate generic research queries
            return [
                query,
                f"{query} overview and background",
                f"{query} recent developments",
                f"{query} financial performance",
                f"{query} market impact"
            ]

    async def get_search_results(self,
                               query: str,
                               history: List[List[str]],
                               num_results: int,
                               max_chars: int,
                               score_threshold: float,
                               temperature: float,
                               scoring_method: str,
                               selected_engines: List[str],
                               safe_search: str,
                               LLMmodel: str,
                               language: str) -> str:
        logger.info(f'Processing search request for query: {query}')
        try:
            # First, rephrase the query using chat history
            rephrased_query = await rephrase_query(history, query, temperature=0.2, LLMmodel=LLMmodel)
            logger.info(f'Original query: {query}')
            logger.info(f'Rephrased query: {rephrased_query}')

            scoring_method_map = {
                "BM25": ScoringMethod.BM25,
                "TF-IDF": ScoringMethod.TFIDF,
                "Combined": ScoringMethod.COMBINED
            }
            self.scoring_method = scoring_method_map[scoring_method]

            safe_search_map = dict(SAFE_SEARCH_OPTIONS)
            safe_search_value = safe_search_map.get(safe_search, SafeSearch.MODERATE.value)

            logger.info(f'Search parameters - Engines: {selected_engines}, Results: {num_results}, Method: {scoring_method}')

            # Use the rephrased query for the search
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": rephrased_query,  # Use rephrased query here
                    "format": "json",
                    "engines": ",".join(selected_engines),
                    "limit": num_results,
                    "safesearch": safe_search_value,
                }

                if language != "all":
                    params["language"] = language

                logger.info("Sending search request to SearxNG")
                try:
                    async with session.get(f"{self.base_url}/search", headers=self.headers, params=params) as response:
                        data = await response.json()
                except Exception as e:
                    logger.error(f'SearxNG connection error: {e}')
                    return f"Error: Could not connect to search service. Please check if SearxNG is running at {self.base_url}. Error: {str(e)}"

                if "results" not in data or not data["results"]:
                    logger.info("No search results found")
                    return "No results found."

                results = data["results"][:num_results]
                logger.info(f'Processing {len(results)} search results')
                valid_results = await scrape_urls_parallel(results, max_chars)

                if not valid_results:
                    logger.info("No valid articles found after scraping")
                    return "No valid articles found after scraping."

                results, scraped_data = zip(*valid_results)
                contents = [article["content"] for article in scraped_data]

                logger.info("Calculating document scores")
                scores = await get_document_scores(query, contents, self.scoring_method)

                scored_articles = []
                for i, (score_tuple, article) in enumerate(zip(scores, scraped_data)):
                    total_score = get_total_score(score_tuple, self.scoring_method)
                    if total_score >= self.score_threshold:
                        scored_articles.append({
                            "url": results[i]["url"],
                            "title": results[i]["title"],
                            "content": article["content"],
                            "publish_date": article["publish_date"],
                            "score": round(total_score, 4),
                            "bm25_score": round(score_tuple[0], 4),
                            "tfidf_score": round(score_tuple[1], 4),
                            "engine": results[i].get("engine", "unknown")
                        })

                scored_articles.sort(key=lambda x: x["score"], reverse=True)
                unique_articles = []
                seen_content = set()

                for article in scored_articles:
                    if article["content"] not in seen_content:
                        seen_content.add(article["content"])
                        unique_articles.append(article)

                # Generate summary using Groq API
                summary = await generate_summary(query, unique_articles, temperature, LLMmodel=LLMmodel)

                # Update the response format to use scoring_method instead of scoring_method_str
                response = f"**Search Parameters:**\n"
                response += f"- Results: {num_results}\n"
                response += f"- Max Characters: {max_chars}\n"
                response += f"- Score Threshold: {score_threshold}\n"
                response += f"- Temperature: {temperature}\n"
                response += f"- Scoring Method: {scoring_method}\n"  # Updated this line
                response += f"- Search Engines: {', '.join(selected_engines)}\n"
                response += f"- Safe Search: Level {safe_search_value}\n"
                response += f"- Language: {self.available_languages.get(language, language)}\n\n"

                response += "**Results Summary:**\n"
                response += summary + "\n\n"

                response += "**Sources:**\n"
                for i, article in enumerate(unique_articles, 1):
                    response += f"{i}. [{article['title']}]({article['url']}) (Score: {article['score']})\n"

                return response

        except Exception as e:
            logger.error(f'Error in search_and_summarize: {e}')
            return f"Error occurred: {str(e)}"

    async def scrape_specific_urls(self, urls: List[str], max_chars: int) -> List[Dict[str, Any]]:
        """Scrape specific URLs provided by the user."""
        logger.info(f'Scraping specific URLs: {urls}')
        try:
            # Create dummy results structure expected by scrape_urls_parallel
            results = [{"url": url} for url in urls]
            valid_results = await scrape_urls_parallel(results, max_chars)

            if not valid_results:
                logger.info("No valid content found from provided URLs")
                return []

            processed_articles = []
            for result, article in valid_results:
                if article:
                    processed_articles.append({
                        "url": result["url"],
                        "title": urlparse(result["url"]).netloc,  # Use domain as title if not available
                        "content": article["content"],
                        "publish_date": article["publish_date"],
                        "score": 1.0,  # Direct URL scraping, so score is 1.0
                        "engine": "direct_url"
                    })

            return processed_articles

        except Exception as e:
            logger.error(f'Error scraping specific URLs: {e}')
            return []


    async def conduct_research(self,
                               query: str,
                               history: List[List[str]],
                               num_results: int = 15,
                               max_chars: int = 15000,
                               score_threshold: float = 0.7,
                               temperature: float = 0.3,
                               scoring_method: str = "Combined",
                               selected_engines: List[str] = None,
                               safe_search: str = "Moderate (1)",
                               language: str = "all",
                               LLMmodel: str = "ollama"
                               ) -> str:
        """Conduct comprehensive research by generating and searching multiple queries."""
        logger.info(f'Starting research mode for query: {query}')

        if selected_engines is None:
            selected_engines = ["google", "bing", "duckduckgo", "brave"]

        # Generate research queries
        try:
            research_queries = await self.generate_research_queries(query, temperature, LLMmodel=LLMmodel)

            # Validate research queries
            if not research_queries:
                logger.warning("No research queries generated")
                return f"Unable to generate research queries for: {query}"
        except Exception as e:
            logger.error(f'Error generating research queries: {e}')
            return f"Error generating research queries: {str(e)}"

        # Perform parallel searches with a reasonable limit
        max_concurrent_searches = 5
        search_results = []

        for i in range(0, len(research_queries), max_concurrent_searches):
            batch_queries = research_queries[i:i+max_concurrent_searches]

            # Create tasks for this batch
            search_tasks = [
                self.get_search_results(
                    query=research_query,
                    history=history,
                    num_results=num_results,
                    max_chars=max_chars,
                    score_threshold=score_threshold,
                    temperature=temperature,
                    scoring_method=scoring_method,
                    selected_engines=selected_engines,
                    safe_search=safe_search,
                    language=language,
                    LLMmodel=LLMmodel
                ) for research_query in batch_queries
            ]

            # Wait for this batch to complete
            batch_results = await asyncio.gather(*search_tasks)
            search_results.extend(batch_results)

        # Combine research results into a comprehensive research paper
        research_paper = f"# Comprehensive Research: {query}\n\n"
        research_paper += "## Research Methodology\n"
        research_paper += f"**Primary Query:** {query}\n"
        research_paper += f"**Research Queries:** {', '.join(research_queries)}\n"
        research_paper += f"**Search Engines:** {', '.join(selected_engines)}\n\n"

        for i, (query_result, search_query) in enumerate(zip(search_results, research_queries), 1):
            research_paper += f"## Research Focus {i}: {search_query}\n\n"
            research_paper += query_result + "\n\n"

        # Final synthesis and conclusion
        research_paper += "## Synthesis and Key Insights\n"

          # Use Groq to synthesize final insights
        try:
              
              messages=[
                  {"role": "system", "content": "You are an expert researcher synthesizing insights from multiple sources."},
                  {"role": "user", "content": f"Synthesize key insights and provide a comprehensive conclusion for this research on '{query}':\n{research_paper}"}
              ]
              
              if LLMmodel == "groq":
                  synthesis_response = groq_manager.execute_groq_api_call(
                    groq_manager.get_next_client().chat.completions.create,
                      messages=messages,
                      model="llama-3.1-70b-versatile",
                      temperature=temperature,
                      max_tokens=1000,
                      stream=False
                  )

                  synthesis = synthesis_response.choices[0].message.content.strip()
              elif LLMmodel == "ollama":
                  synthesis = infer_with_ollama("\n".join([m["content"] for m in messages]), LLMmodel=LLMmodel, temperature=temperature, max_tokens=1000 )
                  synthesis = synthesis.strip()
              else:
                  raise ValueError(f"Invalid model: {LLMmodel}")
              
              research_paper += synthesis

        except Exception as e:
            logger.error(f'Error generating research synthesis: {e}')
            research_paper += "\n**Note:** Synthesis generation encountered an error."

        return research_paper

    async def get_response(self,
                           query: str,
                           history: List[List[str]],
                           num_results: int,
                           max_chars: int,
                           score_threshold: float,
                           temperature: float,
                           scoring_method: str,
                           selected_engines: List[str],
                           safe_search: str,
                           language: str,
                           LLMmodel: str,  # Pass LLMmodel here
                           force_web_search: bool = False,
                           research_mode: bool = False,
                           ) -> str:
        """Enhanced get_response method with research mode capability."""
        logger.info(f'Processing query: {query}, Research Mode: {research_mode}')
        logger.info(f"Selected LLM model: {LLMmodel}")  # Debug statement here

        try:
            if research_mode:
                return await self.conduct_research(
                    query=query,
                    history=history,
                    num_results=num_results,
                    max_chars=max_chars,
                    score_threshold=score_threshold,
                    temperature=temperature,
                    scoring_method=scoring_method,
                    selected_engines=selected_engines,
                    safe_search=safe_search,
                    language=language,
                    LLMmodel=LLMmodel # Pass LLM model to research
                )

            # Extract URLs from the query
            urls = extract_urls(query)

            # If valid URLs are found in the query, directly scrape them
            if urls:
                logger.info(f'Found URLs in query: {urls}')
                articles = await self.scrape_specific_urls(urls, max_chars)

                if not articles:
                    return "I couldn't extract valid content from the provided URLs. Please check if the URLs are accessible."

                # Generate summary using only the scraped content
                summary = await generate_summary(query, articles, temperature, LLMmodel=LLMmodel)  # pass LLM here

                # Format response
                response = "**Direct URL Scraping Results:**\n\n"
                response += summary + "\n\n"
                response += "**Scraped URLs:**\n"
                for i, article in enumerate(articles, 1):
                    response += f"{i}. [{urlparse(article['url']).netloc}]({article['url']})\n"

                return response

            # If no URLs found, proceed with regular query processing
            formatted_history = self.format_chat_history(history)
            force_web_search = force_web_search == "Web Search Only"

            if force_web_search:
                query_type = QueryType.WEB_SEARCH
            else:
                query_type = await determine_query_type(query, history, temperature, LLMmodel=LLMmodel) #pass here

            if query_type == QueryType.KNOWLEDGE_BASE and not force_web_search:
                response = await process_knowledge_base_query(
                    query=query,
                    chat_history=history,
                    temperature=temperature,
                    LLMmodel=LLMmodel
                )
            else:
                # IMPORTANT CHANGE HERE: pass LLMmodel
                response = await self.get_search_results(
                    query=query,
                    history=history,
                    num_results=num_results,
                    max_chars=max_chars,
                    score_threshold=score_threshold,
                    temperature=temperature,
                    scoring_method=scoring_method,
                    selected_engines=selected_engines,
                    safe_search=safe_search,
                    language=language,
                    LLMmodel=LLMmodel #pass LLM here
                )

            return response

        except Exception as e:
            logger.error(f'Error in get_response: {e}')
            return f"I apologize, but I encountered an error: {str(e)}"

    def chat(self,
             message: str,
             history: List[List[str]],
             num_results: int,
             max_chars: int,
             score_threshold: float,
             temperature: float,
             scoring_method: str,
             engines: List[str],
             safe_search: str,
             language: str,
             LLMmodel: str,
             force_web_search: bool,
             research_mode: bool = False) -> str:
        """Process chat messages with context and return responses."""
        # Extract language code and process response
        language_code = language.split(" - ")[0]

        # Update conversation history from the Gradio history
        self.conversation_history = history

        # Add extensive logging
        logger.info(f"Chat method called with LLM Model: {LLMmodel}")
        
        response = asyncio.run(self.get_response(
            message,
            self.conversation_history,
            num_results,
            max_chars,
            score_threshold,
            temperature,
            scoring_method,
            engines,
            safe_search,
            language_code,
            LLMmodel,
            force_web_search,
            research_mode # Add research_mode parameter
        ))
        return response
 
def create_gradio_interface() -> gr.Interface:
    chatbot = ChatBot()

    # Define language options
    language_choices = [
        "all", "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"
    ]

    # Create mapping for language display names
    language_display = {
        "all": "All Languages",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean"
    }

    # Create the interface with all parameters
    iface = gr.ChatInterface(
        chatbot.chat,
        title="Web Scraper for News with Sentinel AI",
        description="Ask Sentinel any question. It will search the web for recent information or use its knowledge base as appropriate.",
        theme=gr.Theme.from_hub("allenai/gradio-theme"),
        additional_inputs=[
            gr.Slider(
                minimum=5,
                maximum=30,
                value=10,
                step=1,
                label="Number of Results"
            ),
            gr.Slider(
                minimum=1000,
                maximum=50000,
                value=10000,
                step=1000,
                label="Max Characters per Article"
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.8,
                step=0.05,
                label="Score Threshold"
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.05,
                label="Temperature"
            ),
            gr.Radio(
                choices=["BM25", "TF-IDF", "Combined"],
                value="Combined",
                label="Scoring Method"
            ),
            gr.CheckboxGroup(
                choices=["google", "bing", "duckduckgo", "brave", "wikipedia"],
                value=["google", "bing", "duckduckgo"],
                label="Search Engines"
            ),
            gr.Radio(
                choices=[option[0] for option in SAFE_SEARCH_OPTIONS],
                value="Moderate (1)",
                label="Safe Search Level",
                info="Controls the filtering level of search results (0=None, 1=Moderate, 2=Strict)"
            ),
            gr.Radio(
                choices=[f"{code} - {language_display[code]}" for code in language_choices],
                value="all - All Languages",
                label="Language",
                info="Select the preferred language for search results"
            ),
            gr.Radio(
                choices=["groq", "ollama"],
                label="Inference Model",
                value="groq",
                info="Select the model for inference (Groq or Ollama)"
            ),
            gr.Radio(
                choices=["Auto (Knowledge Base + Web)", "Web Search Only"],
                value="Auto (Knowledge Base + Web)",
                label="Search Mode",
                info="Choose whether to use both knowledge base and web search, or force web search only"
            ),
            gr.Checkbox(
                label="Research Mode",
                value=False,
                info="When enabled, AI will conduct comprehensive multi-query research"
            )

        ],
        additional_inputs_accordion=gr.Accordion("⚙️ Advanced Parameters", open=True),
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear",
        chatbot=gr.Chatbot(
            show_copy_button=True,
            likeable=True,
            layout="bubble",
            height=500,
        )
    )

    return iface

def create_parameter_description():
    return """
    ### Parameter Descriptions

    - **Number of Results**: Number of search results to fetch
    - **Max Characters**: Maximum characters to analyze per article
    - **Score Threshold**: Minimum relevance score (0-1) for including articles
    - **Temperature**: Controls creativity in summary generation (0=focused, 1=creative)
    - **Scoring Method**: Algorithm for ranking article relevance
        - BM25: Traditional keyword-based ranking
        - TF-IDF: Semantic similarity-based ranking
        - Combined: Balanced approach using both methods
    - **Search Engines**: Select which search engines to use
    - **Safe Search Level**: Filter level for search results
        - Strict: Most restrictive filtering
        - Moderate: Balanced filtering
        - None: No content filtering
    - **Language**: Preferred language for search results
        - All languages: No language restriction
        - Specific languages: Filter results to selected language
    - **Search Mode**: Control how queries are processed
        - Auto: Automatically choose between knowledge base and web search
        - Web Search Only: Always use web search regardless of query type
    """

if __name__ == "__main__":
    iface = create_gradio_interface()

    # Create the layout with two columns
    with gr.Blocks(theme=gr.Theme.from_hub("allenai/gradio-theme")) as demo:
        with gr.Row():
            with gr.Column(scale=3):
                iface.render()
            with gr.Column(scale=1):
                gr.Markdown(create_parameter_description())

    # Launch the interface
    demo.launch(share=True, debug=True)
