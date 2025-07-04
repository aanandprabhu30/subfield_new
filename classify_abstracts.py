#!/usr/bin/env python3
"""
Research Paper Abstract Classification System
High-performance classifier using OpenAI GPT models with parallel processing
"""

import asyncio
import aiohttp
import csv
import json
import time
import pickle
import hashlib
import argparse
import os
import sys
import signal
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import logging
from functools import lru_cache
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
from asyncio import Queue, Semaphore, gather, create_task

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# --- Rate Limit and Performance Settings ---
MAX_CONCURRENT_CALLS = 100  # Increased from 20
BATCH_SIZE = 100  # Increased from 50
MAX_REQUESTS_PER_MINUTE = 1500  # Increased from 150
CONFIDENCE_THRESHOLD = 70  # Threshold for fallback to GPT-4o
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N papers

# Model configuration
PRIMARY_MODEL = "gpt-4o-mini"  # Cost-effective primary model
FALLBACK_MODEL = "gpt-4o"  # More capable fallback model

# Token pricing
PRICING = {
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000}
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Classification schema
CLASSIFICATION_SCHEMA = {
    "Computer Science": {
        "code": "CS",
        "description": "Theoretical algorithms, software development, technical computing research",
        "subfields": [
            "Artificial Intelligence & Machine Learning",
            "Computer Vision & Pattern Recognition",
            "Natural Language Processing",
            "Robotics & Automation",
            "Data Science & Analytics",
            "Database Systems & Data Management",
            "Computer Networks & Distributed Systems",
            "Cybersecurity & Information Security",
            "Software Engineering & Development",
            "Human-Computer Interaction",
            "Computer Graphics & Visualization",
            "Operating Systems & Systems Programming",
            "Algorithms & Data Structures",
            "Computational Theory & Complexity",
            "Bioinformatics & Computational Biology",
            "Computer Architecture & Hardware",
            "Mobile Computing & Ubiquitous Computing",
            "Cloud Computing & Virtualization",
            "Internet of Things (IoT)",
            "Quantum Computing",
            "Game Development & Interactive Media",
            "Computer Education & Pedagogy",
            "Digital Libraries & Information Retrieval",
            "Parallel Computing & High Performance Computing",
            "Embedded Systems & Real-time Computing"
        ]
    },
    "Information Systems": {
        "code": "IS",
        "description": "Business applications, organizational technology use, enterprise systems",
        "subfields": [
            "Enterprise Systems & ERP",
            "Business Intelligence & Analytics",
            "E-commerce & Digital Business",
            "Knowledge Management",
            "Decision Support Systems",
            "Information Systems Management",
            "Digital Transformation",
            "IT Governance & Strategy",
            "Business Process Management",
            "Social Media & Digital Marketing",
            "Healthcare Information Systems",
            "Educational Technology & E-learning",
            "Supply Chain Management Systems",
            "Customer Relationship Management (CRM)",
            "Enterprise Architecture",
            "IT Service Management",
            "Digital Innovation & Entrepreneurship",
            "Information Systems Security",
            "Data Governance & Privacy",
            "Mobile Business Applications",
            "Social Computing & Collaboration",
            "IT Project Management",
            "Digital Strategy & Business Models",
            "Information Systems Research Methods",
            "IT Ethics & Social Responsibility"
        ]
    },
    "Information Technology": {
        "code": "IT",
        "description": "Practical implementation, infrastructure, operational technology",
        "subfields": [
            "IT Infrastructure & Operations",
            "Network Administration & Management",
            "System Administration",
            "IT Support & Help Desk",
            "Web Development & Technologies",
            "Mobile App Development",
            "DevOps & Continuous Integration",
            "IT Security & Risk Management",
            "Data Center Management",
            "Cloud Services & Management",
            "IT Asset Management",
            "IT Service Delivery",
            "Digital Forensics",
            "IT Compliance & Auditing",
            "Telecommunications & Networking",
            "IT Training & Education",
            "IT Consulting & Advisory",
            "Emerging Technologies Integration",
            "IT Performance Monitoring",
            "Disaster Recovery & Business Continuity",
            "IT Procurement & Vendor Management",
            "Digital Workplace Solutions",
            "IT Automation & Scripting",
            "IT Documentation & Knowledge Management",
            "IT Standards & Best Practices"
        ]
    }
}

# --- Enhanced Computing Keywords and Pre-filter ---
COMPUTING_KEYWORDS = {
    'algorithm', 'software', 'system', 'data', 'network', 'database',
    'programming', 'code', 'application', 'platform', 'framework',
    'computer', 'computing', 'computational', 'digital', 'cyber',
    'ai', 'ml', 'artificial intelligence', 'machine learning', 'deep learning',
    'neural', 'cloud', 'blockchain', 'iot', 'internet of things',
    'api', 'microservice', 'container', 'kubernetes', 'devops',
    'information system', 'information technology', 'it ', ' it ',
    'enterprise', 'erp', 'crm', 'business intelligence', 'analytics',
    'implementation', 'deployment', 'infrastructure', 'architecture',
    'acm', 'ieee', 'software engineering', 'computer science',
    'information management', 'technology adoption', 'digital transformation'
}

def is_computing_related(title: str, abstract: str) -> bool:
    text = (title + " " + abstract).lower()
    title_lower = title.lower()
    title_matches = sum(1 for kw in COMPUTING_KEYWORDS if kw in title_lower)
    abstract_matches = sum(1 for kw in COMPUTING_KEYWORDS if kw in text)
    score = (title_matches * 3) + abstract_matches
    return score >= 5

# --- Quick LLM Check for Computing ---
async def _check_if_computing(self, title: str, abstract: str) -> bool:
    prompt = f"""Is this paper about computing, information technology, or computer science?\n\nTitle: {title}\nAbstract: {abstract[:500]}\n\nAnswer with just YES or NO."""
    response = await self._make_api_call(prompt, PRIMARY_MODEL, max_tokens=5)
    if response and 'choices' in response:
        content = response['choices'][0]['message']['content'].strip().upper()
        return 'YES' in content
    return True

# --- Two-Stage Classification Prompts ---
def _build_discipline_prompt(self, title: str, abstract: str) -> str:
    return f"""Classify this research paper into ONE primary discipline.\n\nTitle: {title}\nAbstract: {abstract[:1500]}\n\nDISCIPLINES:\n- CS (Computer Science): Theoretical algorithms, software development, technical research\n- IS (Information Systems): Business applications, organizational IT, enterprise systems\n- IT (Information Technology): Practical implementation, infrastructure, operations\n\nConsider the paper's PRIMARY focus. Many papers blur boundaries - choose the dominant aspect.\n\nRespond with ONLY: CS, IS, or IT"""

def _build_subfield_prompt(self, discipline: str, title: str, abstract: str) -> str:
    disc_name = {'CS': 'Computer Science', 'IS': 'Information Systems', 'IT': 'Information Technology'}[discipline]
    subfields = CLASSIFICATION_SCHEMA[disc_name]['subfields']
    subfield_list = '\n'.join([f"{i}. {sf}" for i, sf in enumerate(subfields, 1)])
    return f"""This {disc_name} paper needs subfield classification.\n\nTitle: {title}\nAbstract: {abstract[:1500]}\n\nAVAILABLE SUBFIELDS:\n{subfield_list}\n\nChoose the MOST SPECIFIC subfield that best describes this paper's main contribution.\nRespond with the EXACT subfield name from the list above."""


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.window_size = 60  # seconds
        self.requests = deque()
        self.lock = asyncio.Lock()
        self.semaphore = Semaphore(MAX_CONCURRENT_CALLS)
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.semaphore:
            async with self.lock:
                now = time.time()
                
                # Remove old requests outside window
                while self.requests and self.requests[0] < now - self.window_size:
                    self.requests.popleft()
                
                # Check if we can make a request
                if len(self.requests) >= self.max_requests:
                    # Calculate wait time
                    oldest_request = self.requests[0]
                    wait_time = oldest_request + self.window_size - now + 0.1
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        return await self.acquire()
                
                # Record this request
                self.requests.append(now)


class LRUCache:
    """Thread-safe LRU cache"""
    
    def __init__(self, maxsize: int = 100000):
        self.cache = {}
        self.access_times = {}
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict]:
        with self.lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Dict):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                # Evict oldest
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class OptimizedClassificationPipeline:
    """Main classification pipeline with parallel processing"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)
        self.cache = LRUCache()
        self.session = None
        self.connector = None
        
        # Statistics
        self.token_usage = defaultdict(lambda: defaultdict(int))
        self.model_usage = defaultdict(int)
        self.classification_stats = defaultdict(int)
        
        # Checkpoint data
        self.processed_papers = []
        self.failed_papers = []
        self.low_confidence_papers = []
        self.checkpoint_file = "checkpoint.pkl"
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Shutdown requested, saving checkpoint...")
        self.shutdown_requested = True
    
    async def __aenter__(self):
        """Initialize async resources"""
        timeout = aiohttp.ClientTimeout(total=60, connect=5, sock_read=30)
        self.connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=500,
            ttl_dns_cache=300,
            keepalive_timeout=30,
            force_close=False
        )
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    def _get_cache_key(self, title: str, abstract: str) -> str:
        """Generate cache key"""
        content = f"{title}:{abstract}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _build_primary_prompt(self, title: str, abstract: str) -> str:
        """Build primary classification prompt for GPT-4o-mini"""
        # Build subfield lists
        subfield_text = ""
        for disc_name, disc_info in CLASSIFICATION_SCHEMA.items():
            subfield_text += f"\n\n{disc_info['code']} - {disc_name} ({disc_info['description']}):\n"
            for i, subfield in enumerate(disc_info['subfields'], 1):
                subfield_text += f"{i}. {subfield}\n"
        
        return f"""You are an expert in computer science and information systems classification with deep knowledge of academic research areas.

Classify this research paper into exactly ONE primary discipline and ONE specific subfield.

PAPER DETAILS:
Title: {title}
Abstract: {abstract[:2000]}

CLASSIFICATION RULES:
1. Choose the PRIMARY DISCIPLINE that best represents the paper's main focus:
   - Computer Science (CS): Theoretical foundations, algorithms, software development, technical research
   - Information Systems (IS): Business applications, organizational IT use, enterprise systems, IT management
   - Information Technology (IT): Practical implementation, infrastructure, operations, deployment

2. Select the MOST SPECIFIC SUBFIELD from the comprehensive list for your chosen discipline.

3. If the paper seems unrelated to computing/IT, respond with:
   Primary Discipline: NON_COMPUTING
   Subfield: NOT_APPLICABLE

4. For interdisciplinary papers, choose the dominant computing aspect.

AVAILABLE SUBFIELDS BY DISCIPLINE:
{subfield_text}

RESPONSE FORMAT (exactly as shown):
Primary Discipline: [CS/IS/IT/NON_COMPUTING]
Subfield: [Exact subfield name from the list]
Confidence: [0-100]
Reasoning: [One sentence explaining your classification]"""
    
    def _build_fallback_prompt(self, title: str, abstract: str) -> str:
        """Build fallback classification prompt for GPT-4o"""
        # Build subfield lists
        subfield_text = ""
        for disc_name, disc_info in CLASSIFICATION_SCHEMA.items():
            subfield_text += f"\n\n{disc_info['code']} - {disc_name} ({disc_info['description']}):\n"
            for i, subfield in enumerate(disc_info['subfields'], 1):
                subfield_text += f"{i}. {subfield}\n"
        
        return f"""You are a senior academic reviewer specializing in computing research classification. This paper requires careful analysis as initial classification was uncertain.

PAPER REQUIRING DETAILED ANALYSIS:
Title: {title}
Abstract: {abstract[:2000]}

CONTEXT: Initial classification attempt yielded low confidence or unclear results. Please provide a thorough analysis.

INSTRUCTIONS:
1. First, identify ALL computing-related aspects in the abstract
2. Determine if this is genuinely a computing paper or from another field
3. If computing-related, identify the primary focus area
4. Match to the most appropriate discipline and subfield from our taxonomy

DISCIPLINE AND SUBFIELD TAXONOMY:
{subfield_text}

DETAILED RESPONSE REQUIRED:
Primary Discipline: [CS/IS/IT/NON_COMPUTING]
Subfield: [Exact subfield name]
Confidence: [0-100]
Key Indicators: [List 2-3 specific phrases/concepts that support this classification]
Alternative Classification: [If applicable, note second-best choice]
Reasoning: [2-3 sentences explaining the classification decision]"""
    
    async def _make_api_call(self, prompt: str, model: str, retry_count: int = 0, max_tokens: int = 200) -> Optional[Dict]:
        """Make API call with retry logic"""
        await self.rate_limiter.acquire()
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
        
        try:
            async with self.session.post(
                OPENAI_API_URL,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Update token usage
                    if 'usage' in data:
                        usage = data['usage']
                        self.token_usage[model]['input'] += usage.get('prompt_tokens', 0)
                        self.token_usage[model]['output'] += usage.get('completion_tokens', 0)
                    
                    return data
                
                elif response.status == 429 and retry_count < 5:
                    # Rate limit - exponential backoff
                    wait_time = min(2 ** retry_count, 32)
                    logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                    return await self._make_api_call(prompt, model, retry_count + 1)
                
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return None
        
        except Exception as e:
            logger.error(f"API exception: {str(e)}")
            if retry_count < 3:
                await asyncio.sleep(2 ** retry_count)
                return await self._make_api_call(prompt, model, retry_count + 1)
            return None
    
    def _parse_response(self, content: str) -> Dict:
        """Parse classification response"""
        result = {
            'discipline': 'UNKNOWN',
            'subfield': 'UNKNOWN',
            'confidence': 0,
            'reasoning': '',
            'key_indicators': [],
            'alternative': ''
        }
        
        try:
            lines = content.strip().split('\n')
            for line in lines:
                line_lower = line.lower()
                
                if 'primary discipline:' in line_lower:
                    disc_part = line.split(':', 1)[1].strip()
                    if disc_part in ['CS', 'IS', 'IT', 'NON_COMPUTING']:
                        result['discipline'] = disc_part
                
                elif 'subfield:' in line_lower:
                    subfield_part = line.split(':', 1)[1].strip()
                    # Validate subfield exists in schema
                    for disc_info in CLASSIFICATION_SCHEMA.values():
                        if subfield_part in disc_info['subfields']:
                            result['subfield'] = subfield_part
                            break
                    if result['subfield'] == 'UNKNOWN' and subfield_part == 'NOT_APPLICABLE':
                        result['subfield'] = 'NOT_APPLICABLE'
                
                elif 'confidence:' in line_lower:
                    try:
                        conf_str = line.split(':', 1)[1].strip()
                        result['confidence'] = int(conf_str.rstrip('%'))
                    except:
                        result['confidence'] = 50
                
                elif 'reasoning:' in line_lower:
                    result['reasoning'] = line.split(':', 1)[1].strip()
                
                elif 'key indicators:' in line_lower:
                    indicators = line.split(':', 1)[1].strip()
                    result['key_indicators'] = [i.strip() for i in indicators.split(',')]
                
                elif 'alternative classification:' in line_lower:
                    result['alternative'] = line.split(':', 1)[1].strip()
        
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
        
        return result
    
    async def classify_paper(self, paper: Dict) -> Dict:
        """Complete classification pipeline for a single paper"""
        title = paper.get('Title', '').strip()
        abstract = paper.get('Abstract', '').strip()
        
        # Check cache
        cache_key = self._get_cache_key(title, abstract)
        cached = self.cache.get(cache_key)
        if cached:
            return {**paper, **cached, 'Cached': True}
        
        # Validate input
        if not title or not abstract or len(abstract.split()) < 20:
            result = {
                'Primary Discipline': 'SKIPPED',
                'Subfield': 'INSUFFICIENT_CONTENT',
                'Confidence': 100,
                'Model Used': 'validation',
                'Reasoning': 'Insufficient content for classification',
                'Cached': False
            }
            self.cache.set(cache_key, result)
            return {**paper, **result}
        
        # Pre-filter
        if not is_computing_related(title, abstract):
            is_computing = await self._check_if_computing(title, abstract)
            if not is_computing:
                result = {
                    'Primary Discipline': 'NON_COMPUTING',
                    'Subfield': 'NOT_APPLICABLE',
                    'Confidence': 95,
                    'Model Used': 'pre-filter',
                    'Reasoning': 'Paper is not related to computing or information technology',
                    'Cached': False
                }
                self.cache.set(cache_key, result)
                return {**paper, **result}
        
        # Stage 1: Discipline
        disc_prompt = self._build_discipline_prompt(title, abstract)
        disc_response = await self._make_api_call(disc_prompt, PRIMARY_MODEL, max_tokens=10)
        discipline = 'UNKNOWN'
        if disc_response and 'choices' in disc_response:
            content = disc_response['choices'][0]['message']['content'].strip().upper()
            if content in ['CS', 'IS', 'IT']:
                discipline = content
        
        # Fallback if needed
        if discipline == 'UNKNOWN':
            logger.info(f"Using fallback for discipline: {title[:50]}")
            fallback_response = await self._make_api_call(self._build_fallback_prompt(title, abstract), FALLBACK_MODEL)
            # Parse fallback response if needed
        
        # Stage 2: Subfield
        subfield = 'UNKNOWN'
        if discipline in ['CS', 'IS', 'IT']:
            subfield_prompt = self._build_subfield_prompt(discipline, title, abstract)
            subfield_response = await self._make_api_call(subfield_prompt, PRIMARY_MODEL, max_tokens=50)
            if subfield_response and 'choices' in subfield_response:
                subfield_content = subfield_response['choices'][0]['message']['content'].strip()
                subfield = subfield_content
        
        result = {
            'Primary Discipline': discipline,
            'Subfield': subfield,
            'Confidence': 80 if discipline != 'UNKNOWN' and subfield != 'UNKNOWN' else 0,
            'Model Used': PRIMARY_MODEL,
            'Reasoning': f"Discipline: {discipline}, Subfield: {subfield}",
            'Cached': False
        }
        self.cache.set(cache_key, result)
        return {**paper, **result}
    
    async def process_batch(self, papers: List[Dict]) -> List[Dict]:
        """Process a batch of papers concurrently"""
        computing_papers = []
        non_computing_results = []
        for paper in papers:
            if is_computing_related(paper['Title'], paper['Abstract']):
                computing_papers.append(paper)
            else:
                non_computing_results.append({
                    **paper,
                    'Primary Discipline': 'NON_COMPUTING',
                    'Subfield': 'NOT_APPLICABLE',
                    'Confidence': 95,
                    'Model Used': 'pre-filter',
                    'Reasoning': 'No computing keywords found',
                    'Cached': False
                })
        computing_results = []
        if computing_papers:
            tasks = [self.classify_paper(paper) for paper in computing_papers]
            computing_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Flatten and handle exceptions
            final_results = []
            for res in computing_results:
                if isinstance(res, Exception):
                    final_results.append({'Primary Discipline': 'ERROR', 'Subfield': 'ERROR', 'Confidence': 0, 'Model Used': 'ERROR', 'Reasoning': str(res), 'Cached': False})
                else:
                    final_results.append(res)
            computing_results = final_results
        return non_computing_results + computing_results
    
    def save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint_data = {
            'processed_papers': self.processed_papers,
            'failed_papers': self.failed_papers,
            'low_confidence_papers': self.low_confidence_papers,
            'classification_stats': dict(self.classification_stats),
            'token_usage': dict(self.token_usage),
            'model_usage': dict(self.model_usage),
            'cache_stats': {
                'hits': self.cache.hits,
                'misses': self.cache.misses,
                'size': len(self.cache.cache)
            }
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {len(self.processed_papers)} papers processed")
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                self.processed_papers = checkpoint_data['processed_papers']
                self.failed_papers = checkpoint_data['failed_papers']
                self.low_confidence_papers = checkpoint_data.get('low_confidence_papers', [])
                self.classification_stats = defaultdict(int, checkpoint_data['classification_stats'])
                self.token_usage = defaultdict(lambda: defaultdict(int), checkpoint_data['token_usage'])
                self.model_usage = defaultdict(int, checkpoint_data.get('model_usage', {}))
                
                logger.info(f"Checkpoint loaded: {len(self.processed_papers)} papers already processed")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                return False
        return False
    
    def get_processed_ids(self) -> Set[str]:
        """Get set of already processed paper IDs"""
        processed_ids = set()
        for paper in self.processed_papers:
            paper_id = self._get_cache_key(paper.get('Title', ''), paper.get('Abstract', ''))
            processed_ids.add(paper_id)
        return processed_ids
    
    async def process_csv(self, input_file: str, output_file: str, resume: bool = False, start_index: int = 0):
        """Process CSV file with papers"""
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()
        
        processed_ids = self.get_processed_ids()
        
        # Read papers to process
        papers_to_process = []
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Skip rows before start_index
                if i < start_index:
                    continue
                    
                paper_id = self._get_cache_key(row.get('Title', ''), row.get('Abstract', ''))
                if paper_id not in processed_ids:
                    papers_to_process.append(row)
        
        total_papers = len(papers_to_process)
        logger.info(f"Papers to process: {total_papers} (already processed: {len(processed_ids)})")
        
        if total_papers == 0:
            logger.info("No new papers to process")
            return
        
        # Process in batches
        start_time = time.time()
        papers_processed = 0
        
        for i in range(0, total_papers, BATCH_SIZE):
            if self.shutdown_requested:
                logger.info("Shutdown requested, saving progress...")
                break
            
            batch = papers_to_process[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_papers + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            # Process batch
            results = await self.process_batch(batch)
            self.processed_papers.extend(results)
            papers_processed += len(results)
            
            # Progress update
            elapsed = time.time() - start_time
            rate = papers_processed / elapsed * 60 if elapsed > 0 else 0
            eta = (total_papers - papers_processed) / rate if rate > 0 else 0
            cost = self.get_total_cost()
            
            logger.info(
                f"Progress: {papers_processed}/{total_papers} ({papers_processed/total_papers*100:.1f}%), "
                f"Rate: {rate:.0f}/min, ETA: {eta:.1f}min, Cost: ${cost:.2f}"
            )
            
            # Save checkpoint periodically
            if papers_processed % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint()
        
        # Save results
        await self.save_results_streaming(output_file)
        
        # Print final statistics
        self.print_statistics()
    
    def get_total_cost(self) -> float:
        """Calculate total cost"""
        total_cost = 0
        for model, usage in self.token_usage.items():
            if model in PRICING:
                total_cost += (
                    usage['input'] * PRICING[model]['input'] +
                    usage['output'] * PRICING[model]['output']
                )
        return total_cost
    
    async def save_results_streaming(self, output_file: str):
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = None
            for paper in self.processed_papers:
                if writer is None:
                    fieldnames = list(paper.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow(paper)
                if len(self.processed_papers) > 10000:
                    self.processed_papers.pop(0)
    
    def save_results(self, output_file: str):
        """Save all results to CSV files"""
        # Main results
        if self.processed_papers:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.processed_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.processed_papers)
            logger.info(f"Results saved to {output_file}")
        
        # Low confidence papers
        if self.low_confidence_papers:
            review_file = output_file.replace('.csv', '_review.csv')
            with open(review_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.low_confidence_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.low_confidence_papers)
            logger.info(f"Low confidence papers saved to {review_file}")
        
        # Failed papers
        if self.failed_papers:
            error_file = output_file.replace('.csv', '_errors.csv')
            with open(error_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.failed_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.failed_papers)
            logger.info(f"Failed papers saved to {error_file}")
        
        # Statistics
        self.save_statistics(output_file)
    
    def save_statistics(self, output_file: str):
        """Save classification statistics"""
        stats = {
            'total_processed': len(self.processed_papers),
            'total_failed': len(self.failed_papers),
            'total_low_confidence': len(self.low_confidence_papers),
            'total_cost': self.get_total_cost(),
            'model_usage': dict(self.model_usage),
            'token_usage': dict(self.token_usage),
            'classification_distribution': dict(self.classification_stats),
            'cache_hit_rate': self.cache.get_hit_rate(),
            'cache_size': len(self.cache.cache),
            'subfield_distribution': self._calculate_subfield_distribution()
        }
        
        stats_file = output_file.replace('.csv', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
    
    def _calculate_subfield_distribution(self) -> Dict[str, int]:
        """Calculate subfield distribution"""
        distribution = defaultdict(int)
        for paper in self.processed_papers:
            subfield = paper.get('Subfield', 'UNKNOWN')
            distribution[subfield] += 1
        return dict(distribution)
    
    def print_statistics(self):
        """Print final statistics"""
        print("\n" + "="*60)
        print("CLASSIFICATION STATISTICS")
        print("="*60)
        
        total = len(self.processed_papers)
        print(f"Total papers processed: {total:,}")
        print(f"Failed classifications: {len(self.failed_papers):,}")
        print(f"Low confidence papers: {len(self.low_confidence_papers):,}")
        print(f"Total cost: ${self.get_total_cost():.2f}")
        print(f"Cache hit rate: {self.cache.get_hit_rate():.1%}")
        
        print("\nModel Usage:")
        for model, count in self.model_usage.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {model}: {count:,} ({percentage:.1f}%)")
        
        print("\nDiscipline Distribution:")
        for disc, count in sorted(self.classification_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {disc}: {count:,} ({percentage:.1f}%)")
        
        # Calculate success metrics
        computing_papers = total - self.classification_stats.get('NON_COMPUTING', 0)
        unknown_papers = self.classification_stats.get('UNKNOWN', 0)
        unknown_rate = (unknown_papers / computing_papers * 100) if computing_papers > 0 else 0
        
        print(f"\nSuccess Metrics:")
        print(f"  Computing papers: {computing_papers:,}")
        print(f"  UNKNOWN rate: {unknown_rate:.1f}%")
        print(f"  Average confidence: {self._calculate_avg_confidence():.1f}%")


    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence score"""
        total_confidence = 0
        count = 0
        for paper in self.processed_papers:
            conf = paper.get('Confidence', 0)
            if conf > 0:  # Exclude errors
                total_confidence += conf
                count += 1
        return (total_confidence / count) if count > 0 else 0


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Classify research paper abstracts')
    parser.add_argument('--input', default='Abstracts.csv', help='Input CSV file')
    parser.add_argument('--output', default='classified_papers.csv', help='Output CSV file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--start-index', type=int, default=0, help='Start processing from this row index (0-based)')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or OPENAI_API_KEY
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run classification
    async with OptimizedClassificationPipeline(api_key) as pipeline:
        try:
            await pipeline.process_csv(args.input, args.output, args.resume, args.start_index)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving checkpoint...")
            pipeline.save_checkpoint()
            print("Checkpoint saved. Use --resume to continue.")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            pipeline.save_checkpoint()
            raise


if __name__ == "__main__":
    asyncio.run(main())