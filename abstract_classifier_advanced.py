#!/usr/bin/env python3
"""
Advanced Abstract Classifier for Computing Research Papers
High-performance classification using parallel processing and multithreading
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
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
import logging
from functools import lru_cache
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from asyncio import Queue, Semaphore, gather, create_task
from itertools import islice
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classifier_advanced.log', mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# OpenAI API Configuration
OPENAI_API_KEY = "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"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Optimized configuration
MAX_REQUESTS_PER_MINUTE = 300
INITIAL_BATCH_SIZE = 25
MAX_BATCH_SIZE = 50
MIN_BATCH_SIZE = 10
CONCURRENT_API_CALLS = 20
CONNECTION_LIMIT = 500
RETRY_ATTEMPTS = 3
RETRY_DELAYS = [0.5, 1.0, 2.0]
CACHE_SIZE = 100000
CACHE_TTL = 7200
CPU_COUNT = mp.cpu_count()
CHUNK_SIZE = 5000
WORKER_THREADS = min(32, CPU_COUNT * 4)
PREFETCH_SIZE = 2000

# Token pricing (GPT-4o-mini)
INPUT_TOKEN_COST = 0.00015 / 1000
OUTPUT_TOKEN_COST = 0.0006 / 1000

# Comprehensive classification schema as provided by user
CLASSIFICATION_SCHEMA = {
    "Computer Science": {
        "code": "CS",
        "description": "Focus on algorithms, software development, technical computing concepts, research-oriented",
        "subfields": {
            "AI_ML": "Artificial Intelligence & Machine Learning",
            "CV": "Computer Vision & Pattern Recognition",
            "NLP": "Natural Language Processing",
            "ROB": "Robotics & Automation",
            "DS": "Data Science & Analytics",
            "DB": "Database Systems & Data Management",
            "NET": "Computer Networks & Distributed Systems",
            "SEC": "Cybersecurity & Information Security",
            "SE": "Software Engineering & Development",
            "HCI": "Human-Computer Interaction",
            "GRAPH": "Computer Graphics & Visualization",
            "OS": "Operating Systems & Systems Programming",
            "ALGO": "Algorithms & Data Structures",
            "THEORY": "Computational Theory & Complexity",
            "BIO": "Bioinformatics & Computational Biology",
            "ARCH": "Computer Architecture & Hardware",
            "MOBILE": "Mobile Computing & Ubiquitous Computing",
            "CLOUD": "Cloud Computing & Virtualization",
            "IOT": "Internet of Things (IoT)",
            "QUANTUM": "Quantum Computing",
            "GAME": "Game Development & Interactive Media",
            "EDU": "Computer Education & Pedagogy",
            "DL": "Digital Libraries & Information Retrieval",
            "HPC": "Parallel Computing & High Performance Computing",
            "EMBED": "Embedded Systems & Real-time Computing"
        }
    },
    "Information Systems": {
        "code": "IS",
        "description": "Focus on business applications, organizational use of technology, enterprise systems",
        "subfields": {
            "ERP": "Enterprise Systems & ERP",
            "BI": "Business Intelligence & Analytics",
            "ECOM": "E-commerce & Digital Business",
            "KM": "Knowledge Management",
            "DSS": "Decision Support Systems",
            "ISM": "Information Systems Management",
            "DT": "Digital Transformation",
            "GOV": "IT Governance & Strategy",
            "BPM": "Business Process Management",
            "SOCIAL": "Social Media & Digital Marketing",
            "HIS": "Healthcare Information Systems",
            "EDTECH": "Educational Technology & E-learning",
            "SCM": "Supply Chain Management Systems",
            "CRM": "Customer Relationship Management (CRM)",
            "EA": "Enterprise Architecture",
            "ITSM": "IT Service Management",
            "INNOV": "Digital Innovation & Entrepreneurship",
            "ISS": "Information Systems Security",
            "PRIV": "Data Governance & Privacy",
            "MBA": "Mobile Business Applications",
            "COLLAB": "Social Computing & Collaboration",
            "PM": "IT Project Management",
            "STRAT": "Digital Strategy & Business Models",
            "RESEARCH": "Information Systems Research Methods",
            "ETHICS": "IT Ethics & Social Responsibility"
        }
    },
    "Information Technology": {
        "code": "IT",
        "description": "Focus on practical IT implementation, infrastructure, operational technology",
        "subfields": {
            "INFRA": "IT Infrastructure & Operations",
            "NETADMIN": "Network Administration & Management",
            "SYSADMIN": "System Administration",
            "SUPPORT": "IT Support & Help Desk",
            "WEBDEV": "Web Development & Technologies",
            "APPDEV": "Mobile App Development",
            "DEVOPS": "DevOps & Continuous Integration",
            "ITSEC": "IT Security & Risk Management",
            "DC": "Data Center Management",
            "CLOUDMGMT": "Cloud Services & Management",
            "ASSET": "IT Asset Management",
            "SERVICE": "IT Service Delivery",
            "FORENSICS": "Digital Forensics",
            "AUDIT": "IT Compliance & Auditing",
            "TELECOM": "Telecommunications & Networking",
            "TRAIN": "IT Training & Education",
            "CONSULT": "IT Consulting & Advisory",
            "EMERGING": "Emerging Technologies Integration",
            "MONITOR": "IT Performance Monitoring",
            "DR": "Disaster Recovery & Business Continuity",
            "VENDOR": "IT Procurement & Vendor Management",
            "WORKPLACE": "Digital Workplace Solutions",
            "AUTO": "IT Automation & Scripting",
            "DOCS": "IT Documentation & Knowledge Management",
            "STANDARDS": "IT Standards & Best Practices"
        }
    }
}

# Computing-related keywords for pre-filtering
COMPUTING_KEYWORDS = {
    # Core computing terms
    'algorithm', 'software', 'computer', 'computing', 'database', 'network',
    'programming', 'system', 'application', 'technology', 'digital', 'cyber',
    'data', 'information', 'code', 'development', 'hardware', 'processor',
    
    # AI/ML terms
    'machine learning', 'artificial intelligence', 'neural', 'deep learning',
    'ai', 'ml', 'model', 'training', 'classification', 'prediction',
    
    # Web/Mobile terms
    'web', 'mobile', 'app', 'browser', 'responsive', 'api', 'framework',
    
    # Cloud/Infrastructure terms
    'cloud', 'server', 'infrastructure', 'virtualization', 'container',
    'kubernetes', 'docker', 'aws', 'azure', 'gcp',
    
    # Security terms
    'security', 'encryption', 'cryptography', 'authentication', 'privacy',
    'blockchain', 'cybersecurity', 'vulnerability', 'threat',
    
    # Data terms
    'big data', 'analytics', 'visualization', 'etl', 'warehouse', 'lake',
    
    # Other domains
    'iot', 'embedded', 'robotics', 'automation', 'quantum', 'bioinformatics',
    'fintech', 'edtech', 'healthtech', 'e-commerce', 'erp', 'crm'
}


class RateLimiter:
    """Token bucket rate limiter with sliding window"""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.window_size = 60  # seconds
        self.requests = deque()
        self.lock = asyncio.Lock()
        self.semaphore = Semaphore(CONCURRENT_API_CALLS)
    
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
    """Thread-safe LRU cache with TTL"""
    
    def __init__(self, maxsize: int, ttl: int):
        self.cache = {}
        self.access_times = {}
        self.maxsize = maxsize
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict]:
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.access_times[key] > self.ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    self.misses += 1
                    return None
                
                # Update access time
                self.access_times[key] = time.time()
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key: str, value: Dict):
        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.maxsize:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class AbstractClassifier:
    """Advanced classifier with improved prompts and error handling"""
    
    def __init__(self, api_key: str, rate_limiter: RateLimiter):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.cache = LRUCache(CACHE_SIZE, CACHE_TTL)
        self.session = None
        self.connector = None
        
        # Performance metrics
        self.api_times = deque(maxlen=100)
        self.current_batch_size = INITIAL_BATCH_SIZE
        self.classification_stats = defaultdict(int)
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60, connect=5, sock_read=30)
        self.connector = aiohttp.TCPConnector(
            limit=CONNECTION_LIMIT,
            limit_per_host=200,
            ttl_dns_cache=300
        )
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    @lru_cache(maxsize=10000)
    def _get_cache_key(self, title: str, abstract: str) -> str:
        """Generate cache key"""
        content = f"{title}:{abstract}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_computing_related(self, title: str, abstract: str) -> bool:
        """Pre-filter to identify computing-related papers"""
        text = (title + " " + abstract).lower()
        
        # Check for at least 2 computing keywords to reduce false negatives
        keyword_count = sum(1 for keyword in COMPUTING_KEYWORDS if keyword in text)
        return keyword_count >= 2
    
    def _build_classification_prompt(self, title: str, abstract: str) -> str:
        """Build comprehensive classification prompt"""
        # Build subfield list for the prompt
        subfield_descriptions = []
        for disc_name, disc_info in CLASSIFICATION_SCHEMA.items():
            subfield_descriptions.append(f"\n{disc_info['code']} - {disc_name} Subfields:")
            for code, name in disc_info['subfields'].items():
                subfield_descriptions.append(f"  {code}: {name}")
        
        subfields_text = "\n".join(subfield_descriptions)
        
        return f"""Please classify this research paper abstract into computing disciplines and subfields.

ABSTRACT: {title}
{abstract[:2000]}

INSTRUCTIONS:
First, determine the PRIMARY DISCIPLINE from these three options:
- Computer Science (CS): Focus on algorithms, software development, technical computing concepts, research-oriented
- Information Systems (IS): Focus on business applications, organizational use of technology, enterprise systems
- Information Technology (IT): Focus on practical IT implementation, infrastructure, operational technology

Then, assign the MOST SPECIFIC SUBFIELD from the comprehensive list provided for that discipline.

AVAILABLE SUBFIELDS BY DISCIPLINE:
{subfields_text}

Provide a confidence score (0-100) for your classification.

If the paper is interdisciplinary, choose the dominant discipline and note this in your reasoning.

RESPONSE FORMAT:
Primary Discipline: [CS/IS/IT]
Subfield: [Specific subfield code from the list above]
Confidence: [0-100]
Reasoning: [Brief explanation of classification]"""
    
    async def _make_api_call(self, prompt: str, retry_count: int = 0) -> Optional[Dict]:
        """Make API call with retry logic"""
        await self.rate_limiter.acquire()
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 150,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                OPENAI_API_URL,
                headers=self.headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Track performance
                    self.api_times.append(time.time() - start_time)
                    
                    # Update token counts
                    if 'usage' in data:
                        self.total_input_tokens += data['usage'].get('prompt_tokens', 0)
                        self.total_output_tokens += data['usage'].get('completion_tokens', 0)
                    
                    return data
                    
                elif response.status == 429 and retry_count < RETRY_ATTEMPTS:
                    # Rate limit hit
                    await asyncio.sleep(RETRY_DELAYS[retry_count])
                    return await self._make_api_call(prompt, retry_count + 1)
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    if retry_count < RETRY_ATTEMPTS:
                        await asyncio.sleep(RETRY_DELAYS[retry_count])
                        return await self._make_api_call(prompt, retry_count + 1)
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("API timeout")
            if retry_count < RETRY_ATTEMPTS:
                return await self._make_api_call(prompt, retry_count + 1)
            return None
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            if retry_count < RETRY_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAYS[retry_count])
                return await self._make_api_call(prompt, retry_count + 1)
            return None
    
    def _parse_classification_response(self, content: str) -> Dict:
        """Parse the classification response"""
        result = {
            'discipline': 'UNKNOWN',
            'subfield': 'UNKNOWN',
            'confidence': 0.0,
            'reasoning': ''
        }
        
        lines = content.strip().split('\n')
        for line in lines:
            line_lower = line.lower()
            
            if 'primary discipline:' in line_lower:
                # Extract discipline code
                for disc_name, disc_info in CLASSIFICATION_SCHEMA.items():
                    if disc_info['code'] in line.upper():
                        result['discipline'] = disc_info['code']
                        break
            
            elif 'subfield:' in line_lower:
                # Extract subfield code
                subfield_part = line.split(':', 1)[1].strip().upper()
                # Check if it's a valid subfield code
                for disc_info in CLASSIFICATION_SCHEMA.values():
                    if subfield_part in disc_info['subfields']:
                        result['subfield'] = subfield_part
                        break
            
            elif 'confidence:' in line_lower:
                # Extract confidence score
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    result['confidence'] = float(conf_str.rstrip('%')) / 100.0
                except:
                    result['confidence'] = 0.5
            
            elif 'reasoning:' in line_lower:
                # Extract reasoning
                result['reasoning'] = line.split(':', 1)[1].strip()
        
        return result
    
    async def classify_abstract(self, paper: Dict) -> Dict:
        """Classify a single abstract"""
        title = paper.get('Title', '')
        abstract = paper.get('Abstract', '')
        
        # Check cache
        cache_key = self._get_cache_key(title, abstract)
        cached = self.cache.get(cache_key)
        if cached:
            return {**paper, **cached}
        
        # Skip if abstract is too short
        if len(abstract.split()) < 30:
            result = {
                'Discipline': 'SKIPPED',
                'Subfield': 'TOO_SHORT',
                'Confidence_Score': 0.0,
                'Reasoning': 'Abstract too short for classification'
            }
            self.cache.set(cache_key, result)
            return {**paper, **result}
        
        # Pre-filter check (less aggressive)
        if not self.is_computing_related(title, abstract):
            # Still try to classify - might be interdisciplinary
            logger.debug(f"Paper may not be computing-related: {title[:50]}")
        
        # Build and send prompt
        prompt = self._build_classification_prompt(title, abstract)
        response = await self._make_api_call(prompt)
        
        if response and 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
            parsed = self._parse_classification_response(content)
            
            result = {
                'Discipline': parsed['discipline'],
                'Subfield': parsed['subfield'],
                'Confidence_Score': parsed['confidence'],
                'Reasoning': parsed['reasoning']
            }
            
            # Track statistics
            self.classification_stats[parsed['discipline']] += 1
            
        else:
            result = {
                'Discipline': 'ERROR',
                'Subfield': 'API_FAILURE',
                'Confidence_Score': 0.0,
                'Reasoning': 'Failed to get API response'
            }
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return {**paper, **result}
    
    def get_total_cost(self) -> float:
        """Calculate total cost"""
        input_cost = self.total_input_tokens * INPUT_TOKEN_COST
        output_cost = self.total_output_tokens * OUTPUT_TOKEN_COST
        return input_cost + output_cost
    
    def adjust_batch_size(self):
        """Dynamically adjust batch size based on performance"""
        if len(self.api_times) >= 10:
            avg_time = sum(list(self.api_times)[-10:]) / 10
            
            if avg_time < 0.5 and self.current_batch_size < MAX_BATCH_SIZE:
                self.current_batch_size = min(int(self.current_batch_size * 1.5), MAX_BATCH_SIZE)
                logger.info(f"Increased batch size to {self.current_batch_size}")
            elif avg_time > 2.0 and self.current_batch_size > MIN_BATCH_SIZE:
                self.current_batch_size = max(int(self.current_batch_size * 0.7), MIN_BATCH_SIZE)
                logger.info(f"Decreased batch size to {self.current_batch_size}")


class BatchProcessor:
    """Batch processor with parallel processing capabilities"""
    
    def __init__(self, classifier: AbstractClassifier):
        self.classifier = classifier
        self.processed_papers = []
        self.failed_papers = []
        self.checkpoint_file = "checkpoint_advanced.pkl"
        self.start_time = time.time()
        self.papers_processed = 0
        self.executor = ThreadPoolExecutor(max_workers=WORKER_THREADS)
        
        # Quality control thresholds
        self.low_confidence_threshold = 0.5
        self.review_papers = []
    
    def save_checkpoint(self):
        """Save processing checkpoint"""
        checkpoint_data = {
            'processed_papers': self.processed_papers,
            'failed_papers': self.failed_papers,
            'review_papers': self.review_papers,
            'cache': dict(self.classifier.cache.cache),
            'total_input_tokens': self.classifier.total_input_tokens,
            'total_output_tokens': self.classifier.total_output_tokens,
            'papers_processed': self.papers_processed,
            'classification_stats': dict(self.classifier.classification_stats)
        }
        
        temp_file = f"{self.checkpoint_file}.tmp"
        with open(temp_file, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_file, self.checkpoint_file)
        logger.info(f"Checkpoint saved: {self.papers_processed} papers processed")
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists"""
        if Path(self.checkpoint_file).exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                self.processed_papers = checkpoint_data['processed_papers']
                self.failed_papers = checkpoint_data['failed_papers']
                self.review_papers = checkpoint_data.get('review_papers', [])
                
                # Restore cache
                for key, value in checkpoint_data['cache'].items():
                    self.classifier.cache.set(key, value)
                
                self.classifier.total_input_tokens = checkpoint_data['total_input_tokens']
                self.classifier.total_output_tokens = checkpoint_data['total_output_tokens']
                self.papers_processed = checkpoint_data['papers_processed']
                self.classifier.classification_stats = defaultdict(int, checkpoint_data.get('classification_stats', {}))
                
                logger.info(f"Checkpoint loaded: {self.papers_processed} papers already processed")
                return True
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {str(e)}")
                return False
        return False
    
    def get_processed_ids(self) -> Set[str]:
        """Get set of already processed paper IDs"""
        processed_ids = set()
        for paper in self.processed_papers:
            paper_id = self._get_paper_id(paper)
            processed_ids.add(paper_id)
        return processed_ids
    
    def _get_paper_id(self, paper: Dict) -> str:
        """Generate unique paper ID"""
        title = paper.get('Title', '')
        abstract = paper.get('Abstract', '')
        return hashlib.md5(f"{title}:{abstract}".encode()).hexdigest()
    
    async def process_batch(self, papers: List[Dict]) -> List[Dict]:
        """Process a batch of papers"""
        # Adjust batch size dynamically
        self.classifier.adjust_batch_size()
        
        # Process papers concurrently
        tasks = []
        for paper in papers:
            task = create_task(self.classifier.classify_abstract(paper))
            tasks.append(task)
        
        results = await gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Classification failed: {str(result)}")
                failed_paper = {
                    **papers[i],
                    'Discipline': 'ERROR',
                    'Subfield': 'EXCEPTION',
                    'Confidence_Score': 0.0,
                    'Reasoning': str(result)
                }
                self.failed_papers.append(failed_paper)
                processed_results.append(failed_paper)
            else:
                # Quality control check
                if result.get('Confidence_Score', 0) < self.low_confidence_threshold:
                    self.review_papers.append(result)
                processed_results.append(result)
        
        return processed_results
    
    def read_csv_parallel(self, filename: str) -> List[List[Dict]]:
        """Read CSV file in parallel chunks"""
        processed_ids = self.get_processed_ids()
        
        # Count total rows
        with open(filename, 'r', encoding='utf-8') as f:
            total_rows = sum(1 for _ in f) - 1  # Subtract header
        
        logger.info(f"Total rows in CSV: {total_rows}")
        
        chunks = []
        current_chunk = []
        papers_to_process = 0
        
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                paper_id = self._get_paper_id(row)
                
                # Skip if already processed
                if paper_id in processed_ids:
                    continue
                
                # Skip if no abstract
                if not row.get('Abstract', '').strip():
                    continue
                
                current_chunk.append(row)
                papers_to_process += 1
                
                if len(current_chunk) >= CHUNK_SIZE:
                    chunks.append(current_chunk)
                    current_chunk = []
            
            if current_chunk:
                chunks.append(current_chunk)
        
        logger.info(f"Papers to process: {papers_to_process} (already processed: {len(processed_ids)})")
        logger.info(f"Split into {len(chunks)} chunks")
        
        return chunks
    
    async def process_csv(self, input_file: str, output_file: str):
        """Main processing function"""
        # Read CSV chunks
        chunks = self.read_csv_parallel(input_file)
        
        if not chunks:
            logger.warning("No papers to process!")
            return
        
        total_papers = sum(len(chunk) for chunk in chunks)
        logger.info(f"Starting classification of {total_papers} papers")
        
        # Process chunks
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} papers)")
            
            # Process in smaller batches within chunk
            for j in range(0, len(chunk), self.classifier.current_batch_size):
                batch = chunk[j:j + self.classifier.current_batch_size]
                
                results = await self.process_batch(batch)
                self.processed_papers.extend(results)
                self.papers_processed += len(results)
                
                # Progress update
                self.print_progress(total_papers)
                
                # Save checkpoint periodically
                if self.papers_processed % 500 == 0:
                    self.save_checkpoint()
        
        print()  # New line after progress
        logger.info(f"Processing complete: {self.papers_processed} papers processed")
        
        # Save results
        self.save_results(output_file)
    
    def print_progress(self, total_papers: int):
        """Print progress information"""
        elapsed_time = time.time() - self.start_time
        papers_per_minute = (self.papers_processed / elapsed_time) * 60 if elapsed_time > 0 else 0
        remaining_papers = total_papers - self.papers_processed
        estimated_time = (remaining_papers / papers_per_minute / 60) if papers_per_minute > 0 else 0
        cost = self.classifier.get_total_cost()
        cache_hit_rate = self.classifier.cache.get_hit_rate()
        
        print(f"\rProcessed {self.papers_processed}/{total_papers} ({self.papers_processed/total_papers*100:.1f}%), "
              f"Rate: {papers_per_minute:.0f}/min, "
              f"Cache: {cache_hit_rate:.1%}, "
              f"Est: {estimated_time:.1f}h, "
              f"Cost: ${cost:.2f}", end='', flush=True)
    
    def save_results(self, output_file: str):
        """Save all results"""
        # Save main results
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if self.processed_papers:
                fieldnames = list(self.processed_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.processed_papers)
        
        # Save low confidence papers for review
        if self.review_papers:
            review_file = output_file.replace('.csv', '_review.csv')
            with open(review_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.review_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.review_papers)
            logger.info(f"Saved {len(self.review_papers)} papers for review to {review_file}")
        
        # Save failed papers
        if self.failed_papers:
            error_file = output_file.replace('.csv', '_errors.csv')
            with open(error_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.failed_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.failed_papers)
            logger.info(f"Saved {len(self.failed_papers)} failed papers to {error_file}")
        
        # Save statistics
        self.save_statistics(output_file)
    
    def save_statistics(self, output_file: str):
        """Save classification statistics"""
        stats = {
            'total_processed': len(self.processed_papers),
            'total_failed': len(self.failed_papers),
            'total_review': len(self.review_papers),
            'total_cost': self.classifier.get_total_cost(),
            'processing_time_seconds': time.time() - self.start_time,
            'average_papers_per_minute': self.papers_processed / ((time.time() - self.start_time) / 60),
            'cache_hit_rate': self.classifier.cache.get_hit_rate(),
            'disciplines': dict(self.classifier.classification_stats),
            'subfields': {},
            'confidence_distribution': {
                'high (>0.7)': 0,
                'medium (0.5-0.7)': 0,
                'low (<0.5)': 0
            }
        }
        
        # Analyze results
        for paper in self.processed_papers:
            # Subfield counts
            subfield = paper.get('Subfield', 'UNKNOWN')
            stats['subfields'][subfield] = stats['subfields'].get(subfield, 0) + 1
            
            # Confidence distribution
            conf = paper.get('Confidence_Score', 0)
            if conf > 0.7:
                stats['confidence_distribution']['high (>0.7)'] += 1
            elif conf >= 0.5:
                stats['confidence_distribution']['medium (0.5-0.7)'] += 1
            else:
                stats['confidence_distribution']['low (<0.5)'] += 1
        
        # Calculate rates
        total = len(self.processed_papers)
        if total > 0:
            stats['unknown_rate'] = stats['disciplines'].get('UNKNOWN', 0) / total
            stats['error_rate'] = stats['disciplines'].get('ERROR', 0) / total
        
        stats_file = output_file.replace('.csv', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n=== Classification Summary ===")
        print(f"Total papers processed: {stats['total_processed']}")
        print(f"Failed classifications: {stats['total_failed']}")
        print(f"Papers flagged for review: {stats['total_review']}")
        print(f"Total cost: ${stats['total_cost']:.2f}")
        print(f"Processing time: {stats['processing_time_seconds']/3600:.1f} hours")
        print(f"Average rate: {stats['average_papers_per_minute']:.0f} papers/minute")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
        print("\nDiscipline distribution:")
        for disc, count in sorted(stats['disciplines'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {disc}: {count} ({percentage:.1f}%)")
        
        print("\nTop 10 subfields:")
        sorted_subfields = sorted(stats['subfields'].items(), key=lambda x: x[1], reverse=True)[:10]
        for subfield, count in sorted_subfields:
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {subfield}: {count} ({percentage:.1f}%)")
        
        print("\nConfidence distribution:")
        for level, count in stats['confidence_distribution'].items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {level}: {count} ({percentage:.1f}%)")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Advanced abstract classifier')
    parser.add_argument('--input', default='Abstracts.csv', help='Input CSV file')
    parser.add_argument('--output', default='classified_papers_advanced.csv', help='Output CSV file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return
    
    # Set API key
    global OPENAI_API_KEY
    if args.api_key:
        OPENAI_API_KEY = args.api_key
    elif not OPENAI_API_KEY or OPENAI_API_KEY == "":
        logger.error("Please provide OpenAI API key via --api-key or OPENAI_API_KEY environment variable")
        return
    
    # Initialize components
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)
    
    async with AbstractClassifier(OPENAI_API_KEY, rate_limiter) as classifier:
        processor = BatchProcessor(classifier)
        
        # Load checkpoint if resuming
        if args.resume:
            processor.load_checkpoint()
        
        try:
            # Process CSV
            await processor.process_csv(args.input, args.output)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving checkpoint...")
            processor.save_checkpoint()
            print("Checkpoint saved. Use --resume to continue.")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            processor.save_checkpoint()
            raise


if __name__ == "__main__":
    # Run with optimized settings
    asyncio.run(main())