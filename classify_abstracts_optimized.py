#!/usr/bin/env python3
"""
Research Paper Abstract Classification System - Optimized Version
High-performance classifier with robust error handling and reliability improvements
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
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
import logging
from collections import deque, defaultdict
import threading
from asyncio import Semaphore, gather, create_task, wait_for
from asyncio.exceptions import TimeoutError
from dataclasses import dataclass
import traceback

# Rate limiting - more conservative
RATE_LIMIT_REQUESTS = 45  # Reduced from 60
RATE_LIMIT_TOKENS = 90000  # Reduced from 150000

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# Configuration
BATCH_SIZE = 25  # Reduced from 50
MAX_CONCURRENT_CALLS = 3  # Reduced from 5
MAX_RETRIES = 3
MAX_RETRY_DELAY = 60
CHECKPOINT_INTERVAL = 50
CONFIDENCE_THRESHOLD = 50  # Lowered from 70 to reduce fallback usage
SHUTDOWN_TIMEOUT = 30

# Timeout settings
API_TIMEOUT = 30  # seconds
BATCH_TIMEOUT = 300  # 5 minutes per batch

# Retry settings
INITIAL_RETRY_DELAY = 1  # seconds

# Model configuration
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o"

# Token pricing
PRICING = {
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000}
}

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Subfield acronym mapping
SUBFIELD_ACRONYMS = {
    # Computer Science subfields
    "Artificial Intelligence & Machine Learning": "AI_ML",
    "Computer Vision & Pattern Recognition": "CV_PR",
    "Natural Language Processing": "NLP",
    "Robotics & Automation": "ROB_AUTO",
    "Data Science & Analytics": "DS_ANALYTICS",
    "Database Systems & Data Management": "DB_DM",
    "Computer Networks & Distributed Systems": "CN_DS",
    "Cybersecurity & Information Security": "CYBER_SEC",
    "Software Engineering & Development": "SW_ENG",
    "Human-Computer Interaction": "HCI",
    "Computer Graphics & Visualization": "CG_VIS",
    "Operating Systems & Systems Programming": "OS_SP",
    "Algorithms & Data Structures": "ALGO_DS",
    "Computational Theory & Complexity": "CT_COMPLEX",
    "Bioinformatics & Computational Biology": "BIOINFO",
    "Computer Architecture & Hardware": "CA_HW",
    "Mobile Computing & Ubiquitous Computing": "MOBILE_UBI",
    "Cloud Computing & Virtualization": "CLOUD_VIRT",
    "Internet of Things (IoT)": "IOT",
    "Quantum Computing": "QUANTUM",
    "Game Development & Interactive Media": "GAME_IM",
    "Computer Education & Pedagogy": "CS_EDU",
    "Digital Libraries & Information Retrieval": "DL_IR",
    "Parallel Computing & High Performance Computing": "PARALLEL_HPC",
    "Embedded Systems & Real-time Computing": "EMBEDDED_RT",
    
    # Information Systems subfields
    "Enterprise Systems & ERP": "ES_ERP",
    "Business Intelligence & Analytics": "BI_ANALYTICS",
    "E-commerce & Digital Business": "ECOMM_DB",
    "Knowledge Management": "KM",
    "Decision Support Systems": "DSS",
    "Information Systems Management": "ISM",
    "Digital Transformation": "DIGITAL_TRANS",
    "IT Governance & Strategy": "IT_GOV_STRAT",
    "Business Process Management": "BPM",
    "Social Media & Digital Marketing": "SM_DM",
    "Healthcare Information Systems": "HIS",
    "Educational Technology & E-learning": "ED_TECH_ELEARN",
    "Supply Chain Management Systems": "SCMS",
    "Customer Relationship Management (CRM)": "CRM",
    "Enterprise Architecture": "EA",
    "IT Service Management": "ITSM",
    "Digital Innovation & Entrepreneurship": "DIGITAL_INNOV",
    "Information Systems Security": "IS_SEC",
    "Data Governance & Privacy": "DG_PRIVACY",
    "Mobile Business Applications": "MBA",
    "Social Computing & Collaboration": "SOC_COMP",
    "IT Project Management": "IT_PM",
    "Digital Strategy & Business Models": "DIGITAL_STRAT",
    "Information Systems Research Methods": "IS_RM",
    "IT Ethics & Social Responsibility": "IT_ETHICS",
    
    # Information Technology subfields
    "IT Infrastructure & Operations": "IT_INFRA_OPS",
    "Network Administration & Management": "NET_ADMIN",
    "System Administration": "SYS_ADMIN",
    "IT Support & Help Desk": "IT_SUPPORT",
    "Web Development & Technologies": "WEB_DEV",
    "Mobile App Development": "MOBILE_APP",
    "DevOps & Continuous Integration": "DEVOPS_CI",
    "IT Security & Risk Management": "IT_SEC_RISK",
    "Data Center Management": "DC_MGMT",
    "Cloud Services & Management": "CLOUD_SVC",
    "IT Asset Management": "IT_ASSET",
    "IT Service Delivery": "IT_SVC_DEL",
    "Digital Forensics": "DIGITAL_FORENSICS",
    "IT Compliance & Auditing": "IT_COMPLIANCE",
    "Telecommunications & Networking": "TELECOM_NET",
    "IT Training & Education": "IT_TRAINING",
    "IT Consulting & Advisory": "IT_CONSULT",
    "Emerging Technologies Integration": "EMERGING_TECH",
    "IT Performance Monitoring": "IT_PERF_MON",
    "Disaster Recovery & Business Continuity": "DR_BC",
    "IT Procurement & Vendor Management": "IT_PROCVENDOR",
    "Digital Workplace Solutions": "DIGITAL_WORKPLACE",
    "IT Automation & Scripting": "IT_AUTO_SCRIPT",
    "IT Documentation & Knowledge Management": "IT_DOC_KM",
    "IT Standards & Best Practices": "IT_STANDARDS"
}

# Classification schema (same as original)
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

# Computing keywords for pre-filtering
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

@dataclass
class CircuitBreakerState:
    """Enhanced circuit breaker pattern for handling service failures"""
    failures: int = 0
    last_failure_time: Optional[float] = None
    is_open: bool = False
    success_count: int = 0
    total_requests: int = 0
    failure_threshold: int = 10  # Increased from 5
    success_threshold: int = 5   # Increased from 3
    timeout_seconds: int = 60
    half_open_timeout: int = 30
    
    def record_success(self):
        self.success_count += 1
        self.total_requests += 1
        if self.success_count >= self.success_threshold:
            self.reset()
    
    def record_failure(self, failure_type: str = "general"):
        self.failures += 1
        self.total_requests += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        # Adjust threshold based on failure type
        if failure_type == "timeout":
            self.failure_threshold = min(self.failure_threshold + 2, 15)  # More lenient for timeouts
        elif failure_type == "rate_limit":
            self.failure_threshold = max(self.failure_threshold - 1, 5)   # Less lenient for rate limits
        elif failure_type == "api_error":
            self.failure_threshold = min(self.failure_threshold + 1, 12)  # More lenient for API errors
        
        if self.failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def reset(self):
        if self.failures > 0:  # Only log if there were actual failures
            logger.info(f"Circuit breaker reset after {self.success_threshold} successes")
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False
        self.success_count = 0
        self.failure_threshold = 10  # Reset to default
    
    def can_attempt(self) -> bool:
        if not self.is_open:
            return True
        
        # Check if circuit should be half-open
        if self.last_failure_time and time.time() - self.last_failure_time > self.half_open_timeout:
            self.is_open = False  # Try half-open state
            logger.info("Circuit breaker entering half-open state")
            return True
        
        return False
    
    def get_failure_rate(self) -> float:
        """Get failure rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failures / self.total_requests) * 100


class EnhancedRateLimiter:
    """Enhanced token bucket rate limiter with better backoff and monitoring"""
    
    def __init__(self, max_requests_per_minute: int, max_tokens_per_minute: int = None):
        self.max_requests = max_requests_per_minute
        self.max_tokens = max_tokens_per_minute
        self.window_size = 60  # seconds
        self.requests = deque()
        self.tokens_used = deque()
        self.lock = asyncio.Lock()
        self.semaphore = Semaphore(MAX_CONCURRENT_CALLS)
        self.backoff_until = 0
        self.consecutive_rate_limits = 0
        self._last_cleanup = 0
        self._cached_wait_time = 0
        self._cache_valid_until = 0
    
    async def acquire(self, estimated_tokens: int = 0) -> bool:
        """Acquire permission with timeout and better backoff"""
        try:
            # Check if we're in backoff period
            if time.time() < self.backoff_until:
                wait_time = self.backoff_until - time.time()
                logger.info(f"Rate limiter in backoff, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
            
            # Try to acquire semaphore with timeout
            try:
                await wait_for(self.semaphore.acquire(), timeout=5.0)
            except TimeoutError:
                logger.warning("Semaphore acquisition timeout")
                return False
            
            try:
                async with self.lock:
                    now = time.time()
                    
                    # Clean old entries periodically (not on every call)
                    if now - self._last_cleanup > 5:  # Clean every 5 seconds
                        self._clean_old_entries(now)
                        self._last_cleanup = now
                    
                    # Check limits
                    if not self._check_limits(estimated_tokens):
                        # Calculate wait time (cached)
                        wait_time = self._get_cached_wait_time(now)
                        if wait_time > 60:  # Too long to wait
                            self.consecutive_rate_limits += 1
                            self.backoff_until = now + min(wait_time, 300)  # Max 5 min backoff
                            return False
                        
                        logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                        return await self.acquire(estimated_tokens)
                    
                    # Record request
                    self.requests.append(now)
                    if estimated_tokens > 0:
                        self.tokens_used.append((now, estimated_tokens))
                    
                    self.consecutive_rate_limits = 0
                    return True
                    
            finally:
                self.semaphore.release()
                
        except Exception as e:
            logger.error(f"Rate limiter error: {e}")
            return False
    
    def _clean_old_entries(self, now: float):
        """Remove old entries outside window"""
        cutoff = now - self.window_size
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        while self.tokens_used and self.tokens_used[0][0] < cutoff:
            self.tokens_used.popleft()
        # Invalidate cache
        self._cache_valid_until = 0
    
    def _check_limits(self, estimated_tokens: int) -> bool:
        """Check if request would exceed limits"""
        if len(self.requests) >= self.max_requests:
            return False
        
        if self.max_tokens:
            current_tokens = sum(tokens for _, tokens in self.tokens_used)
            if current_tokens + estimated_tokens > self.max_tokens:
                return False
        
        return True
    
    def _get_cached_wait_time(self, now: float) -> float:
        """Get cached wait time or calculate new one"""
        if now < self._cache_valid_until:
            return self._cached_wait_time
        
        # Calculate new wait time
        if self.requests:
            oldest = self.requests[0]
            wait_time = max(0, oldest + self.window_size - now + 0.1)
        else:
            wait_time = 1.0
        
        # Cache for 1 second
        self._cached_wait_time = wait_time
        self._cache_valid_until = now + 1.0
        
        return wait_time


class OptimizedClassificationPipeline:
    """Enhanced classification pipeline with robust error handling"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.rate_limiter = EnhancedRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_TOKENS)
        self.session = None
        self.connector = None
        
        # Circuit breakers for each model
        self.circuit_breakers = {
            PRIMARY_MODEL: CircuitBreakerState(),
            FALLBACK_MODEL: CircuitBreakerState()
        }
        
        # Statistics
        self.token_usage = defaultdict(lambda: defaultdict(int))
        self.model_usage = defaultdict(int)
        self.classification_stats = defaultdict(int)
        self.error_stats = defaultdict(int)
        
        # Checkpoint data
        self.processed_papers = []
        self.failed_papers = []
        self.low_confidence_papers = []
        self.checkpoint_file = "checkpoint.pkl"
        self.checkpoint_lock = asyncio.Lock()
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
        # Progress tracking
        self.start_time = time.time()
        self.last_progress_time = time.time()
        self.papers_since_last_progress = 0
    
    async def graceful_shutdown(self):
        """Handle graceful shutdown with proper cleanup"""
        logger.info("Initiating graceful shutdown...")
        self.shutdown_requested = True
        
        # Wait for any ongoing operations to complete
        try:
            await wait_for(self.shutdown_event.wait(), timeout=SHUTDOWN_TIMEOUT)
        except TimeoutError:
            logger.warning("Shutdown timeout, forcing cleanup")
        
        # Save final checkpoint
        try:
            await self.save_checkpoint_atomic()
            logger.info("Final checkpoint saved successfully")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        
        # Print final statistics
        if self.processed_papers:
            logger.info(f"Shutdown complete. Processed {len(self.processed_papers)} papers total.")
        else:
            logger.info("Shutdown complete. No papers processed.")
    
    def _handle_shutdown(self, signum, frame):  # noqa: ARG002
        """Handle graceful shutdown"""
        logger.info("Shutdown requested, saving checkpoint...")
        self.shutdown_requested = True
        self.shutdown_event.set()
    
    async def __aenter__(self):
        """Initialize async resources with robust settings"""
        timeout = aiohttp.ClientTimeout(
            total=API_TIMEOUT,
            connect=5,
            sock_read=API_TIMEOUT
        )
        self.connector = aiohttp.TCPConnector(
            limit=100,  # More conservative
            limit_per_host=50,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=True  # Force close to prevent hanging
        )
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        """Cleanup async resources"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
        # Give time for connections to close
        await asyncio.sleep(0.25)
    
    def _get_cache_key(self, title: str, abstract: str) -> str:
        """Generate cache key"""
        content = f"{title}:{abstract}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def is_computing_related(self, title: str, abstract: str) -> bool:
        """Enhanced computing detection"""
        text = (title + " " + abstract).lower()
        title_lower = title.lower()
        
        # Check for explicit computing indicators
        title_matches = sum(1 for kw in COMPUTING_KEYWORDS if kw in title_lower)
        abstract_matches = sum(1 for kw in COMPUTING_KEYWORDS if kw in text)
        
        # Weight title matches more heavily
        score = (title_matches * 3) + abstract_matches
        
        # Lower threshold for better coverage
        return score >= 3
    
    def get_subfield_acronym(self, subfield_name: str) -> str:
        """Convert full subfield name to acronym"""
        return SUBFIELD_ACRONYMS.get(subfield_name, subfield_name)
    
    def _build_classification_prompt(self, title: str, abstract: str, model: str) -> str:
        """Build optimized classification prompt"""
        return f"""You are an expert research paper classifier. Analyze the title and abstract below and classify the paper into the most appropriate computing discipline and subfield.

TITLE: {title}
ABSTRACT: {abstract}

Available disciplines and subfields:
- AI/ML: Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision, NLP, Robotics
- CS: Algorithms, Data Structures, Theory, Complexity, Cryptography, Formal Methods
- SE: Software Engineering, Testing, Debugging, Program Analysis, Software Architecture
- SYSTEMS: Operating Systems, Networks, Distributed Systems, Databases, Security
- HCI: Human-Computer Interaction, User Experience, Accessibility, Visualization
- NON_COMPUTING: Papers not related to computing (medicine, physics, etc.)

Instructions:
1. First determine if this is computing-related
2. If computing-related, identify the primary discipline and most specific subfield
3. Provide confidence score (0-100) based on how clearly the paper fits the classification
4. Give brief reasoning (1-2 sentences)

Respond in this exact format:
DISCIPLINE: [discipline]
SUBFIELD: [subfield] 
CONFIDENCE: [0-100]
REASONING: [brief explanation]

If the paper is clearly not computing-related, use DISCIPLINE: NON_COMPUTING, SUBFIELD: NON_COMPUTING."""
    
    def _build_fallback_prompt(self, title: str, abstract: str) -> str:
        """Build detailed fallback prompt for difficult cases"""
        # Show all subfields for fallback
        subfield_details = []
        for disc_name, disc_info in CLASSIFICATION_SCHEMA.items():
            subfield_list = [f"  - {sf}" for sf in disc_info['subfields']]
            subfield_details.append(f"\n{disc_info['code']} ({disc_name}):\n" + "\n".join(subfield_list))
        
        return f"""Carefully analyze this paper that requires expert classification.

Title: {title}
Abstract: {abstract[:2000]}

Available classifications:
{chr(10).join(subfield_details)}

Provide classification with:
Discipline: [exact code: CS/IS/IT/NON_COMPUTING]
Subfield: [exact subfield name from lists above]
Confidence: [0-100]
Reason: [explanation in 1-2 sentences]"""
    
    async def _make_api_call_with_timeout(self, prompt: str, model: str, max_tokens: int = 150) -> Optional[Dict]:
        """Make API call with timeout and circuit breaker"""
        circuit_breaker = self.circuit_breakers[model]
        
        # Check circuit breaker
        if not circuit_breaker.can_attempt():
            logger.warning(f"Circuit breaker open for {model}")
            return None
        
        try:
            # Estimate tokens
            estimated_tokens = len(prompt) // 4 + max_tokens
            
            # Acquire rate limit
            if not await self.rate_limiter.acquire(estimated_tokens):
                logger.warning("Rate limiter denied request")
                return None
            
            # Make API call with timeout
            result = await wait_for(
                self._make_api_call(prompt, model, max_tokens),
                timeout=API_TIMEOUT
            )
            
            if result:
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()
            
            return result
            
        except TimeoutError:
            logger.error(f"API call timeout for {model}")
            circuit_breaker.record_failure(failure_type="timeout")
            self.error_stats['timeout'] += 1
            return None
        except Exception as e:
            logger.error(f"API call error for {model}: {e}")
            circuit_breaker.record_failure(failure_type="api_error")
            self.error_stats['api_error'] += 1
            return None
    
    async def _make_api_call(self, prompt: str, model: str, max_tokens: int = 150) -> Optional[Dict]:
        """Make actual API call with retries"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "presence_penalty": 0,
            "frequency_penalty": 0
        }
        
        for retry in range(MAX_RETRIES):
            try:
                async with self.session.post(
                    OPENAI_API_URL,
                    json=payload,
                    headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Update token usage
                        if 'usage' in data:
                            usage = data['usage']
                            self.token_usage[model]['input'] += usage.get('prompt_tokens', 0)
                            self.token_usage[model]['output'] += usage.get('completion_tokens', 0)
                        
                        return data
                    
                    elif response.status == 429:
                        # Rate limit - exponential backoff
                        retry_after = int(response.headers.get('Retry-After', 2 ** retry))
                        wait_time = min(retry_after, MAX_RETRY_DELAY)
                        logger.warning(f"Rate limit 429, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        if retry == MAX_RETRIES - 1:  # Last retry
                            circuit_breaker.record_failure(failure_type="rate_limit")
                        continue
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text[:200]}")
                        if response.status >= 500:  # Server error, retry
                            await asyncio.sleep(2 ** retry)
                            continue
                        return None
                        
            except aiohttp.ClientError as e:
                logger.error(f"Network error: {e}")
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** retry)
                    continue
                return None
            except Exception as e:
                logger.error(f"Unexpected error in API call: {e}")
                return None
        
        return None
    
    def _parse_response(self, content: str) -> Dict:
        """Parse classification response with validation"""
        result = {
            'discipline': 'UNKNOWN',
            'subfield': 'UNKNOWN',
            'confidence': 0,
            'reasoning': ''
        }
        
        try:
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    key_lower = key.lower().strip()
                    value = value.strip()
                    
                    if 'discipline' in key_lower:
                        if value.upper() in ['CS', 'IS', 'IT', 'NON_COMPUTING']:
                            result['discipline'] = value.upper()
                    
                    elif 'subfield' in key_lower:
                        # Validate and convert to acronym
                        for disc_info in CLASSIFICATION_SCHEMA.values():
                            if value in disc_info['subfields']:
                                result['subfield'] = self.get_subfield_acronym(value)
                                break
                        if result['subfield'] == 'UNKNOWN' and value in ['NOT_APPLICABLE', 'N/A']:
                            result['subfield'] = 'NOT_APPLICABLE'
                    
                    elif 'confidence' in key_lower:
                        try:
                            conf_str = value.replace('%', '').strip()
                            conf_val = int(float(conf_str))
                            result['confidence'] = max(0, min(100, conf_val))
                        except:
                            result['confidence'] = 50
                    
                    elif 'reason' in key_lower:
                        result['reasoning'] = value[:200]  # Limit length
        
        except Exception as e:
            logger.error(f"Parse error: {e}")
        
        return result
    
    async def classify_paper(self, paper: Dict) -> Dict:
        """Classify a single paper with comprehensive error handling and recovery"""
        title = paper.get('Title', '').strip()
        abstract = paper.get('Abstract', '').strip()
        
        # Validate input
        if not title or not abstract or len(abstract.split()) < 20:
            return {
                **paper,
                'Discipline': 'SKIPPED',
                'Subfield': 'INSUFFICIENT_CONTENT',
                'Confidence_Score': 100,
                'Model_Used': 'validation',
                'Reasoning': 'Insufficient content',
                'Processing_Time': 0,
                'Error_Type': None
            }
        
        start_time = time.time()
        error_type = None
        
        try:
            # Pre-filter check
            if not self.is_computing_related(title, abstract):
                # Still give it a chance with primary model
                pass
            
            # Try primary model
            primary_prompt = self._build_classification_prompt(title, abstract, PRIMARY_MODEL)
            primary_response = await self._make_api_call_with_timeout(
                primary_prompt, PRIMARY_MODEL, max_tokens=100
            )
            
            discipline = 'UNKNOWN'
            subfield = 'UNKNOWN'
            confidence = 0
            reasoning = ''
            model_used = PRIMARY_MODEL
            
            if primary_response and 'choices' in primary_response:
                try:
                    choices = primary_response['choices']
                    if choices and len(choices) > 0:
                        message = choices[0].get('message', {})
                        content = message.get('content', '')
                        if content:
                            parsed = self._parse_response(content)
                            
                            if parsed['discipline'] != 'UNKNOWN' and parsed['subfield'] != 'UNKNOWN':
                                discipline = parsed['discipline']
                                subfield = parsed['subfield']
                                confidence = parsed['confidence']
                                reasoning = parsed['reasoning']
                except Exception as e:
                    logger.error(f"Error parsing primary response: {e}")
                    error_type = 'parse_error'
            
            # Use fallback if needed (but only for computing papers)
            if (discipline == 'UNKNOWN' or subfield == 'UNKNOWN' or confidence < CONFIDENCE_THRESHOLD) and discipline != 'NON_COMPUTING':
                logger.info(f"Using fallback for: {title[:50]}")
                fallback_prompt = self._build_fallback_prompt(title, abstract)
                fallback_response = await self._make_api_call_with_timeout(
                    fallback_prompt, FALLBACK_MODEL, max_tokens=150
                )
                
                if fallback_response and 'choices' in fallback_response:
                    try:
                        choices = fallback_response['choices']
                        if choices and len(choices) > 0:
                            message = choices[0].get('message', {})
                            content = message.get('content', '')
                            if content:
                                parsed = self._parse_response(content)
                                
                                if parsed['discipline'] != 'UNKNOWN':
                                    discipline = parsed['discipline']
                                    subfield = parsed['subfield']
                                    confidence = parsed['confidence']
                                    reasoning = parsed['reasoning']
                                    model_used = FALLBACK_MODEL
                    except Exception as e:
                        logger.error(f"Error parsing fallback response: {e}")
                        error_type = 'parse_error'
            
            # Ensure we always have valid values
            if discipline == 'UNKNOWN':
                discipline = 'UNABLE_TO_CLASSIFY'
            if subfield == 'UNKNOWN':
                subfield = 'UNABLE_TO_CLASSIFY'
            
            # Track statistics
            self.model_usage[model_used] += 1
            self.classification_stats[discipline] += 1
            
            processing_time = time.time() - start_time
            
            result = {
                **paper,
                'Discipline': discipline,
                'Subfield': subfield,
                'Confidence_Score': confidence,
                'Model_Used': model_used,
                'Reasoning': reasoning,
                'Processing_Time': round(processing_time, 2),
                'Error_Type': error_type
            }
            
            # Track low confidence
            if confidence < CONFIDENCE_THRESHOLD and discipline != 'NON_COMPUTING':
                self.low_confidence_papers.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error classifying paper: {e}")
            processing_time = time.time() - start_time
            return {
                **paper,
                'Discipline': 'ERROR',
                'Subfield': 'EXCEPTION',
                'Confidence_Score': 0,
                'Model_Used': 'none',
                'Reasoning': f'Unexpected error: {str(e)[:100]}',
                'Processing_Time': round(processing_time, 2),
                'Error_Type': 'unexpected_error'
            }
    
    async def process_batch_with_timeout(self, papers: List[Dict]) -> List[Dict]:
        """Process batch with timeout and error recovery"""
        try:
            # Process with timeout
            results = await wait_for(
                self.process_batch(papers),
                timeout=BATCH_TIMEOUT
            )
            return results
        except TimeoutError:
            logger.error(f"Batch timeout after {BATCH_TIMEOUT}s")
            # Return partial results with errors for unprocessed papers
            results = []
            for paper in papers:
                results.append({
                    **paper,
                    'Discipline': 'ERROR',
                    'Subfield': 'BATCH_TIMEOUT',
                    'Confidence_Score': 0,
                    'Model_Used': 'none',
                    'Reasoning': 'Batch processing timeout',
                    'Processing_Time': 0
                })
            return results
    
    async def process_batch(self, papers: List[Dict]) -> List[Dict]:
        """Process a batch of papers with error handling"""
        tasks = []
        for paper in papers:
            if self.shutdown_requested:
                break
            task = create_task(self.classify_paper(paper))
            tasks.append(task)
        
        # Wait for all tasks with proper error handling
        results = []
        completed_tasks = await gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
                error_result = {
                    **papers[i],
                    'Discipline': 'ERROR',
                    'Subfield': 'EXCEPTION',
                    'Confidence_Score': 0,
                    'Model_Used': 'none',
                    'Reasoning': str(result)[:200],
                    'Processing_Time': 0
                }
                self.failed_papers.append(error_result)
                results.append(error_result)
            else:
                results.append(result)
        
        return results
    
    async def save_checkpoint_atomic(self):
        """Save checkpoint atomically to prevent corruption"""
        async with self.checkpoint_lock:
            checkpoint_data = {
                'processed_papers': self.processed_papers,
                'failed_papers': self.failed_papers,
                'low_confidence_papers': self.low_confidence_papers,
                'classification_stats': dict(self.classification_stats),
                'token_usage': dict(self.token_usage),
                'model_usage': dict(self.model_usage),
                'error_stats': dict(self.error_stats),
                'timestamp': datetime.now().isoformat()
            }
            
            # Write to temporary file first
            temp_file = self.checkpoint_file + '.tmp'
            try:
                with open(temp_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                # Atomic rename
                shutil.move(temp_file, self.checkpoint_file)
                logger.info(f"Checkpoint saved: {len(self.processed_papers)} papers")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def load_checkpoint(self):
        """Load checkpoint with validation"""
        checkpoint_file = Path(self.checkpoint_file)
        if not checkpoint_file.exists():
            logger.info("No checkpoint found, starting fresh")
            return
        
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.processed_papers = checkpoint_data.get('processed_papers', [])
            self.model_usage = checkpoint_data.get('model_usage', defaultdict(int))
            self.classification_stats = checkpoint_data.get('classification_stats', defaultdict(int))
            self.token_usage = checkpoint_data.get('token_usage', defaultdict(lambda: {'input': 0, 'output': 0}))
            
            logger.info(f"Loaded checkpoint with {len(self.processed_papers)} papers")
            
            # Validate and clean checkpoint data
            if not self.validate_checkpoint():
                logger.warning("Checkpoint validation failed, starting fresh")
                self.processed_papers = []
                self.model_usage = defaultdict(int)
                self.classification_stats = defaultdict(int)
                self.token_usage = defaultdict(lambda: {'input': 0, 'output': 0})
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.info("Starting fresh due to checkpoint error")
            self.processed_papers = []
            self.model_usage = defaultdict(int)
            self.classification_stats = defaultdict(int)
            self.token_usage = defaultdict(lambda: {'input': 0, 'output': 0})
    
    def validate_checkpoint(self) -> bool:
        """Validate checkpoint data and clean if necessary"""
        if not self.processed_papers:
            return True
        
        logger.info("Validating checkpoint data...")
        original_count = len(self.processed_papers)
        
        # Remove papers with invalid discipline/subfield combinations
        valid_papers = []
        invalid_count = 0
        
        for paper in self.processed_papers:
            discipline = paper.get('Discipline', '')
            subfield = paper.get('Subfield', '')
            
            # Check for valid combinations
            if discipline in ['UNKNOWN', 'ERROR', 'UNABLE_TO_CLASSIFY']:
                if subfield not in ['UNKNOWN', 'ERROR', 'UNABLE_TO_CLASSIFY', 'INSUFFICIENT_CONTENT']:
                    invalid_count += 1
                    continue
            
            if discipline == 'NON_COMPUTING' and subfield != 'NON_COMPUTING':
                invalid_count += 1
                continue
            
            # Check for required fields
            if not all(key in paper for key in ['Title', 'Abstract', 'Discipline', 'Subfield', 'Confidence_Score']):
                invalid_count += 1
                continue
            
            valid_papers.append(paper)
        
        if invalid_count > 0:
            logger.warning(f"Removed {invalid_count} invalid papers from checkpoint")
            self.processed_papers = valid_papers
        
        # Rebuild statistics
        self._rebuild_statistics()
        
        logger.info(f"Checkpoint validation complete: {len(self.processed_papers)}/{original_count} papers valid")
        return len(self.processed_papers) > 0
    
    def _rebuild_statistics(self):
        """Rebuild statistics from processed papers"""
        self.model_usage = defaultdict(int)
        self.classification_stats = defaultdict(int)
        self.token_usage = defaultdict(lambda: {'input': 0, 'output': 0})
        self.low_confidence_papers = []
        self.failed_papers = []
        
        for paper in self.processed_papers:
            model = paper.get('Model_Used', 'unknown')
            discipline = paper.get('Discipline', 'unknown')
            confidence = paper.get('Confidence_Score', 0)
            
            self.model_usage[model] += 1
            self.classification_stats[discipline] += 1
            
            if confidence < CONFIDENCE_THRESHOLD and discipline != 'NON_COMPUTING':
                self.low_confidence_papers.append(paper)
            
            if discipline in ['ERROR', 'UNABLE_TO_CLASSIFY']:
                self.failed_papers.append(paper)
    
    def get_processed_ids(self) -> Set[str]:
        """Get set of processed paper IDs"""
        processed_ids = set()
        for paper in self.processed_papers:
            paper_id = self._get_cache_key(paper.get('Title', ''), paper.get('Abstract', ''))
            processed_ids.add(paper_id)
        return processed_ids
    
    def update_progress(self, papers_processed: int, total_papers: int):
        """Update and display progress"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate rates
        overall_rate = papers_processed / elapsed * 60 if elapsed > 0 else 0
        
        # Recent rate (last interval)
        interval_time = current_time - self.last_progress_time
        if interval_time > 0:
            recent_rate = self.papers_since_last_progress / interval_time * 60
        else:
            recent_rate = overall_rate
        
        # ETA
        remaining = total_papers - papers_processed
        eta_seconds = remaining / recent_rate * 60 if recent_rate > 0 else 0
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # Cost
        cost = self.get_total_cost()
        
        # Error rate
        error_rate = len(self.failed_papers) / papers_processed * 100 if papers_processed > 0 else 0
        
        logger.info(
            f"Progress: {papers_processed}/{total_papers} ({papers_processed/total_papers*100:.1f}%) | "
            f"Rate: {recent_rate:.0f}/min | ETA: {eta_str} | Cost: ${cost:.2f} | "
            f"Errors: {error_rate:.1f}%"
        )
        
        # Reset interval tracking
        self.last_progress_time = current_time
        self.papers_since_last_progress = 0
    
    async def process_csv(self, input_file: str, output_file: str, resume: bool = False):
        """Process CSV file with enhanced reliability and streaming"""
        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()
        
        processed_ids = self.get_processed_ids()
        
        # Count total papers first (for progress tracking)
        total_papers = 0
        papers_to_process_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_papers += 1
                paper_id = self._get_cache_key(row.get('Title', ''), row.get('Abstract', ''))
                if paper_id not in processed_ids:
                    papers_to_process_count += 1
        
        logger.info(f"Total papers in file: {total_papers}")
        logger.info(f"Papers to process: {papers_to_process_count} (already processed: {len(processed_ids)})")
        
        if papers_to_process_count == 0:
            logger.info("No new papers to process")
            self.save_results(output_file)
            return
        
        # Reset timing
        self.start_time = time.time()
        self.last_progress_time = time.time()
        
        # Process in streaming batches
        batch_count = 0
        papers_processed = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            current_batch = []
            
            for row in reader:
                if self.shutdown_requested:
                    logger.info("Shutdown requested, saving progress...")
                    break
                
                paper_id = self._get_cache_key(row.get('Title', ''), row.get('Abstract', ''))
                if paper_id not in processed_ids:
                    current_batch.append(row)
                    
                    # Process batch when full
                    if len(current_batch) >= BATCH_SIZE:
                        batch_count += 1
                        total_batches = (papers_to_process_count + BATCH_SIZE - 1) // BATCH_SIZE
                        
                        logger.info(f"Processing batch {batch_count}/{total_batches}")
                        
                        # Process with timeout
                        results = await self.process_batch_with_timeout(current_batch)
                        self.processed_papers.extend(results)
                        self.papers_since_last_progress += len(results)
                        papers_processed += len(results)
                        
                        # Update progress
                        self.update_progress(papers_processed, papers_to_process_count)
                        
                        # Save checkpoint periodically
                        if papers_processed % CHECKPOINT_INTERVAL == 0:
                            await self.save_checkpoint_atomic()
                        
                        # Clear batch and brief pause
                        current_batch = []
                        if not self.shutdown_requested:
                            await asyncio.sleep(0.5)
            
            # Process remaining papers in final batch
            if current_batch and not self.shutdown_requested:
                batch_count += 1
                logger.info(f"Processing final batch {batch_count}")
                
                results = await self.process_batch_with_timeout(current_batch)
                self.processed_papers.extend(results)
                papers_processed += len(results)
                
                self.update_progress(papers_processed, papers_to_process_count)
        
        # Final save
        await self.save_checkpoint_atomic()
        self.save_results(output_file)
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
    
    def save_results(self, output_file: str):
        """Save all results with proper formatting"""
        # Ensure all papers have consistent fields
        if self.processed_papers:
            # Save main results
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.processed_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.processed_papers)
            logger.info(f"Results saved to {output_file}")
        
        # Save low confidence papers
        if self.low_confidence_papers:
            review_file = output_file.replace('.csv', '_review.csv')
            with open(review_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.low_confidence_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.low_confidence_papers)
            logger.info(f"Review papers saved to {review_file}")
        
        # Save failed papers
        if self.failed_papers:
            error_file = output_file.replace('.csv', '_errors.csv')
            with open(error_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = list(self.failed_papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.failed_papers)
            logger.info(f"Error papers saved to {error_file}")
        
        # Save statistics
        self.save_statistics(output_file)
    
    def save_statistics(self, output_file: str):
        """Save detailed statistics"""
        # Calculate subfield distribution
        subfield_dist = defaultdict(int)
        discipline_subfield_dist = defaultdict(lambda: defaultdict(int))
        
        for paper in self.processed_papers:
            discipline = paper.get('Discipline', 'UNKNOWN')
            subfield = paper.get('Subfield', 'UNKNOWN')
            subfield_dist[subfield] += 1
            discipline_subfield_dist[discipline][subfield] += 1
        
        stats = {
            'summary': {
                'total_processed': len(self.processed_papers),
                'total_failed': len(self.failed_papers),
                'total_low_confidence': len(self.low_confidence_papers),
                'error_rate': len(self.failed_papers) / len(self.processed_papers) * 100 if self.processed_papers else 0,
                'total_cost': round(self.get_total_cost(), 2),
                'processing_time': round(time.time() - self.start_time, 2)
            },
            'model_usage': dict(self.model_usage),
            'token_usage': {
                model: {
                    'input': usage['input'],
                    'output': usage['output'],
                    'total': usage['input'] + usage['output']
                }
                for model, usage in self.token_usage.items()
            },
            'classification_distribution': dict(self.classification_stats),
            'subfield_distribution': dict(subfield_dist),
            'discipline_subfield_distribution': {
                disc: dict(subfields)
                for disc, subfields in discipline_subfield_dist.items()
            },
            'error_statistics': dict(self.error_stats),
            'quality_metrics': {
                'average_confidence': self._calculate_avg_confidence(),
                'high_confidence_rate': self._calculate_high_confidence_rate(),
                'unknown_rate': self.classification_stats.get('UNKNOWN', 0) / len(self.processed_papers) * 100 if self.processed_papers else 0
            }
        }
        
        stats_file = output_file.replace('.csv', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence score"""
        total_conf = sum(
            paper.get('Confidence_Score', 0)
            for paper in self.processed_papers
            if paper.get('Confidence_Score', 0) > 0
        )
        count = sum(
            1 for paper in self.processed_papers
            if paper.get('Confidence_Score', 0) > 0
        )
        return round(total_conf / count, 1) if count > 0 else 0
    
    def _calculate_high_confidence_rate(self) -> float:
        """Calculate rate of high confidence classifications"""
        high_conf = sum(
            1 for paper in self.processed_papers
            if paper.get('Confidence_Score', 0) >= CONFIDENCE_THRESHOLD
        )
        total = len(self.processed_papers)
        return round(high_conf / total * 100, 1) if total > 0 else 0
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        print("\n" + "="*70)
        print("CLASSIFICATION COMPLETE - SUMMARY STATISTICS")
        print("="*70)
        
        total = len(self.processed_papers)
        print(f"Total papers processed: {total:,}")
        print(f"Failed classifications: {len(self.failed_papers):,} ({len(self.failed_papers)/total*100:.1f}%)")
        print(f"Low confidence papers: {len(self.low_confidence_papers):,} ({len(self.low_confidence_papers)/total*100:.1f}%)")
        print(f"Processing time: {time.time() - self.start_time:.1f} seconds")
        print(f"Total cost: ${self.get_total_cost():.2f}")
        
        print("\nModel Usage:")
        for model, count in sorted(self.model_usage.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            tokens = self.token_usage[model]['input'] + self.token_usage[model]['output']
            print(f"  {model}: {count:,} papers ({percentage:.1f}%), {tokens:,} tokens")
        
        print("\nDiscipline Distribution:")
        for disc, count in sorted(self.classification_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {disc}: {count:,} ({percentage:.1f}%)")
        
        print("\nQuality Metrics:")
        print(f"  Average confidence: {self._calculate_avg_confidence():.1f}%")
        print(f"  High confidence rate: {self._calculate_high_confidence_rate():.1f}%")
        
        if self.error_stats:
            print("\nError Types:")
            for error_type, count in self.error_stats.items():
                print(f"  {error_type}: {count}")
        
        # Circuit breaker statistics
        print("\nCircuit Breaker Status:")
        for model, cb in self.circuit_breakers.items():
            status = "OPEN" if cb.is_open else "CLOSED"
            failure_rate = cb.get_failure_rate()
            print(f"  {model}: {status}, {failure_rate:.1f}% failure rate ({cb.failures}/{cb.total_requests})")
        
        print("="*70)


async def main():
    """Main entry point with enhanced error handling"""
    parser = argparse.ArgumentParser(
        description='Classify research paper abstracts with robust error handling'
    )
    parser.add_argument('--input', default='Abstracts.csv', help='Input CSV file')
    parser.add_argument('--output', default='classified_papers.csv', help='Output CSV file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    args = parser.parse_args()
    
    # Validate API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")
        sys.exit(1)
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run classification
    pipeline = None
    try:
        async with OptimizedClassificationPipeline(api_key) as pipeline:
            await pipeline.process_csv(args.input, args.output, args.resume)
    except KeyboardInterrupt:
        print("\nGraceful shutdown initiated...")
        if pipeline:
            await pipeline.graceful_shutdown()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        if pipeline:
            await pipeline.graceful_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    # Set up asyncio for Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())