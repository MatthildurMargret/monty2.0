"""
Groq API wrapper with rate limiting and retry logic
"""

import os
import time
import logging
import random
import threading
from dotenv import load_dotenv
from groq import Groq  # <-- Correct import

# Load environment variables from .env file
load_dotenv()

free_models = ["gemma2-9b-it", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b", "mistral-saba-24b"]

# Module logger
logger = logging.getLogger("groq_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Default to INFO unless LOG_LEVEL explicitly set
    try:
        logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))
    except Exception:
        logger.setLevel(logging.INFO)

# ---- Global Groq rate limiting & retry config ----
GROQ_RPM = int(os.getenv("GROQ_RPM", "8"))  # Groq requests per minute
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "2"))  # Keep low to avoid nested retries w/ SDK
GROQ_BASE_BACKOFF = float(os.getenv("GROQ_BASE_BACKOFF", "1.5"))  # seconds

_groq_last_ts = 0.0
_groq_lock = threading.Lock()
_groq_min_interval = 60.0 / max(GROQ_RPM, 1)

def _groq_wait_for_rate_limit():
    """Global limiter to respect GROQ_RPM across all calls."""
    global _groq_last_ts
    with _groq_lock:
        now = time.monotonic()
        elapsed = now - _groq_last_ts
        if elapsed < _groq_min_interval:
            time.sleep(_groq_min_interval - elapsed)
        _groq_last_ts = time.monotonic()

# Reuse a single Groq client
_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # <-- Correct initialization

def get_groq_response(prompt, model=free_models[0]):
    """Get a response from the Groq API."""
    # Global rate limit before each call
    _groq_wait_for_rate_limit()
    
    # Try all available models if needed
    available_models = free_models.copy()
    current_model = model
    max_retries = GROQ_MAX_RETRIES
    retry_delay = GROQ_BASE_BACKOFF  # seconds
    
    for attempt in range(max_retries):
        try:
            chat_completion = _groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=current_model,
                timeout=60,  # Increased timeout to reduce SDK retries on slow responses
            )
            # If successful, return the response
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            error_type = str(type(e).__name__)
            error_msg = str(e)
            logger.warning("Groq error (%s) with model %s: %s", error_type, current_model, error_msg)
            
            # Handle rate limiting errors with exponential backoff
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                wait_time = retry_delay * (2 ** attempt) + 5 + random.random() * 0.5
                logger.info("Rate limited by Groq. Backing off %.2fs (attempt %d/%d)", wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue

            # Connection/timeouts and 5xx often surface as generic errors; apply backoff
            if any(k in error_msg.lower() for k in ["timeout", "temporarily unavailable", "500", "502", "503", "504"]):
                wait_time = retry_delay * (2 ** attempt) + random.random() * 0.5
                logger.info("Transient Groq error. Backing off %.2fs (attempt %d/%d)", wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue

            # Try a different model if available
            if current_model in available_models:
                available_models.remove(current_model)
            
            if available_models:
                current_model = available_models[0]
                logger.info("Switching Groq model to: %s", current_model)
            else:
                # If we've tried all models, wait and retry the first one
                logger.info("All Groq models attempted. Retrying original model after delay...")
                available_models = free_models.copy()
                current_model = model
                time.sleep(retry_delay * (2 ** attempt) + random.random() * 0.5)
    
    # If all retries and models fail, return a fallback message
    logger.error("Groq request failed after retries for prompt prefix: %s", prompt[:80])
    return "Unable to process."
