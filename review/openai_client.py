import os
import time
import random
import logging
from typing import Optional, Tuple

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

logger = logging.getLogger("ai_code_review")


def create_openai_client() -> Tuple["OpenAI", str]:  # type: ignore[name-defined]
    """Initialize OpenAI client and return (client, model_name).

    Environment variables:
      OPENAI_API_KEY (required)
      OPENAI_MODEL   (required)
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add 'openai' to requirements.txt")
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    if not model:
        raise RuntimeError("OPENAI_MODEL not set")
    client = OpenAI(api_key=api_key)
    return client, model  # type: ignore


def openai_review(client, model: str, prompt: str) -> str:
    """Execute a chat completion with retry, exponential backoff, fallback model, jitter.

    Optional environment variables:
      OPENAI_MAX_RETRIES
      OPENAI_BASE_DELAY
      OPENAI_FALLBACK_MODEL
      OPENAI_RETRY_JITTER
    """
    max_retries = int(os.getenv('OPENAI_MAX_RETRIES', '5'))
    base_delay = float(os.getenv('OPENAI_BASE_DELAY', '1.0'))
    fallback_model = os.getenv('OPENAI_FALLBACK_MODEL')
    jitter_cap = float(os.getenv('OPENAI_RETRY_JITTER', '0.25'))
    using_fallback = False

    def should_retry(status_code: Optional[int]) -> bool:
        if status_code is None:
            return True
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise senior code reviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=900
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover
            status_code = getattr(e, 'status_code', None)
            retry_after = None
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                try:
                    status_code = getattr(e.response, 'status_code', status_code)
                    rh = getattr(e.response, 'headers', {}) or {}
                    retry_after = rh.get('Retry-After') or rh.get('retry-after')
                except Exception:  # pragma: no cover
                    pass

            if not should_retry(status_code):
                raise RuntimeError(f"Non-retriable OpenAI error (status={status_code}): {e}")

            if attempt == (max_retries // 2) and fallback_model and not using_fallback:
                logger.warning("Switching to fallback model '%s' after %d failed attempts of '%s'", fallback_model, attempt, model)
                model = fallback_model  # type: ignore
                using_fallback = True

            if attempt == max_retries:
                raise RuntimeError(f"OpenAI chat completion failed after {max_retries} retries: {e}")

            if retry_after:
                try:
                    delay = min(float(retry_after), 60.0)
                except ValueError:
                    delay = base_delay * (2 ** (attempt - 1))
            else:
                delay = base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0, jitter_cap)
            logger.warning(
                "OpenAI request failed (attempt %d/%d, status=%s, fallback=%s): %s; retrying in %.2fs",
                attempt, max_retries, status_code, using_fallback, e, delay
            )
            time.sleep(delay)
