"""MIT License

Copyright (c) 2025 ji.dong@hotmail.co.uk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

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


def create_openai_client(api_key: Optional[str] = None,
                         model: Optional[str] = None) -> Tuple["OpenAI", str]:  # type: ignore[name-defined]
    """Initialize OpenAI client and return (client, model_name).

    Precedence:
      1. Explicit function arguments (api_key, model) if provided
      2. Environment variables OPENAI_API_KEY, OPENAI_MODEL

    Raises RuntimeError if required values are missing.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Add 'openai' to requirements.txt")
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if model is None:
        model = os.getenv('OPENAI_MODEL')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not provided (param or env)")
    if not model:
        raise RuntimeError("OPENAI_MODEL not provided (param or env)")
    client = OpenAI(api_key=api_key)
    return client, model  # type: ignore


def openai_review(
        client,
        model: str,
        prompt: str,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        jitter_cap: float = 0.25,
        temperature: float = 0.2,
        max_tokens: int = 900,
        system_prompt: str = "You are a precise senior code reviewer.",
) -> str:
    """Execute a chat completion with retry, exponential backoff, jitter.

    Parameters:
      client: OpenAI client instance
      model: model/deployment name
      prompt: user prompt text
      max_retries: maximum retry attempts for transient errors (default 5)
      base_delay: base delay (seconds) for exponential backoff (default 1.0)
      jitter_cap: maximum random jitter added to delay (seconds) (default 0.25)
      temperature: sampling temperature (default 0.2)
      max_tokens: max tokens for the completion (default 900)
      system_prompt: system role content
    """
    def should_retry(status_code: Optional[int]) -> bool:
        if status_code is None:
            return True
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
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
                "OpenAI request failed (attempt %d/%d, status=%s): %s; retrying in %.2fs",
                attempt, max_retries, status_code, e, delay
            )
            time.sleep(delay)
