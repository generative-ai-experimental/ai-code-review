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
from typing import Optional
from openai import AzureOpenAI

logger = logging.getLogger("ai_code_review")

# Module summary:
# Lightweight helpers to interact with Azure/OpenAI chat completions for code review.
# Provides:
#   - create_openai_client: constructs a client with precedence (explicit args > env vars).
#   - openai_review: performs a resilient chat completion (retry + exponential backoff + jitter).
# Design notes:
#   * We intentionally keep dependencies minimal.
#   * Keyword-only tuning parameters avoid accidental positional misuse.
#   * This module can be extended to return richer metadata (token counts, latency) if needed.

def create_openai_client(
    api_key: Optional[str] = None,
    api_version: str = "2024-12-01-preview",
    azure_endpoint: Optional[str] = None,
) -> AzureOpenAI:
    """Create and return an AzureOpenAI client.

    Parameter precedence (explicit over environment):
      api_key        -> OPENAI_API_KEY
      azure_endpoint -> AZURE_OPENAI_ENDPOINT

    Args:
      api_key: Azure OpenAI resource key. If None, falls back to env.
      api_version: API version string; default pinned for stability.
      azure_endpoint: Full endpoint (e.g. https://myres.openai.azure.com/). If None, env is used.

    Raises:
      RuntimeError: if required values are missing after applying precedence.
    """
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
    if azure_endpoint is None:
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not provided (param or env)")
    if not azure_endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT not provided (param or env)")

    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )


def openai_review(
    client,
    prompt: str,
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    jitter_cap: float = 0.25,
    temperature: float = 0.2,
    max_tokens: int = 900,
    model: Optional[str] = "gpt-5-mini",
    system_prompt: str = "You are a precise senior code reviewer.",
) -> str:
    """Perform a chat completion with resilient retry strategy.

    Strategy:
      * Retries on common transient / rate / service errors (HTTP 408/409/429/5xx).
      * Exponential backoff: delay = base_delay * 2^(attempt-1) + jitter.
      * Jitter (uniform 0..jitter_cap) reduces thundering herd contention.
      * Gives up after max_retries attempts and raises RuntimeError.

        Args:
            client: OpenAI (Azure) client.
            prompt: User content to review.
            max_retries: Maximum attempts before failing.
            base_delay: Base seconds for exponential schedule.
            jitter_cap: Max random seconds added to each backoff.
            temperature: Sampling temperature.
            max_tokens: Desired upper bound on completion length. Some newer models
                have renamed this parameter to 'max_completion_tokens'. This function
                automatically retries once with the alternate parameter name if the
                API returns a 400 indicating 'max_tokens' is unsupported.
            model: Deployment/model identifier (defaults to 'gpt-5-mini').
            system_prompt: System role instructions controlling reviewer persona.

    Returns:
      Assistant reply text (stripped).

    Raises:
      RuntimeError: Non-retriable error or exhausted retries.
    """
    def should_retry(status_code: Optional[int]) -> bool:
        if status_code is None:  # conservative: treat unknown as retryable
            return True
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    # Start assuming legacy/stable 'max_tokens' parameter; adapt if API rejects it.
    token_param_name = "max_tokens"

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
            }
            kwargs[token_param_name] = max_tokens

            try:
                completion = client.chat.completions.create(**kwargs)
            except Exception as inner_e:  # handle possible param rename issue
                msg = str(inner_e)
                # Detect unsupported param error suggesting alternate name
                if ("Unsupported parameter" in msg and "max_tokens" in msg and
                        "max_completion_tokens" in msg and token_param_name == "max_tokens"):
                    # Switch parameter name and retry once immediately (not counting as a network retry attempt)
                    token_param_name = "max_completion_tokens"
                    kwargs.pop("max_tokens", None)
                    kwargs[token_param_name] = max_tokens
                    completion = client.chat.completions.create(**kwargs)
                else:
                    raise  # re-raise if unrelated
            if not completion.choices:
                raise RuntimeError("Empty completion choices returned from API")
            content = completion.choices[0].message.content or ""
            return content.strip()
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
            # Cap exponential growth to avoid excessive waits (e.g. 30s)
            delay = min(delay, 30.0)
            delay += random.uniform(0, jitter_cap)
            logger.warning(
                "OpenAI request failed (attempt %d/%d, status=%s): %s; retrying in %.2fs",
                attempt, max_retries, status_code, e, delay
            )
            time.sleep(delay)
