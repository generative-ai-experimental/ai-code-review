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

Azure DevOps API helper module.

Core responsibilities:
 - Environment validation / configuration construction
 - Auth header construction
 - Pull Request metadata retrieval
 - Changed file listing (commit diff API)
 - Discussion thread management (create, close outdated)
 - Resilient HTTP operations with retry/backoff and transient classification

Configuration Model (New):
---------------------------------
The module exposes an immutable :class:`DevOpsConfig` for explicit, testable
usage. Acquire one via :func:`load_config` (preferred) or :func:`ensure_env`
(backward-compatible shim). Functions accept an optional ``config`` parameter;
when omitted they fall back to ``GLOBAL_CONFIG`` or legacy module-level globals.

Migration Path:
 1. Replace calls to ``ensure_env()`` with ``cfg = load_config(verify=True)``.
 2. Pass ``config=cfg`` to subsequent function calls.
 3. Remove any reliance on implicit globals. (Future deprecation planned.)

Backward Compatibility:
``ensure_env`` still populates historical globals so existing code continues
working. New code should avoid relying on those globals for clarity and easier
unit testing (dependency injection of config objects).

Error Handling Strategy:
 - Missing / invalid environment or auth → ``MissingEnvironmentError``
 - API / network / format issues → ``AzureDevOpsAPIError`` (raised after JSON
     parsing or structural validation failures). Raw ``requests.HTTPError`` may
     still surface from ``run_with_retry`` when retries exhausted; future work
     may wrap these consistently.

Retry/Backoff:
Exponential backoff with jitter (cap at ``MAX_BACKOFF_DELAY``) for transient
statuses (408, 409, 429, 500, 502, 503, 504) and network errors. Honors the
``Retry-After`` header (capped to 60s).

Thread Closing Logic:
AI-tagged threads (identified by ``AI_COMMENT_TAG``) are programmatically
closed if their anchored line number disappears from the latest diff.

Planned Enhancements:
 - Pagination for large thread / diff responses
 - Session reuse (inject ``requests.Session``)
 - Unified error wrapping after final retry attempt
 - Config-driven retry policy instead of per-function parameters
 - Deprecation and removal of legacy mutable globals
"""

import os
import base64
import time
import random
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable, TypedDict, Any

import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger("ai_code_review")

AI_COMMENT_TAG = "[AI Review]"
EXCLUDE_FOLDERS = {'.github', 'review'}

# HTTP / API constants
API_VERSION = "7.0"
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
MAX_BACKOFF_DELAY = 20.0  # seconds cap on exponential backoff

class MissingEnvironmentError(RuntimeError):
    """Raised when required Azure DevOps environment variables are missing."""
    pass

class AzureDevOpsAPIError(RuntimeError):
    """Raised for non-environment API related errors (network/format)."""
    pass

__all__ = [
    'AI_COMMENT_TAG', 'EXCLUDE_FOLDERS', 'TRANSIENT_STATUS', 'API_VERSION',
    'MissingEnvironmentError', 'AzureDevOpsAPIError',
    'DevOpsConfig', 'load_config', 'ensure_env', 'GLOBAL_CONFIG',
    'auth_headers', 'run_with_retry', 'get_pr_commits', 'get_pr_changed_files',
    'fetch_existing_threads', 'collect_existing_comments', 'post_review_comment', 'close_outdated_ai_threads'
]

class ThreadContext(TypedDict, total=False):
    filePath: str
    rightFileStart: Dict[str, int]
    rightFileEnd: Dict[str, int]

class Thread(TypedDict, total=False):
    id: int
    threadContext: ThreadContext
    comments: List[Dict[str, Any]]

AZURE_DEVOPS_ORG = os.getenv('AZURE_DEVOPS_ORG')
AZURE_DEVOPS_PROJECT = os.getenv('AZURE_DEVOPS_PROJECT')
AZURE_DEVOPS_PR_ID = os.getenv('AZURE_DEVOPS_PR_ID')
AZURE_DEVOPS_REPO_ID = os.getenv('AZURE_DEVOPS_REPO_ID')
AZURE_DEVOPS_PAT = os.getenv('AZURE_DEVOPS_PAT')
AZURE_DEVOPS_AUTH_B64 = os.getenv('AZURE_DEVOPS_AUTH')  # might be set directly


@dataclass(frozen=True)
class DevOpsConfig:
    """Immutable configuration object for Azure DevOps operations.

    Prefer passing this explicitly to functions for testability and to
    eliminate reliance on mutable module-level globals.
    """
    org: str
    project: str
    pr_id: str
    repo_id: str
    auth_b64: str
    api_version: str = API_VERSION


# Holds the most recently validated configuration (set via ensure_env).
GLOBAL_CONFIG: Optional[DevOpsConfig] = None


def load_config(verify: bool = False, timeout: float = 5.0) -> DevOpsConfig:
    """Build a :class:`DevOpsConfig` from environment variables without mutating globals.

    Parameters:
        verify: If True perform a lightweight authenticated request (fetch PR metadata) to
                validate credentials and permissions.
        timeout: Timeout (seconds) for the optional verification request.

    Returns:
        DevOpsConfig: Immutable configuration object.

    Raises:
        MissingEnvironmentError: Missing required variables or failed verification/auth.
    """
    org = os.getenv('AZURE_DEVOPS_ORG')
    project = os.getenv('AZURE_DEVOPS_PROJECT')
    pr_id = os.getenv('AZURE_DEVOPS_PR_ID')
    repo_id = os.getenv('AZURE_DEVOPS_REPO_ID')
    pat = os.getenv('AZURE_DEVOPS_PAT')
    auth_b64 = os.getenv('AZURE_DEVOPS_AUTH')

    required = [
        ('AZURE_DEVOPS_ORG', org),
        ('AZURE_DEVOPS_PROJECT', project),
        ('AZURE_DEVOPS_PR_ID', pr_id),
        ('AZURE_DEVOPS_REPO_ID', repo_id),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        raise MissingEnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    if not auth_b64 and pat:
        auth_b64 = base64.b64encode(f":{pat}".encode()).decode()
    if not auth_b64:
        raise MissingEnvironmentError("Provide either AZURE_DEVOPS_AUTH (base64 basic token) or AZURE_DEVOPS_PAT.")

    cfg = DevOpsConfig(org=org, project=project, pr_id=pr_id, repo_id=repo_id, auth_b64=auth_b64)

    if verify:
        # Perform minimal call to verify auth & scope
        url = (f"https://dev.azure.com/{cfg.org}/{cfg.project}/_apis/git/repositories/"
               f"{cfg.repo_id}/pullRequests/{cfg.pr_id}?api-version={cfg.api_version}")
        try:
            resp = requests.get(url, headers={"Authorization": f"Basic {cfg.auth_b64}"}, timeout=timeout)
            if resp.status_code == 401:
                raise MissingEnvironmentError("Authentication failed (401). Check PAT or encoded auth token.")
            if resp.status_code == 403:
                raise MissingEnvironmentError("Forbidden (403). PAT lacks required scopes (Code: Read & Status).")
            if resp.status_code == 404:
                raise MissingEnvironmentError("PR or repository not found (404). Verify IDs and permissions.")
            if not (200 <= resp.status_code < 300):
                raise MissingEnvironmentError(f"Verification request failed (status={resp.status_code}).")
        except requests.RequestException as e:
            raise MissingEnvironmentError(f"Verification network error: {e}")
    return cfg


def ensure_env(verify: bool = False, timeout: float = 5.0) -> DevOpsConfig:
    """Validate environment and store resulting config globally.

    Backward-compatible wrapper around :func:`load_config`.

    Side Effects:
        - Assigns GLOBAL_CONFIG
        - Updates legacy AZURE_DEVOPS_* module-level globals for older calling code

    Parameters:
        verify: Optional verification request (see ``load_config``)
        timeout: Timeout for verification request

    Returns:
        DevOpsConfig: The validated configuration object
    """
    global GLOBAL_CONFIG, AZURE_DEVOPS_ORG, AZURE_DEVOPS_PROJECT, AZURE_DEVOPS_PR_ID, AZURE_DEVOPS_REPO_ID
    global AZURE_DEVOPS_PAT, AZURE_DEVOPS_AUTH_B64
    cfg = load_config(verify=verify, timeout=timeout)
    # Update legacy globals for backward compatibility
    AZURE_DEVOPS_ORG = cfg.org
    AZURE_DEVOPS_PROJECT = cfg.project
    AZURE_DEVOPS_PR_ID = cfg.pr_id
    AZURE_DEVOPS_REPO_ID = cfg.repo_id
    AZURE_DEVOPS_AUTH_B64 = cfg.auth_b64
    AZURE_DEVOPS_PAT = None  # Pat no longer required once auth_b64 derived
    GLOBAL_CONFIG = cfg
    return cfg


def auth_headers(extra: Optional[Dict[str, str]] = None, config: Optional[DevOpsConfig] = None) -> Dict[str, str]:
    """Build authorization headers, merging any extras.

    Resolution order for credentials:
        1. Explicit ``config`` parameter (preferred)
        2. GLOBAL_CONFIG (if previously set via ensure_env)
        3. Legacy module-level AZURE_DEVOPS_AUTH_B64

    Parameters:
        extra: Optional additional headers to merge (e.g., Content-Type)
        config: Optional explicit DevOpsConfig instance

    Raises:
        MissingEnvironmentError: If no credentials available from any source.
    """
    cfg = config or GLOBAL_CONFIG
    if cfg is None:
        # Fallback to legacy global token (pre-config migration)
        if not AZURE_DEVOPS_AUTH_B64:
            raise MissingEnvironmentError("auth_headers called before ensure_env/load_config set credentials")
        h = {"Authorization": f"Basic {AZURE_DEVOPS_AUTH_B64}"}
    else:
        h = {"Authorization": f"Basic {cfg.auth_b64}"}
    if extra:
        h.update(extra)
    return h


def _compute_delay(attempt: int, base: float, backoff: float) -> float:
    return base * (backoff ** (attempt - 1))

def _cap_delay(d: float) -> float:
    return d if d < MAX_BACKOFF_DELAY else MAX_BACKOFF_DELAY


def run_with_retry(func: Callable[[], requests.Response],
                   retries: int = 5,
                   base_delay: float = 0.8,
                   backoff: float = 2.0,
                   jitter: float = 0.25) -> requests.Response:
    """Run a callable returning a requests.Response with retry on transient failures.

    Retries on:
      - Connection errors / timeouts
      - HTTP status in TRANSIENT_STATUS
    Honors Retry-After header if present (caps at 60s)
    """
    attempt = 1
    while True:
        try:
            resp = func()
            if resp.status_code in TRANSIENT_STATUS and attempt <= retries:
                raise requests.HTTPError(f"Transient status {resp.status_code}", response=resp)
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            err = f"network error: {e}"  # transient
        except requests.HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            if status not in TRANSIENT_STATUS or attempt > retries:
                raise
            retry_after = None
            if e.response is not None:
                hdr = e.response.headers or {}
                retry_after = hdr.get('Retry-After') or hdr.get('retry-after')
            if retry_after:
                try:
                    delay = min(float(retry_after), 60.0)
                except ValueError:
                    delay = _compute_delay(attempt, base_delay, backoff)
            else:
                delay = _compute_delay(attempt, base_delay, backoff)
            delay = _cap_delay(delay + random.uniform(0, jitter))
            logger.warning("Transient HTTP %s (attempt %d/%d) retrying in %.2fs", status, attempt, retries, delay)
            if attempt == retries:
                raise
            time.sleep(delay)
            attempt += 1
            continue
        except Exception:
            # Unknown exception: do not classify as transient
            raise
        # Connection / timeout path
        if attempt == retries:
            raise
        delay = _cap_delay(_compute_delay(attempt, base_delay, backoff) + random.uniform(0, jitter))
        logger.warning("Transient %s (attempt %d/%d) retrying in %.2fs", err, attempt, retries, delay)
        time.sleep(delay)
        attempt += 1


def _api_base(config: Optional[DevOpsConfig] = None) -> str:
    if config is None:
        return (f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/"
                f"{AZURE_DEVOPS_REPO_ID}")
    return (f"https://dev.azure.com/{config.org}/{config.project}/_apis/git/repositories/"
            f"{config.repo_id}")

def get_pr_commits(timeout: float = 10.0, config: Optional[DevOpsConfig] = None) -> Tuple[str, str]:
    """Return (source_head_sha, target_base_sha) for the current PR.

    Parameters:
        timeout: Request timeout seconds.
        config: Optional DevOpsConfig. If omitted, falls back to GLOBAL_CONFIG / legacy globals.
    """
    cfg = config or GLOBAL_CONFIG
    api_version = cfg.api_version if cfg else API_VERSION
    pr_id = cfg.pr_id if cfg else AZURE_DEVOPS_PR_ID
    url = f"{_api_base(cfg)}/pullRequests/{pr_id}?api-version={api_version}"
    resp = run_with_retry(lambda: requests.get(url, headers=auth_headers(config=cfg), timeout=timeout))
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as e:
        raise AzureDevOpsAPIError(f"Non-JSON response for PR commits (status={resp.status_code})") from e
    try:
        return data['lastMergeSourceCommit']['commitId'], data['lastMergeTargetCommit']['commitId']
    except KeyError as e:
        raise AzureDevOpsAPIError(f"Missing expected keys in PR metadata: {e}") from e


def get_pr_changed_files(pr_source_sha: str, pr_target_sha: str, timeout: float = 10.0, config: Optional[DevOpsConfig] = None) -> List[str]:
    """List changed file paths between target and source commits for the PR.

    Parameters:
        pr_source_sha: Head commit of the PR (source branch)
        pr_target_sha: Target branch commit
        timeout: Request timeout seconds
        config: Optional DevOpsConfig
    """
    cfg = config or GLOBAL_CONFIG
    api_version = cfg.api_version if cfg else API_VERSION
    diff_url = (f"{_api_base(cfg)}/diffs/commits?baseVersion={pr_target_sha}&targetVersion={pr_source_sha}&api-version={api_version}")
    resp = run_with_retry(lambda: requests.post(diff_url, headers=auth_headers({"Content-Type": "application/json"}, config=cfg), json={}, timeout=timeout))
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError as e:
        raise AzureDevOpsAPIError(f"Non-JSON response for diff (status={resp.status_code})") from e
    files: List[str] = []
    for change in payload.get('changes', []):
        item = change.get('item') or {}
        if item.get('gitObjectType') == 'blob':
            path = item.get('path', '').lstrip('/')
            if path and not any(path.startswith(folder + '/') for folder in EXCLUDE_FOLDERS):
                files.append(path)
    return files


def fetch_existing_threads(timeout: float = 10.0, config: Optional[DevOpsConfig] = None) -> List[Thread]:
    """Return all discussion threads for the PR.

    Parameters:
        timeout: Request timeout seconds
        config: Optional DevOpsConfig
    """
    cfg = config or GLOBAL_CONFIG
    api_version = cfg.api_version if cfg else API_VERSION
    pr_id = cfg.pr_id if cfg else AZURE_DEVOPS_PR_ID
    url = f"{_api_base(cfg)}/pullRequests/{pr_id}/threads?api-version={api_version}"
    resp = run_with_retry(lambda: requests.get(url, headers=auth_headers(config=cfg), timeout=timeout))
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as e:
        raise AzureDevOpsAPIError(f"Non-JSON response for threads (status={resp.status_code})") from e
    return data.get('value', [])  # type: ignore[return-value]


def collect_existing_comments(threads, file_path: str) -> Tuple[List[str], List[dict]]:
    """Separate human comments and AI-tagged threads for a specific file path."""
    human_comments: List[str] = []
    ai_threads: List[dict] = []
    for thread in threads:
        context = thread.get('threadContext', {})
        if context.get('filePath', '').lstrip('/') == file_path:
            is_ai = False
            for c in thread.get('comments', []):
                content = c.get('content', '')
                if AI_COMMENT_TAG in content:
                    is_ai = True
                else:
                    human_comments.append(content)
            if is_ai:
                ai_threads.append(thread)
    return human_comments, ai_threads


def post_review_comment(file_path: str, comment: str, line: Optional[int] = None, timeout: float = 10.0, config: Optional[DevOpsConfig] = None) -> bool:
    """Post a review comment.

    Returns True on success, False otherwise. Automatically prepends AI tag
    if missing.

    Parameters:
        file_path: Path of file in repo
        comment: Comment body (AI tag auto-added if missing)
        line: Optional line number anchor (right side)
        timeout: Request timeout seconds
        config: Optional DevOpsConfig
    """
    body = f"{AI_COMMENT_TAG} {comment}" if AI_COMMENT_TAG not in comment else comment
    thread_context = {"filePath": f"/{file_path}"}
    if line:
        thread_context["rightFileStart"] = {"line": line}
        thread_context["rightFileEnd"] = {"line": line}
    payload = {
        "comments": [{"parentCommentId": 0, "content": body, "commentType": 1}],
        "status": 1,
        "threadContext": thread_context
    }
    cfg = config or GLOBAL_CONFIG
    api_version = cfg.api_version if cfg else API_VERSION
    pr_id = cfg.pr_id if cfg else AZURE_DEVOPS_PR_ID
    url = f"{_api_base(cfg)}/pullRequests/{pr_id}/threads?api-version={api_version}"
    resp = run_with_retry(lambda: requests.post(url, headers=auth_headers({"Content-Type": "application/json"}, config=cfg), json=payload, timeout=timeout))
    if not (200 <= resp.status_code < 300):
        logger.warning("Failed to post comment for %s (status=%s): %s", file_path, resp.status_code, resp.text[:300])
        return False
    return True


def close_outdated_ai_threads(ai_threads: List[Thread], current_added_lines: set, timeout: float = 10.0, config: Optional[DevOpsConfig] = None):
    """Close AI threads whose anchor line no longer appears in the newest diff.

    Parameters:
        ai_threads: List of AI-tagged threads (subset of fetch_existing_threads output)
        current_added_lines: Set of currently valid added line numbers for the file
        timeout: Request timeout seconds
        config: Optional DevOpsConfig
    """
    cfg = config or GLOBAL_CONFIG
    api_version = cfg.api_version if cfg else API_VERSION
    pr_id = cfg.pr_id if cfg else AZURE_DEVOPS_PR_ID
    for thread in ai_threads:
        thread_id = thread.get('id')
        context = thread.get('threadContext', {})
        start = context.get('rightFileStart', {}).get('line')
        if start and start not in current_added_lines:
            url = f"{_api_base(cfg)}/pullRequests/{pr_id}/threads/{thread_id}?api-version={api_version}"
            payload = {"status": 2}
            run_with_retry(lambda: requests.patch(url, headers=auth_headers({"Content-Type": "application/json"}, config=cfg), json=payload, timeout=timeout))
