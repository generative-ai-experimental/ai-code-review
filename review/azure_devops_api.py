import os
import sys
import base64
import time
import logging
from typing import Optional, Dict, List, Tuple

import requests

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger("ai_code_review")

# Constants & environment-derived configuration
AI_COMMENT_TAG = "[AI Review]"
EXCLUDE_FOLDERS = {'.github', 'review'}

AZURE_DEVOPS_ORG = os.getenv('AZURE_DEVOPS_ORG')
AZURE_DEVOPS_PROJECT = os.getenv('AZURE_DEVOPS_PROJECT')
AZURE_DEVOPS_PR_ID = os.getenv('AZURE_DEVOPS_PR_ID')
AZURE_DEVOPS_REPO_ID = os.getenv('AZURE_DEVOPS_REPO_ID')
AZURE_DEVOPS_PAT = os.getenv('AZURE_DEVOPS_PAT')
AZURE_DEVOPS_AUTH_B64 = os.getenv('AZURE_DEVOPS_AUTH')  # might be set directly


def ensure_env():
    required = [
        ('AZURE_DEVOPS_ORG', AZURE_DEVOPS_ORG),
        ('AZURE_DEVOPS_PROJECT', AZURE_DEVOPS_PROJECT),
        ('AZURE_DEVOPS_PR_ID', AZURE_DEVOPS_PR_ID),
        ('AZURE_DEVOPS_REPO_ID', AZURE_DEVOPS_REPO_ID),
        ('OPENAI_MODEL', os.getenv('OPENAI_MODEL')),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        logger.error("Missing required environment variables: %s", ', '.join(missing))
        sys.exit(2)
    global AZURE_DEVOPS_AUTH_B64
    if not AZURE_DEVOPS_AUTH_B64 and AZURE_DEVOPS_PAT:
        AZURE_DEVOPS_AUTH_B64 = base64.b64encode(f":{AZURE_DEVOPS_PAT}".encode()).decode()
    if not AZURE_DEVOPS_AUTH_B64:
        logger.error("Provide either AZURE_DEVOPS_AUTH (base64 basic token) or AZURE_DEVOPS_PAT.")
        sys.exit(2)


def auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {"Authorization": f"Basic {AZURE_DEVOPS_AUTH_B64}"}
    if extra:
        h.update(extra)
    return h


def run_with_retry(func, retries=3, delay=1.5, backoff=2.0):
    attempt = 0
    while True:
        try:
            return func()
        except Exception as e:  # broad for pipeline resilience
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(delay)
            delay *= backoff
            logger.warning("Retrying after error: %s", e)


def get_pr_commits() -> Tuple[str, str]:
    url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}?api-version=7.0"
    resp = run_with_retry(lambda: requests.get(url, headers=auth_headers()))
    resp.raise_for_status()
    data = resp.json()
    return data['lastMergeSourceCommit']['commitId'], data['lastMergeTargetCommit']['commitId']


def get_pr_changed_files(pr_source_sha: str, pr_target_sha: str) -> List[str]:
    diff_url = (f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/"
                f"{AZURE_DEVOPS_REPO_ID}/diffs/commits?baseVersion={pr_target_sha}&targetVersion={pr_source_sha}&api-version=7.0")
    resp = run_with_retry(lambda: requests.post(diff_url, headers=auth_headers({"Content-Type": "application/json"}), json={}))
    resp.raise_for_status()
    files = []
    for change in resp.json().get('changes', []):
        item = change.get('item') or {}
        if item.get('gitObjectType') == 'blob':
            path = item.get('path', '').lstrip('/')
            if path and not any(path.startswith(folder + '/') for folder in EXCLUDE_FOLDERS):
                files.append(path)
    return files


def fetch_existing_threads() -> List[dict]:
    url = (f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/"
           f"{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}/threads?api-version=7.0")
    resp = run_with_retry(lambda: requests.get(url, headers=auth_headers()))
    resp.raise_for_status()
    return resp.json().get('value', [])


def collect_existing_comments(threads, file_path: str):
    human_comments = []
    ai_threads = []
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


def post_review_comment(file_path: str, comment: str, line: Optional[int] = None):
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
    url = (f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/"
           f"{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}/threads?api-version=7.0")
    resp = run_with_retry(lambda: requests.post(url, headers=auth_headers({"Content-Type": "application/json"}), json=payload))
    if not (200 <= resp.status_code < 300):
        logger.warning("Failed to post comment for %s: %s", file_path, resp.text)


def close_outdated_ai_threads(ai_threads: List[dict], current_added_lines: set):
    for thread in ai_threads:
        thread_id = thread.get('id')
        context = thread.get('threadContext', {})
        start = context.get('rightFileStart', {}).get('line')
        if start and start not in current_added_lines:
            url = (f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/"
                   f"{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}/threads/{thread_id}?api-version=7.0")
            payload = {"status": 2}
            run_with_retry(lambda: requests.patch(url, headers=auth_headers({"Content-Type": "application/json"}), json=payload))
