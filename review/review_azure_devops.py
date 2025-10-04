import os
import sys
import json
import time
import base64
import logging
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import mimetypes

import requests
import openai

AI_COMMENT_TAG = "[AI Review]"
EXCLUDE_FOLDERS = {'.github', 'review'}
MAX_FILE_BYTES = 60_000  # guard against very large files
MAX_CHANGED_LINES = 800   # skip overly large diffs to control token usage
DIFF_CONTEXT_LINES = 3

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger("ai_code_review")

# Get PR info from environment variables (set by Azure DevOps)
AZURE_DEVOPS_ORG = os.getenv('AZURE_DEVOPS_ORG')
AZURE_DEVOPS_PROJECT = os.getenv('AZURE_DEVOPS_PROJECT')
AZURE_DEVOPS_PR_ID = os.getenv('AZURE_DEVOPS_PR_ID')
AZURE_DEVOPS_REPO_ID = os.getenv('AZURE_DEVOPS_REPO_ID')
AZURE_DEVOPS_PAT = os.getenv('AZURE_DEVOPS_PAT')  # optional if using pre-built basic token
AZURE_DEVOPS_AUTH_B64 = os.getenv('AZURE_DEVOPS_AUTH')  # pre-encoded basic auth (username:pat -> base64)

def ensure_env():
    required = [
        ('AZURE_DEVOPS_ORG', AZURE_DEVOPS_ORG),
        ('AZURE_DEVOPS_PROJECT', AZURE_DEVOPS_PROJECT),
        ('AZURE_DEVOPS_PR_ID', AZURE_DEVOPS_PR_ID),
        ('AZURE_DEVOPS_REPO_ID', AZURE_DEVOPS_REPO_ID),
        ('OPENAI_MODEL', os.getenv('OPENAI_MODEL')),
    ]
    missing = [k for k,v in required if not v]
    if missing:
        logger.error("Missing required environment variables: %s", ', '.join(missing))
        sys.exit(2)
    global AZURE_DEVOPS_AUTH_B64
    if not AZURE_DEVOPS_AUTH_B64 and AZURE_DEVOPS_PAT:
        # Azure DevOps Basic auth uses :PAT as user:pass pair
        AZURE_DEVOPS_AUTH_B64 = base64.b64encode(f":{AZURE_DEVOPS_PAT}".encode()).decode()
    if not AZURE_DEVOPS_AUTH_B64:
        logger.error("Provide either AZURE_DEVOPS_AUTH (base64 basic token) or AZURE_DEVOPS_PAT.")
        sys.exit(2)

def auth_headers(extra: Optional[Dict[str,str]] = None) -> Dict[str,str]:
    h = {"Authorization": f"Basic {AZURE_DEVOPS_AUTH_B64}"}
    if extra:
        h.update(extra)
    return h

@dataclass
class ChangedHunk:
    file_path: str
    added_start: int
    added_lines: List[str]
    removed_start: int
    removed_count: int
    header: str

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

# Fetch changed files in PR

def get_pr_commits() -> Tuple[str,str]:
    url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}?api-version=7.0"
    resp = run_with_retry(lambda: requests.get(url, headers=auth_headers()))
    resp.raise_for_status()
    data = resp.json()
    return data['lastMergeSourceCommit']['commitId'], data['lastMergeTargetCommit']['commitId']

def get_pr_changed_files(base_sha: str, target_sha: str) -> List[str]:
    # Use diff api
    diff_url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/diffs/commits?baseVersion={target_sha}&targetVersion={base_sha}&api-version=7.0"
    resp = run_with_retry(lambda: requests.post(diff_url, headers=auth_headers({"Content-Type":"application/json"}), json={}))
    resp.raise_for_status()
    files = []
    for change in resp.json().get('changes', []):
        item = change.get('item') or {}
        if item.get('gitObjectType') == 'blob':
            path = item.get('path','').lstrip('/')
            if path and not any(path.startswith(folder + '/') for folder in EXCLUDE_FOLDERS):
                files.append(path)
    return files

TEXT_EXTENSIONS = {
    '.c','.h','.cpp','.hpp','.cc','.py','.js','.ts','.tsx','.jsx','.json','.md','.txt','.yml','.yaml','.xml','.html','.css','.scss','.ini','.cfg','.conf','.toml','.sh','.bash','.zsh','.gitignore','.dockerfile','Dockerfile','Makefile'
}

def is_probably_text(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(path)
    if mime and (mime.startswith('text/') or 'json' in mime or 'xml' in mime):
        return True
    # Fallback heuristic: small and no NUL bytes
    if os.path.exists(path) and os.path.getsize(path) < 200_000:
        try:
            with open(path,'rb') as f:
                chunk = f.read(4096)
            if b'\x00' not in chunk:
                return True
        except Exception:
            return False
    return False

def parse_diff_for_file(base_sha: str, target_sha: str, file_path: str) -> List[ChangedHunk]:
    # Get raw unified diff for a file
    diff_api = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/diffs/commits?baseVersion={target_sha}&targetVersion={base_sha}&$top=1&api-version=7.0&diffCommonCommit=false"
    # Fall back to local git if available
    try:
        import subprocess
        proc = subprocess.run(['git','show', f'{target_sha}:{file_path}'], capture_output=True, text=True)
        old_content = proc.stdout if proc.returncode==0 else ''
        proc2 = subprocess.run(['git','show', f'{base_sha}:{file_path}'], capture_output=True, text=True)
        new_content = proc2.stdout if proc2.returncode==0 else ''
    except Exception:
        old_content = new_content = ''
    # Simpler approach: use git diff locally for precise hunks
    hunks: List[ChangedHunk] = []
    try:
        import subprocess
        diff_out = subprocess.run(['git','diff', f'{target_sha}...{base_sha}', '--', file_path], capture_output=True, text=True).stdout
        current_header = ''
        for line in diff_out.splitlines():
            if line.startswith('@@'):
                # @@ -a,b +c,d @@ optional header
                header = line
                current_header = header
                parts = header.split(' ')
                removed_part = parts[1]  # -a,b
                added_part = parts[2]    # +c,d
                r_start, r_count = removed_part[1:].split(',') if ',' in removed_part else (removed_part[1:], '1')
                a_start, a_count = added_part[1:].split(',') if ',' in added_part else (added_part[1:], '1')
                hunks.append(ChangedHunk(
                    file_path=file_path,
                    added_start=int(a_start),
                    added_lines=[],
                    removed_start=int(r_start),
                    removed_count=int(r_count),
                    header=current_header
                ))
            elif line.startswith('+') and not line.startswith('+++') and hunks:
                hunks[-1].added_lines.append(line[1:])
    except Exception as e:
        logger.warning("Failed to parse diff for %s: %s", file_path, e)
    return hunks

# Review code using Azure OpenAI

def fetch_existing_threads() -> List[dict]:
    url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}/threads?api-version=7.0"
    resp = run_with_retry(lambda: requests.get(url, headers=auth_headers()))
    resp.raise_for_status()
    return resp.json().get('value', [])

def collect_existing_comments(threads, file_path: str) -> Tuple[List[str], List[dict]]:
    human_comments = []
    ai_threads = []
    for thread in threads:
        context = thread.get('threadContext', {})
        if context.get('filePath','').lstrip('/') == file_path:
            is_ai = False
            for c in thread.get('comments', []):
                content = c.get('content','')
                if AI_COMMENT_TAG in content:
                    is_ai = True
                else:
                    human_comments.append(content)
            if is_ai:
                ai_threads.append(thread)
    return human_comments, ai_threads

def build_prompt(file_path: str, full_code: str, hunks: List[ChangedHunk], human_comments: List[str]) -> str:
    hunk_snippets = []
    total_added = sum(len(h.added_lines) for h in hunks)
    for h in hunks:
        context_block = '\n'.join(h.added_lines[:50])  # cap lines per hunk portion
        hunk_snippets.append(f"{h.header}\n{context_block}")
    human_comment_text = '\n'.join(human_comments) if human_comments else 'None'
    prompt = (
        "You are an experienced senior software engineer performing a focused code review. "
        "Only comment on the changed lines/hunks below. Provide concise, actionable feedback on correctness, security, performance, maintainability, and style. "
        "If everything in a hunk looks good, respond with 'LGTM for this hunk'. Do not repeat previous human comments.\n\n"
        f"File: {file_path}\n"
        f"Total added lines under review: {total_added}\n\n"
        f"Changed hunks (unified headers + added lines only):\n{chr(10).join(hunk_snippets)}\n\n"
        f"Relevant prior human comments (avoid duplication):\n{human_comment_text}\n"
        "Output format: For each hunk, prefix feedback with 'HUNK:' and a short summary. If no issues, state 'HUNK: <summary> - LGTM'."
    )
    return prompt

def openai_review(model: str, prompt: str) -> str:
    try:
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise senior code reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=900
        )
        return completion['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI chat completion failed: {e}")

def post_review_comment(file_path: str, comment: str, line: Optional[int]=None):
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
    url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}/threads?api-version=7.0"
    resp = run_with_retry(lambda: requests.post(url, headers=auth_headers({"Content-Type":"application/json"}), json=payload))
    if not (200 <= resp.status_code < 300):
        logger.warning("Failed to post comment for %s: %s", file_path, resp.text)

def close_outdated_ai_threads(ai_threads: List[dict], current_added_lines: set):
    for thread in ai_threads:
        thread_id = thread.get('id')
        context = thread.get('threadContext', {})
        start = context.get('rightFileStart', {}).get('line')
        if start and start not in current_added_lines:
            # close
            url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/pullRequests/{AZURE_DEVOPS_PR_ID}/threads/{thread_id}?api-version=7.0"
            payload = {"status": 2}
            run_with_retry(lambda: requests.patch(url, headers=auth_headers({"Content-Type":"application/json"}), json=payload))

def review_file(model: str, base_sha: str, target_sha: str, file_path: str, threads: List[dict], dry_run: bool=False):
    if not os.path.exists(file_path):
        logger.info("File %s no longer exists locally; skipping", file_path)
        return
    size = os.path.getsize(file_path)
    if size > MAX_FILE_BYTES:
        logger.info("Skipping %s (size %d > limit)", file_path, size)
        return
    with open(file_path,'r', encoding='utf-8', errors='replace') as f:
        full_code = f.read()
    hunks = parse_diff_for_file(base_sha, target_sha, file_path)
    if not hunks:
        logger.info("No hunks parsed for %s", file_path)
        return
    total_added = sum(len(h.added_lines) for h in hunks)
    if total_added > MAX_CHANGED_LINES:
        logger.info("Skipping %s (added lines %d > limit)", file_path, total_added)
        return
    human_comments, ai_threads = collect_existing_comments(threads, file_path)
    prompt = build_prompt(file_path, full_code, hunks, human_comments)
    ai_text = openai_review(model, prompt)
    # Split feedback by hunk markers
    feedback_lines = [l for l in ai_text.splitlines() if l.strip()]
    # Map each hunk to first feedback line containing 'HUNK:' sequentially
    hunk_feedback: List[Tuple[ChangedHunk, str]] = []
    fi = 0
    for h in hunks:
        fb = []
        while fi < len(feedback_lines):
            line = feedback_lines[fi]
            if line.startswith('HUNK:') and fb:
                break
            fb.append(line)
            fi += 1
            if line.startswith('HUNK:') and len(fb) == 1:
                # continue collecting until next HUNK or end
                continue
        hunk_feedback.append((h, '\n'.join(fb)))
    added_line_numbers = set()
    for h, fb in hunk_feedback:
        # choose first added line as anchor
        anchor_line = h.added_start
        added_line_numbers.update(range(h.added_start, h.added_start + len(h.added_lines)))
        if dry_run:
            logger.info("(Dry-run) Would comment on %s line %d:\n%s", file_path, anchor_line, fb)
        else:
            post_review_comment(file_path, fb, line=anchor_line)
    if not dry_run:
        close_outdated_ai_threads(ai_threads, added_line_numbers)

# Post review comments to Azure DevOps

    # Old posting & closing functions replaced by post_review_comment and close_outdated_ai_threads
    pass

def main():
    parser = argparse.ArgumentParser(description="AI code review for Azure DevOps PR")
    parser.add_argument('--dry-run', action='store_true', help='Do not post comments; just log intended actions')
    args = parser.parse_args()
    ensure_env()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(2)
    openai.api_key = openai_api_key
    model_name = os.getenv('OPENAI_MODEL')
    base_sha, target_sha = get_pr_commits()
    logger.info("Base (source) commit: %s Target (main) commit: %s", base_sha[:8], target_sha[:8])
    files = get_pr_changed_files(base_sha, target_sha)
    logger.info("Changed files (raw): %s", files)
    text_files = [f for f in files if is_probably_text(f)]
    skipped = set(files) - set(text_files)
    if skipped:
        logger.info("Skipping non-text files: %s", sorted(skipped))
    logger.info("Text files to review: %s", text_files)
    threads = fetch_existing_threads()
    for fp in text_files:
        review_file(model_name, base_sha, target_sha, fp, threads, dry_run=args.dry_run)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        sys.exit(1)
