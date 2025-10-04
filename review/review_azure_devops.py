import os
import sys
import json
import time
import random
import base64
import logging
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import mimetypes

import requests
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

AI_COMMENT_TAG = "[AI Review]"
EXCLUDE_FOLDERS = {'.github', 'review'}
MAX_FILE_BYTES = 60_000  # guard against very large files
MAX_CHANGED_LINES = 800   # skip overly large diffs to control token usage
DIFF_CONTEXT_LINES = 3

# Optional environment variables for tuning OpenAI retry behavior:
#   OPENAI_MAX_RETRIES      (int, default 5)
#   OPENAI_BASE_DELAY       (float seconds, default 1.0)
#   OPENAI_FALLBACK_MODEL   (string model name; switches after half the retries fail)
#   OPENAI_RETRY_JITTER     (float seconds max random jitter, default 0.25)

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
    context_block: List[str]  # lines including context (without diff +/- prefixes except we keep '+' for added to highlight)

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

def get_pr_changed_files(pr_source_sha: str, pr_target_sha: str) -> List[str]:
    """Return list of changed file paths in the PR.

    Azure DevOps diff API expects baseVersion=target (e.g., main) and targetVersion=source (PR head) to compute changes introduced.
    We intentionally maintain the naming alignment with parse_diff_for_file which uses pr_source_sha (head) and pr_target_sha (base).
    """
    diff_url = f"https://dev.azure.com/{AZURE_DEVOPS_ORG}/{AZURE_DEVOPS_PROJECT}/_apis/git/repositories/{AZURE_DEVOPS_REPO_ID}/diffs/commits?baseVersion={pr_target_sha}&targetVersion={pr_source_sha}&api-version=7.0"
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

def parse_diff_for_file(pr_source_sha: str, pr_target_sha: str, file_path: str) -> List[ChangedHunk]:
    """Parse diff hunks for a single file.

    Parameters:
      pr_source_sha: The HEAD/source commit of the PR (what is being merged) (aka baseSha in Azure PR object lastMergeSourceCommit)
      pr_target_sha: The target branch commit (e.g., main) (aka targetSha in Azure PR object lastMergeTargetCommit)
      file_path: Path to file to diff

    Env:
      AI_REVIEW_DIFF_MODE: 'two-dot' (default) or 'triple-dot'

    Rationale:
      We previously used a triple-dot (A...B) diff which produces the diff of merge base vs B. For PR review we typically
      want the direct change set between target (main) and the PR head. GitHub style comparisons often use three-dot, but
      local validation / correctness of added lines is clearer with two-dot when the branch is rebased. Allow both.
    """
    diff_mode = os.getenv('AI_REVIEW_DIFF_MODE', 'two-dot').strip().lower()
    if diff_mode not in {'two-dot', 'triple-dot'}:
        logger.warning("AI_REVIEW_DIFF_MODE '%s' invalid; defaulting to two-dot", diff_mode)
        diff_mode = 'two-dot'

    # Choose diff ref expression
    # two-dot: git diff pr_target_sha pr_source_sha => shows changes introduced by PR relative to target branch
    # triple-dot: git diff pr_target_sha...pr_source_sha => changes since common ancestor (can hide rebased removals/additions semantics differences)
    ref_expr = f"{pr_target_sha} {'...' if diff_mode == 'triple-dot' else ''} {pr_source_sha}".replace('  ', ' ').strip()
    # Build argument list without stray spaces
    if diff_mode == 'triple-dot':
        diff_args = ['git', 'diff', f'{pr_target_sha}...{pr_source_sha}', '--', file_path]
    else:
        diff_args = ['git', 'diff', pr_target_sha, pr_source_sha, '--', file_path]

    hunks: List[ChangedHunk] = []
    try:
        import subprocess
        diff_out = subprocess.run(diff_args, capture_output=True, text=True).stdout
        current_header = ''
        current_hunk_all_lines: List[str] = []  # raw lines (with +/-/space prefixes)
        for line in diff_out.splitlines():
            if line.startswith('@@'):
                # Flush previous hunk lines (if any) into last hunk's context_block
                if hunks and current_hunk_all_lines:
                    # finalize previous hunk context
                    hunks[-1].context_block = _extract_context_block(current_hunk_all_lines, hunks[-1].added_lines, DIFF_CONTEXT_LINES)
                current_hunk_all_lines = []
                header = line
                current_header = header
                parts = header.split(' ')
                if len(parts) < 3:
                    continue
                removed_part = parts[1]
                added_part = parts[2]
                r_start, r_count = removed_part[1:].split(',') if ',' in removed_part else (removed_part[1:], '1')
                a_start, a_count = added_part[1:].split(',') if ',' in added_part else (added_part[1:], '1')
                hunks.append(ChangedHunk(
                    file_path=file_path,
                    added_start=int(a_start),
                    added_lines=[],
                    removed_start=int(r_start),
                    removed_count=int(r_count),
                    header=current_header,
                    context_block=[]
                ))
            elif line.startswith('+') and not line.startswith('+++') and hunks:
                hunks[-1].added_lines.append(line[1:])
                current_hunk_all_lines.append(line)
            elif (line.startswith(' ') or line.startswith('-')) and not line.startswith('---') and hunks:
                current_hunk_all_lines.append(line)
        # finalize last hunk
        if hunks and current_hunk_all_lines:
            hunks[-1].context_block = _extract_context_block(current_hunk_all_lines, hunks[-1].added_lines, DIFF_CONTEXT_LINES)
    except Exception as e:
        logger.warning("Failed to parse diff for %s (%s mode): %s", file_path, diff_mode, e)
    return hunks

def _extract_context_block(raw_hunk_lines: List[str], added_lines: List[str], context_lines: int) -> List[str]:
    """Return a subset of the hunk including context_lines around added lines.

    raw_hunk_lines: lines with original diff prefixes ('+','-',' ') excluding the @@ header.
    We collect indices of added lines then expand by context_lines in both directions.
    Removed lines are retained only if within the selected window; we strip leading space prefix for neutral lines and keep '+' for added, '-' for removed.
    """
    if not added_lines:
        return []
    # Identify positions of added lines in raw_hunk_lines
    added_positions = [i for i,l in enumerate(raw_hunk_lines) if l.startswith('+')]
    if not added_positions:
        return []
    start = max(0, min(added_positions) - context_lines)
    end = min(len(raw_hunk_lines), max(added_positions) + context_lines + 1)
    window = raw_hunk_lines[start:end]
    # Normalize: remove leading space for context lines to reduce tokens
    normalized = []
    for l in window:
        if l.startswith(' '):
            normalized.append(l[1:])
        else:
            normalized.append(l)
    return normalized

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
        # Prefer context block if available; fallback to added lines only
        if h.context_block:
            snippet_lines = h.context_block[: 100]  # simple cap
        else:
            snippet_lines = h.added_lines[:50]
        hunk_snippets.append(f"{h.header}\n" + '\n'.join(snippet_lines))
    human_comment_text = '\n'.join(human_comments) if human_comments else 'None'
    lower_path = file_path.lower()
    is_markdown = lower_path.endswith('.md')
    is_c = lower_path.endswith('.c') or lower_path.endswith('.h')
    if is_markdown:
        prompt = (
            "You are an expert technical editor focused strictly on grammar, clarity, conciseness, tone consistency, and markdown formatting. "
            "Review ONLY the added '+' lines (surrounding context is provided for reference). Do NOT comment on unchanged context unless it directly affects an added sentence. "
            "Ignore code-style or implementation details—this is a documentation/markdown quality pass. "
            "Combine minor nits where reasonable. Avoid repeating human feedback. If a hunk is fine, respond 'LGTM for this hunk'.\n\n"
            f"File: {file_path}\n"
            f"Total added lines under review: {total_added}\n\n"
            f"Changed hunks (headers + context with '+' additions marked):\n{chr(10).join(hunk_snippets)}\n\n"
            f"Relevant prior human comments (avoid duplication):\n{human_comment_text}\n"
            "Output format: For each hunk, start a new line with 'HUNK:' followed by either 'LGTM' or a concise list of suggestions. "
            "For suggestions use bullet style '- original -> improved' or '- suggestion: <text>'. Do not include any preamble before the first HUNK line."
        )
    elif is_c:
        prompt = (
            "You are an expert embedded/C systems reviewer. Evaluate ONLY the added '+' lines (context shown) with focus on: MISRA C (latest edition) guideline adherence, "
            "memory safety, undefined/implementation-defined behavior, integer overflow, pointer misuse, resource leaks, concurrency/race conditions, security vulnerabilities (e.g., buffer overflows, injection, UB), and obvious typos or spelling mistakes in comments or newly introduced identifiers (flag only if clarity or maintainability is impacted). "
            "If a guideline is violated, cite it briefly (e.g., 'MISRA Dir 4.1' or 'MISRA Rule 17.7') without lengthy quotation. "
            "If everything in a hunk is acceptable, respond 'LGTM for this hunk'. "
            "Do NOT repeat human comments or comment on lines not changed unless directly required to explain a defect.\n\n"
            f"File: {file_path}\n"
            f"Total added lines under review: {total_added}\n\n"
            f"Changed hunks (headers + context; '+' indicates new code):\n{chr(10).join(hunk_snippets)}\n\n"
            f"Relevant prior human comments (avoid duplication):\n{human_comment_text}\n"
            "Output format: For each hunk, start with 'HUNK:' then zero or more issue bullets of the form '- [SEVERITY] Category: description (optional rule/ref)'. "
            "Severity must be one of: CRITICAL, HIGH, MEDIUM, LOW, INFO. If no issues: 'HUNK: <summary> - LGTM'. "
            "Categories examples: Memory, Concurrency, UndefinedBehavior, Style, Portability, Security. Keep each bullet concise."
        )
    else:
        prompt = (
            "You are an experienced senior software engineer performing a focused code review. "
            "Only comment on the changed '+' lines (context provided). Provide concise, actionable feedback on correctness, security, performance, maintainability, and style. "
            "If everything in a hunk looks good, respond with 'LGTM for this hunk'. Do not repeat previous human comments.\n\n"
            f"File: {file_path}\n"
            f"Total added lines under review: {total_added}\n\n"
            f"Changed hunks (unified headers + context):\n{chr(10).join(hunk_snippets)}\n\n"
            f"Relevant prior human comments (avoid duplication):\n{human_comment_text}\n"
            "Output format: For each hunk, prefix feedback with 'HUNK:' and a short summary. If no issues, state 'HUNK: <summary> - LGTM'."
        )
    return prompt

def openai_review(client, model: str, prompt: str) -> str:
    """Call OpenAI chat completion with advanced retry/backoff & optional fallback.

    Env vars (optional):
      OPENAI_MAX_RETRIES: int (default 5)
      OPENAI_BASE_DELAY: float seconds (default 1.0)
      OPENAI_FALLBACK_MODEL: model name to switch to mid-way if primary failing
      OPENAI_RETRY_JITTER: float max random jitter seconds (default 0.25)
    """
    max_retries = int(os.getenv('OPENAI_MAX_RETRIES', '5'))
    base_delay = float(os.getenv('OPENAI_BASE_DELAY', '1.0'))
    fallback_model = os.getenv('OPENAI_FALLBACK_MODEL')
    jitter_cap = float(os.getenv('OPENAI_RETRY_JITTER', '0.25'))
    using_fallback = False

    def should_retry(status_code: Optional[int]) -> bool:
        if status_code is None:
            return True  # network / unknown
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
        except Exception as e:
            # Attempt to introspect error shape (OpenAI python client may change)
            status_code = getattr(e, 'status_code', None)
            retry_after = None
            if hasattr(e, 'response') and getattr(e, 'response') is not None:
                try:
                    status_code = getattr(e.response, 'status_code', status_code)
                    rh = getattr(e.response, 'headers', {}) or {}
                    retry_after = rh.get('Retry-After') or rh.get('retry-after')
                except Exception:
                    pass

            if not should_retry(status_code):
                raise RuntimeError(f"Non-retriable OpenAI error (status={status_code}): {e}")

            if attempt == (max_retries // 2) and fallback_model and not using_fallback:
                logger.warning("Switching to fallback model '%s' after %d failed attempts of '%s'", fallback_model, attempt, model)
                model = fallback_model
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
            logger.warning("OpenAI request failed (attempt %d/%d, status=%s, fallback=%s): %s; retrying in %.2fs", attempt, max_retries, status_code, using_fallback, e, delay)
            time.sleep(delay)

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

def review_file(client, model: str, pr_source_sha: str, pr_target_sha: str, file_path: str, threads: List[dict], dry_run: bool=False, severity_totals: Optional[Dict[str,int]] = None):
    fail_fast = os.getenv('AI_REVIEW_FAIL_FAST', 'false').lower() in {'1','true','yes'}
    try:
        if not os.path.exists(file_path):
            logger.info("File %s no longer exists locally; skipping", file_path)
            return
        size = os.path.getsize(file_path)
        if size > MAX_FILE_BYTES:
            logger.info("Skipping %s (size %d > limit)", file_path, size)
            return
        with open(file_path,'r', encoding='utf-8', errors='replace') as f:
            full_code = f.read()
        hunks = parse_diff_for_file(pr_source_sha, pr_target_sha, file_path)
        if not hunks:
            logger.info("No hunks parsed for %s", file_path)
            return
        total_added = sum(len(h.added_lines) for h in hunks)
        if total_added > MAX_CHANGED_LINES:
            logger.info("Skipping %s (added lines %d > limit)", file_path, total_added)
            return
        human_comments, ai_threads = collect_existing_comments(threads, file_path)
        prompt = build_prompt(file_path, full_code, hunks, human_comments)
        ai_text = openai_review(client, model, prompt)
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
            anchor_line = h.added_start
            added_line_numbers.update(range(h.added_start, h.added_start + len(h.added_lines)))
            if dry_run:
                logger.info("(Dry-run) Would comment on %s line %d:\n%s", file_path, anchor_line, fb)
            else:
                post_review_comment(file_path, fb, line=anchor_line)
            # Aggregate severity counts if present and we are in C mode or generic; regex parse [SEVERITY]
            if severity_totals is not None:
                for line in fb.splitlines():
                    if line.startswith('- [') and ']' in line:
                        sev_token = line[3: line.find(']')].strip().upper()
                        if sev_token in {"CRITICAL","HIGH","MEDIUM","LOW","INFO"}:
                            severity_totals[sev_token] = severity_totals.get(sev_token, 0) + 1
        if not dry_run:
            close_outdated_ai_threads(ai_threads, added_line_numbers)
    except Exception as e:
        logger.error("Error processing file %s: %s", file_path, e, exc_info=True)
        if fail_fast:
            raise

# Post review comments to Azure DevOps

    # Old posting & closing functions replaced by post_review_comment and close_outdated_ai_threads
    pass

def main():
    parser = argparse.ArgumentParser(description="AI code review for Azure DevOps PR")
    parser.add_argument('--dry-run', action='store_true', help='Do not post comments; just log intended actions')
    args = parser.parse_args()
    ensure_env()
    if OpenAI is None:
        logger.error("openai library (>=1.x) not installed. Please add 'openai' to requirements.")
        sys.exit(2)
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(2)
    client = OpenAI(api_key=api_key)
    model_name = os.getenv('OPENAI_MODEL')
    pr_source_sha, pr_target_sha = get_pr_commits()
    logger.info("PR source (head) commit: %s Target (base) commit: %s", pr_source_sha[:8], pr_target_sha[:8])
    files = get_pr_changed_files(pr_source_sha, pr_target_sha)
    logger.info("Changed files (raw): %s", files)
    text_files = [f for f in files if is_probably_text(f)]
    skipped = set(files) - set(text_files)
    if skipped:
        logger.info("Skipping non-text files: %s", sorted(skipped))
    logger.info("Text files to review: %s", text_files)
    threads = fetch_existing_threads()
    severity_totals: Dict[str,int] = {}
    summary_enabled = os.getenv('AI_REVIEW_SUMMARY', 'true').lower() in {'1','true','yes','on'}
    for fp in text_files:
        try:
            review_file(client, model_name, pr_source_sha, pr_target_sha, fp, threads, dry_run=args.dry_run, severity_totals=severity_totals if summary_enabled else None)
        except Exception as e:
            logger.error("Failed reviewing %s: %s", fp, e)
    # Post summary comment if enabled and not dry-run
    if summary_enabled and not args.dry_run:
        total_comments = sum(severity_totals.values())
        if total_comments:
            # Build summary body
            ordered = [f"{k}: {severity_totals[k]}" for k in ["CRITICAL","HIGH","MEDIUM","LOW","INFO"] if k in severity_totals]
            summary_lines = ["AI Review Summary (severity counts across C/security analysis):", *ordered]
            body = '\n'.join(summary_lines)
            post_review_comment("PR_SUMMARY", body, line=None)
        else:
            logger.info("No severity-tagged issues detected; skipping summary comment.")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        sys.exit(1)
