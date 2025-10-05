# ai-code-review

This repository contains a minimal C project plus an automated Azure DevOps pull request reviewer powered by the OpenAI API.

## C Project

Located under `src/` with a simple `main.c`. Build using:
```
make
./main
```

## AI Pull Request Code Review (Azure DevOps)

The script `review/review_azure_devops.py` performs automated line-level review on PRs.

### Key Features
* Fetches changed files via Azure DevOps Diff API
* Parses unified diff hunks and only reviews added lines
* Configurable diff orientation: two-dot (default) or triple-dot via `AI_REVIEW_DIFF_MODE`
* Uses OpenAI Chat Completions (>=1.x python client) with advanced retry/backoff and optional fallback model
* Posts line-anchored comments tagged with `[AI Review]`
* Avoids duplicating human reviewer comments
* Auto-closes outdated AI comment threads when lines are removed/changed
* Skips very large files or excessively large diffs to control token usage
* Per-file failure isolation (optional fail-fast)
* Dry-run mode for local experimentation
* Markdown-aware mode: for `.md` files the AI focuses on grammar, clarity, tone, and formatting suggestions only
* C code awareness: for `.c` / `.h` files emphasizes MISRA C guideline adherence, memory safety, undefined behavior, concurrency/race risks, security vulnerabilities, and obvious typos in comments/identifiers (with severity tagging)
* Aggregated severity summary: posts a final PR-level comment with counts per severity (can disable via `AI_REVIEW_SUMMARY=false`)
* Explicit configuration object (`DevOpsConfig`) for testability and reduced global state coupling (new)

### Environment Variables
Set the following in your pipeline or local shell:
```
AZURE_DEVOPS_ORG
AZURE_DEVOPS_PROJECT
AZURE_DEVOPS_PR_ID
AZURE_DEVOPS_REPO_ID
# One of these for authentication:
AZURE_DEVOPS_AUTH        # base64(:PAT) or base64(user:PAT)
AZURE_DEVOPS_PAT         # plain PAT; script derives basic auth if AUTH not set

# OpenAI API
OPENAI_API_KEY           # standard OpenAI API key
OPENAI_MODEL             # e.g. gpt-4o-mini, gpt-4.1, etc.
OPENAI_MAX_RETRIES       # optional; default 5
OPENAI_BASE_DELAY        # optional; default 1.0 seconds
OPENAI_FALLBACK_MODEL    # optional; switches mid-way if primary keeps failing
OPENAI_RETRY_JITTER      # optional; default 0.25 seconds

# Review behavior
AI_REVIEW_FAIL_FAST      # true/1 to abort entire run on first file error
AI_REVIEW_DIFF_MODE      # two-dot (default) or triple-dot
AI_REVIEW_SUMMARY        # true/false (default true) to enable final severity summary comment
```

### Configuration Refactor (DevOpsConfig)

The Azure DevOps helper module (`review/azure_devops_api.py`) now supports an immutable configuration object `DevOpsConfig`.

Why this matters:
* Eliminates hidden mutable global state
* Enables easier unit testing (you can construct a config directly)
* Allows future injection of session objects, custom API versions, or multi-PR batch operations

Two ways to obtain configuration:
1. Preferred: `cfg = load_config(verify=True)` – validates env and returns a `DevOpsConfig` without mutating globals (unless you then call `ensure_env`).
2. Legacy-compatible: `cfg = ensure_env(verify=True)` – builds the same config and also updates legacy module-level globals for existing code.

### Usage Examples

Legacy style (still works):
```python
from review.azure_devops_api import ensure_env, get_pr_commits, get_pr_changed_files

ensure_env(verify=True)
src_sha, tgt_sha = get_pr_commits()
files = get_pr_changed_files(src_sha, tgt_sha)
```

Preferred explicit style:
```python
from review.azure_devops_api import load_config, get_pr_commits, get_pr_changed_files

cfg = load_config(verify=True)
src_sha, tgt_sha = get_pr_commits(config=cfg)
files = get_pr_changed_files(src_sha, tgt_sha, config=cfg)
```

Posting a comment with explicit config:
```python
from review.azure_devops_api import post_review_comment
post_review_comment("src/main.c", "Consider checking return value.", line=42, config=cfg)
```

Closing outdated AI threads while supplying the config:
```python
from review.azure_devops_api import close_outdated_ai_threads, fetch_existing_threads
threads = fetch_existing_threads(config=cfg)
# ...derive ai_threads + current_added_lines...
close_outdated_ai_threads(ai_threads, current_added_lines, config=cfg)
```

### Migration Guidance
You can migrate incrementally:
1. Start calling `load_config()` and threading the `config=` parameter into functions in new code paths.
2. Leave older code calling `ensure_env()` untouched until convenient to refactor.
3. Once all callers are explicit, you can remove uses of `ensure_env()` (future release may deprecate globals).

### Deprecation Notice (Planned)
Global implicit state (module-level `AZURE_DEVOPS_*` variables) is slated for deprecation in a future cleanup release. A warning will precede removal. Adopt the explicit `DevOpsConfig` pattern to future‑proof your integrations.

### Local Dry Run
```
pip install -r requirements.txt
python review/review_azure_devops.py --dry-run
```

### Azure DevOps Pipeline Snippet
```yaml
steps:
	- task: UsePythonVersion@0
		inputs:
			versionSpec: '3.11'
	- script: |
			pip install -r requirements.txt
			python review/review_azure_devops.py
		env:
			AZURE_DEVOPS_ORG: $(System.TeamFoundationCollectionUri:trimEnd('/').split('/')[3])
			AZURE_DEVOPS_PROJECT: $(System.TeamProject)
			AZURE_DEVOPS_PR_ID: $(System.PullRequest.PullRequestId)
			AZURE_DEVOPS_REPO_ID: $(Build.Repository.ID)
			AZURE_DEVOPS_PAT: $(AI_REVIEW_PAT)
			OPENAI_API_KEY: $(OPENAI_API_KEY)
			OPENAI_MODEL: gpt-4o-mini
			AI_REVIEW_DIFF_MODE: two-dot
		displayName: AI Code Review
```

Ensure the PAT used has permissions: Code (Read), Pull Request Threads (Read & Write).

### Provided Pipeline Examples
Two ready-to-use YAML files are included under `review/`:

| File | Purpose |
|------|---------|
| `review/azure-pipelines-minimal.yml` | Minimal single-job PR-only review pipeline |
| `review/azure-pipelines-multistage.yml` | Build stage + AI review stage separation |

Copy one of these to the repo root as `azure-pipelines.yml` or reference them via `extends`.

### Limitations / Future Ideas
* Could add support for markdown summary comment.
* Potential enhancement: semantic filtering of unchanged context.
* Add unit tests for diff parsing logic.
* Pagination for large thread/diff responses.
* Inject `requests.Session` for connection reuse & performance.
* Config-driven retry/backoff policy (currently per-call parameters).
* Deprecate legacy global environment usage path fully.

---
Generated and maintained by an automated assistant.