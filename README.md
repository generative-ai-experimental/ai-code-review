# ai-code-review

This repository contains a minimal C project plus an automated Azure DevOps pull request reviewer powered by Azure OpenAI.

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
* Uses Azure OpenAI (chat completions first, falls back to legacy completions) for focused feedback
* Posts line-anchored comments tagged with `[AI Review]`
* Avoids duplicating human reviewer comments
* Auto-closes outdated AI comment threads when lines are removed/changed
* Skips very large files or excessively large diffs to control token usage
* Dry-run mode for local experimentation

### Environment Variables
Set the following in your pipeline or local shell:
```
AZURE_DEVOPS_ORG
AZURE_DEVOPS_PROJECT
AZURE_DEVOPS_PR_ID
AZURE_DEVOPS_REPO_ID
# One of these for authentication:
AZURE_DEVOPS_AUTH   # base64(:PAT) or base64(user:PAT)
AZURE_DEVOPS_PAT    # plain PAT; script derives basic auth if AUTH not set

AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT  # name of the deployed model (e.g. gpt-4o-mini)
```

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
			AZURE_OPENAI_ENDPOINT: $(AZURE_OPENAI_ENDPOINT)
			AZURE_OPENAI_DEPLOYMENT: $(AZURE_OPENAI_DEPLOYMENT)
		displayName: AI Code Review
```

Ensure the PAT used has permissions: Code (Read), Pull Request Threads (Read & Write).

### Limitations / Future Ideas
* Could add support for markdown summary comment.
* Potential enhancement: semantic filtering of unchanged context.
* Add unit tests for diff parsing logic.

---
Generated and maintained by an automated assistant.