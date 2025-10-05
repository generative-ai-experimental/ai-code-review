from pathlib import Path
from openai_client import create_openai_client, openai_review

c = create_openai_client(
    api_key="keykeykey",
    azure_endpoint="https://myopenaiabc.openai.azure.com/")

def read_file(path: Path) -> str:
    text = path.read_text(encoding='utf-8', errors='replace')
    return text

source = read_file(Path("src/main.c"))
prompt = f"Please provide a concise, actionable code review for the following file (path: src/main.c):\n\n" + source

response = openai_review(c, prompt=prompt, max_retries=3, temperature=0.1)

print(response)