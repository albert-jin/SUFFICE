"""External artifact paths keep weights and datasets out of the release tree."""
import json
from pathlib import Path

def external_output(path):
    path = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    if (repository / 'pyproject.toml').exists() and (path == repository or repository in path.parents):
        raise ValueError('Write data, predictions, and weights outside the repository.')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_jsonl(path, rows):
    path = external_output(path)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
