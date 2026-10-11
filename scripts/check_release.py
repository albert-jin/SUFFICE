"""Check the publishable Git file set for payloads, links, paths, and assets."""
import ast
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npz', '.npy',
                      '.parquet', '.arrow', '.jsonl', '.zip', '.tar', '.gz', '.key', '.pem'}

def main():
    listed = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=REPO)
    files = sorted({p.decode('utf-8') for p in listed.split(b'\0') if p})
    errors = []
    for name in files:
        path = REPO / name
        if not path.exists():
            errors.append('Missing tracked file: ' + name)
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith('.env'):
            errors.append('Excluded payload type: ' + name)
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append('Unexpected large file: ' + name)
        if path.suffix in {'.py', '.md', '.json', '.toml', '.txt', '.csv'}:
            text = path.read_text(encoding='utf-8-sig')
            if re.search(r'[A-Za-z]:[\\/](?:Users|huawei-work|Asoftware|Bsoftware)', text, re.I):
                errors.append('Private absolute path: ' + name)
            if re.search(r'github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----', text):
                errors.append('Potential credential: ' + name)
            if path.suffix == '.py':
                ast.parse(text, filename=name)
    readme = (REPO / 'README.md').read_text(encoding='utf-8')
    if re.search(r'https?://|www\.|<a\b|(?<!!)\[[^\]]+\]\s*\(|^\s*\[[^\]]+\]:|[\w.+-]+@[\w.-]+\.[a-z]{2,}', readme, re.I | re.M):
        errors.append('README contains a hyperlink or contact address.')
    images = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', readme)
    if len(images) < 3:
        errors.append('README is missing expected displayed figures.')
    for name in images:
        resolved = (REPO / name).resolve()
        if REPO not in resolved.parents or not resolved.is_file():
            errors.append('Invalid relative image: ' + name)
        if not resolved.with_suffix('.pdf').is_file():
            errors.append('Missing PDF companion: ' + name)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({'files_checked': len(files), 'readme_images': len(images),
                      'bytes': sum((REPO / p).stat().st_size for p in files), 'status': 'passed'}))

if __name__ == '__main__':
    main()
