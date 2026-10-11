"""Stage historical source files in a separate, new experiment workspace."""
import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO.parent / 'SUFFICE-WORKSPACE')
    args = parser.parse_args()
    output = args.output.resolve()
    if output == REPO or REPO in output.parents:
        parser.error('Experiment workspace must be outside the repository.')
    for folder in ['src', 'upstream']:
        target = output / folder
        if target.exists():
            parser.error('Refusing to overwrite existing source: ' + str(target))
    for folder in ['src', 'upstream']:
        shutil.copytree(REPO / 'research' / folder, output / folder, ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ['data', 'models', 'runs', 'manifests', 'logs']:
        (output / folder).mkdir(exist_ok=True)
    print('Workspace staged. Set AT_ROOT to this directory before running historical scripts.')

if __name__ == '__main__':
    main()
