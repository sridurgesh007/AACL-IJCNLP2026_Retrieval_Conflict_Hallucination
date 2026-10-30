"""CPU-only source, profile, dependency-layout and repository hygiene checks."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run


def main():
    files=list((ROOT/'implementation').rglob('*.py'))+list((ROOT/'tools').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for path in files:compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
    profiles=json.loads((ROOT/'configs/experiments.json').read_text())
    for name,profile in profiles.items():
        run.make_command(profile)
        if profile['script']=='63_run_popqa_intervention.py':
            if 'input_path' in profile['args'] and 'conflict_path' in profile['args']:raise AssertionError(f'Conflicting alias flags: {name}')
    subprocess.run([sys.executable,str(ROOT/'tools/bootstrap.py'),'--verify'],check=True)
    # Check actual staged/tracked payload rather than ignored local experiment files.
    tracked=subprocess.run(['git','ls-files','--cached'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines()
    forbidden={'.pt','.pth','.bin','.safetensors','.parquet','.pqt','.tsv','.csv','.jsonl','.npy','.npz','.pkl','.pickle','.png','.pdf','.zip'}
    for name in tracked:
        p=Path(name)
        if p.suffix in forbidden or any(s in p.parts for s in ['data','results','cache','checkpoints','logs']):raise AssertionError(f'Generated/benchmark payload staged: {name}')
        if (ROOT/p).is_file() and (ROOT/p).stat().st_size>2_000_000:raise AssertionError(f'Oversized file: {name}')
    for path in (ROOT/'implementation').rglob('*.py'):
        text=path.read_text(encoding='utf-8-sig')
        if '/home/s/sri007' in text:raise AssertionError(f'Personal cluster path: {path}')
    print(f'PASS: {len(files)} Python sources compile; {len(profiles)} profiles satisfy CLI contracts; frozen IDs verified; {len(tracked)} tracked files checked.')


if __name__=='__main__':main()
