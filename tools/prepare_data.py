"""Validate and import user-provided prepared inputs. Never downloads implicitly."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
SCHEMAS={
    'medqa':('medrag/data/medqa_conflict_500.jsonl',{'question','options','context','correct_answer','parametric_answer'}),
    'popqa':('conflictqa/data/popqa_conflict_3141_natural.jsonl',{'question','context','correct_answer','parametric_answer'}),
    'counterfact':('Evidence_aware/data/counterfact_solvent.json',{'id','question','evidence','answer_liquid','answer_solid'}),
}


def validate(kind,rows):
    if not rows:raise ValueError('Input is empty')
    required=SCHEMAS[kind][1]
    seen=set()
    for index,row in enumerate(rows):
        if not isinstance(row,dict) or required-set(row):
            raise ValueError(f'Row {index}: required fields {sorted(required)}')
        if kind=='medqa':
            if not isinstance(row['options'],dict) or not {'A','B','C','D'}.issubset(row['options']):raise ValueError(f'Row {index}: missing A-D options')
            if row['correct_answer'] not in set('ABCD') or row['parametric_answer'] not in set('ABCD'):raise ValueError(f'Row {index}: invalid answer letters')
        if kind=='counterfact':
            if row['id'] in seen:raise ValueError(f'Duplicate CounterFact ID: {row["id"]}')
            seen.add(row['id'])
        if not str(row['question']).strip():raise ValueError(f'Row {index}: empty question')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=SCHEMAS)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--expect-sha256',default='')
    p.add_argument('--expect-count',type=int)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    raw=args.input.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    if args.expect_sha256 and digest!=args.expect_sha256:raise ValueError('Input SHA256 does not match the expected paper input')
    content=raw.decode('utf-8-sig')
    rows=json.loads(content) if args.kind=='counterfact' else [json.loads(s) for s in content.splitlines() if s.strip()]
    validate(args.kind,rows)
    if args.expect_count is not None and len(rows)!=args.expect_count:raise ValueError(f'Expected {args.expect_count} rows, got {len(rows)}')
    print(json.dumps({'kind':args.kind,'count':len(rows),'sha256':digest},indent=2))
    if args.validate_only:return
    target=ROOT/'implementation'/SCHEMAS[args.kind][0]
    if target.exists() and target.read_bytes()!=raw:raise FileExistsError(f'Refusing to replace different data: {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.resolve()!=args.input.resolve():shutil.copyfile(args.input,target)
    target.with_suffix('.manifest.json').write_text(json.dumps({'kind':args.kind,'count':len(rows),'sha256':digest,'source_name':args.input.name},indent=2)+'\n')
    print('Imported:',target)


if __name__=='__main__':main()
