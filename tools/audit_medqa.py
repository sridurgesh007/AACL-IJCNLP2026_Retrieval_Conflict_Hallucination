"""Recompute paired MedQA statistics from per-item rows, with explicit denominators."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_stats',ROOT/'implementation/medrag/src/medrag/utils/stats.py')
stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)


def load_condition(path):
    out={}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line)
        item=str(row['item_id'])
        if item in out:raise ValueError(f'Duplicate item_id in {path}: {item}')
        if row.get('skipped',False):raise ValueError(f'Skipped item {item}; resolve before paired reporting')
        value=row.get('condition_correct',row.get('ft_correct'))
        if not isinstance(value,bool):raise ValueError(f'Expected boolean condition_correct/ft_correct at {item}')
        out[item]={**row,'correct':value}
    if not out:raise ValueError(f'Empty condition: {path}')
    return out


def compare(base,current,population):
    a={k:r for k,r in base.items() if r.get(population,False)}
    b={k:r for k,r in current.items() if r.get(population,False)}
    if not a or set(a)!=set(b):raise ValueError(f'Nonempty exact ID match required for {population}')
    ids=sorted(a)
    wrong=[k for k in ids if not a[k]['correct']]
    right=[k for k in ids if a[k]['correct']]
    restored=sum(b[k]['correct'] for k in wrong)
    harmed=sum(not b[k]['correct'] for k in right)
    def proportion(k,n):
        return {'numerator':k,'denominator':n,'rate':k/n if n else None,'wilson95':stats.wilson_ci(k,n) if n else None}
    return {'population':population,'n_paired':len(ids),'restoration':proportion(restored,len(wrong)),'harm':proportion(harmed,len(right)),'mcnemar':stats.mcnemar_from_paired([a[k]['correct'] for k in ids],[b[k]['correct'] for k in ids])}


def audit(directory):
    directory=Path(directory)
    conditions={p.name:load_condition(p/'results.jsonl') for p in sorted(directory.glob('C[1-7]')) if (p/'results.jsonl').exists()}
    if 'C1' not in conditions or 'C4' not in conditions:raise ValueError('Need C1 and C4 condition rows')
    result={}
    for name,rows in conditions.items():
        baseline='C2' if name=='C5' else 'C1'
        if baseline not in conditions:raise ValueError(f'Missing baseline {baseline}')
        result[name]={
            'restoration_baseline':baseline,
            'clean':compare(conditions[baseline],rows,'in_clean_subset'),
            'matched':compare(conditions[baseline],rows,'in_matched_subset'),
            'versus_C4_clean':compare(conditions['C4'],rows,'in_clean_subset'),
            'versus_C4_matched':compare(conditions['C4'],rows,'in_matched_subset'),
        }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    text=json.dumps(audit(args.results_dir),indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n')
    print(text)


if __name__=='__main__':main()
