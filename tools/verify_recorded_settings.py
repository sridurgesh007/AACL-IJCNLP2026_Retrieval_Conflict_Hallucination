"""Read-only comparison of release profiles with locally held historical run metadata.

--archive-root is the original experiment workspace containing medqa_final_v3/
and hallucination_experiment/. This tool does not run models or copy outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def compare(profile, original):
    current=profile['args']
    differences={key:{'recorded':value,'release':current.get(key,'MISSING')} for key,value in original.items() if current.get(key,'MISSING')!=value}
    return differences


def verify(archive):
    catalog=json.loads((ROOT/'configs/experiments.json').read_text())
    results=[]
    for name,profile in catalog.items():
        meta=profile.get('source_metadata')
        if not meta:
            results.append({'profile':name,'status':'not_verified_against_recorded_run','reason':'Preparation/helper/new-run profile; no archived CLI manifest is attached.'})
            continue
        path=archive/meta['source'] if meta.get('source_root')=='experiment' else archive/'hallucination_experiment'/meta['source']
        if not path.is_file():
            results.append({'profile':name,'status':'source_unavailable','source':meta['source']})
            continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        obj=json.loads(path.read_text(encoding='utf-8'))
        original=obj.get('metadata',{}).get('cli_args')
        if not isinstance(original,dict):raise ValueError(f'No recorded CLI arguments: {path}')
        differences=compare(profile,original)
        hash_matches=digest==meta['sha256']
        results.append({'profile':name,'status':'match' if not differences and hash_matches else 'mismatch','source':meta['source'],'source_hash_matches':hash_matches,'recorded_arguments_checked':len(original),'differences':differences,'release_explicit_additions':{k:v for k,v in profile['args'].items() if k not in original}})
    return {'scope':'Recorded CLI settings and source hashes only. No assertion of identical upstream data, model/tokenizer revision, historical package environment or GPU outputs.','profiles':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=verify(args.archive_root)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    for row in report['profiles']:
        if row['status'] in ('match','mismatch','source_unavailable'):print(row['profile']+': '+row['status'])
    return 1 if any(r['status'] in ('mismatch','source_unavailable') for r in report['profiles']) else 0


if __name__=='__main__':raise SystemExit(main())
