"""Install small frozen selection/configuration metadata into ignored work directories."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true',help='Verify selections without writing')
    args=p.parse_args()
    frozen=ROOT/'configs/frozen'
    matched=json.loads((frozen/'medqa_matched_subset.json').read_text())
    audit=json.loads((frozen/'medqa_label_audit.json').read_text())
    ids=set(matched['matched_item_ids'])
    clean={r['item_id'] for r in audit['items'] if r.get('label_status')=='clean'} & ids
    if len(ids)!=473 or len(clean)!=40:
        raise RuntimeError(f'Unexpected frozen selections: matched={len(ids)}, clean={len(clean)}')
    print(f'Frozen MedQA selection verified: {len(ids)} matched IDs; {len(clean)} clean IDs.')
    if args.verify:return
    mappings={
        'medqa_matched_subset.json':'matched_subset.json',
        'medqa_label_audit.json':'label_audit.json',
        'medqa_headline.json':'pr_sweep/headline.json',
        'medqa_best_cell.json':'decomposition/best_cell.json',
    }
    for source,destination in mappings.items():
        target=ROOT/'implementation/medrag/results/medqa_final_v3'/destination
        if target.exists():
            if target.read_bytes()!=(frozen/source).read_bytes():
                raise RuntimeError(f'Refusing to overwrite different existing metadata: {target}')
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(frozen/source,target)
    print('Metadata installed. Download/import data separately; no benchmark rows were bundled.')


if __name__=='__main__':main()
