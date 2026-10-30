import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run
import prepare_data
import audit_medqa
import verify_recorded_settings


class ReleaseTests(unittest.TestCase):
    def test_exact_mcnemar_clean_denominator(self):
        self.assertAlmostEqual(audit_medqa.stats.mcnemar(27,0)['p_value'],1.4901161193847656e-8)
        self.assertAlmostEqual(audit_medqa.stats.mcnemar(8,3)['p_value'],0.2265625)
        self.assertLessEqual(audit_medqa.stats.mcnemar(0,0)['p_value'],1)

    def test_wilson_counterfact(self):
        lo,hi=audit_medqa.stats.wilson_ci(48,279)
        self.assertAlmostEqual(lo,0.132288,places=5)
        self.assertAlmostEqual(hi,0.220707,places=5)

    def test_join_does_not_silently_drop_ids(self):
        a={'a':{'correct':False,'in_clean_subset':True}}
        b={'b':{'correct':True,'in_clean_subset':True}}
        with self.assertRaises(ValueError):audit_medqa.compare(a,b,'in_clean_subset')

    def test_restoration_and_harm_denominators(self):
        base={str(i):{'correct':i>=2,'in_matched_subset':True} for i in range(5)}
        post={str(i):{'correct':i!=3,'in_matched_subset':True} for i in range(5)}
        result=audit_medqa.compare(base,post,'in_matched_subset')
        self.assertEqual(result['restoration']['denominator'],2)
        self.assertEqual(result['restoration']['numerator'],2)
        self.assertEqual(result['harm']['denominator'],3)
        self.assertEqual(result['harm']['numerator'],1)

    def test_duplicate_condition_ids_fail(self):
        scratch=ROOT/'run_records'/'test_tmp'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            path=Path(directory)/'rows.jsonl'
            path.write_text((' {"item_id":"x","condition_correct":true}\n')*2)
            with self.assertRaises(ValueError):audit_medqa.load_condition(path)

    def test_data_rejects_duplicate_counterfact_ids(self):
        row={'id':1,'question':'synthetic test','evidence':'test','answer_liquid':'a','answer_solid':'b'}
        with self.assertRaises(ValueError):prepare_data.validate('counterfact',[row,row])

    def test_every_profile_dry_run(self):
        profiles=json.loads((ROOT/'configs/experiments.json').read_text())
        for name,profile in profiles.items():
            with self.subTest(name=name):run.make_command(profile)

    def test_unknown_flags_fail(self):
        profile={'family':'medrag','script':'32_run_medqa_killtest.py','args':{'invented_flag':1}}
        with self.assertRaises(ValueError):run.make_command(profile)

    def test_historical_setting_mismatch_is_detected(self):
        differences=verify_recorded_settings.compare({'args':{'seed':1,'max_new_tokens':5}},{'seed':1234,'max_new_tokens':5})
        self.assertEqual(set(differences),{'seed'})

    def test_published_settings_audit_has_no_missing_recorded_run(self):
        report=json.loads((ROOT/'docs/recorded_settings_audit.json').read_text())
        verified=[row for row in report['profiles'] if row['status']=='match']
        self.assertEqual(len(verified),12)
        self.assertFalse(any(row['status'] in ('mismatch','source_unavailable') for row in report['profiles']))

    def test_disruption_priority_partitions_overlap(self):
        path=ROOT/'implementation/medrag/scripts/83_run_matched_head_controls.py'
        tree=ast.parse(path.read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_exclusive_disruption_outcome')
        namespace={'VALID_LETTERS':('A','B','C','D')}
        exec('from __future__ import annotations\n'+ast.unparse(function),namespace)
        f=namespace[function.name]
        # dest_base == source_correct: unchanged has precedence, so it cannot be counted twice.
        outcomes=[f(post=post,dest_base_letter='A',source_correct_letter='A',dest_correct_letter='B') for post in ('A','B','C',None)]
        self.assertEqual(outcomes[0],'unchanged')
        self.assertEqual(len(outcomes),4)
        self.assertEqual(len(set(outcomes)),3)


if __name__=='__main__':unittest.main()
