"""Portable entry point; dry-run by default, execute only with --execute.

Historical scripts resolve paths relative to their implementation family.
This runner fixes cwd/PYTHONPATH, validates explicit inputs, and logs the command.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('medrag', 'Evidence_aware', 'conflictqa')


def cli_contract(script, _seen=None):
    _seen=set() if _seen is None else _seen
    script=Path(script)
    if script in _seen:return {}
    _seen.add(script)
    result = {}
    for node in ast.walk(ast.parse(Path(script).read_text(encoding='utf-8-sig'))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value,str) and a.value.startswith('--')]
        fields = {}
        for kw in node.keywords:
            try:
                fields[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                pass
        for flag in flags:
            result[flag[2:]] = fields
    if not result:
        # Thin historical wrappers delegate their parser to a sibling script.
        for node in ast.walk(ast.parse(script.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.endswith('.py'):
                candidate=script.with_name(Path(node.value).name)
                if candidate.exists() and candidate!=script:
                    inherited=cli_contract(candidate,_seen)
                    if inherited:return inherited
    return result


def make_command(profile, overrides=None):
    family = profile['family']
    if family not in FAMILIES:
        raise ValueError(f'Unknown family: {family}')
    cwd = ROOT/'implementation'/family
    script = cwd/'scripts'/profile['script']
    if not script.is_file():
        raise FileNotFoundError(script)
    args = dict(profile.get('args', {}))
    args.update(overrides or {})
    contract = cli_contract(script)
    unknown = set(args)-set(contract)
    if unknown:
        raise ValueError(f'Unsupported arguments for {script.name}: {sorted(unknown)}')
    missing = [key for key, spec in contract.items() if spec.get('required') and key not in args]
    if missing:
        raise ValueError(f'Missing required arguments: {missing}')
    command = [sys.executable, str(script)]
    for key,value in args.items():
        if value is None:
            continue
        if isinstance(value, bool) and contract[key].get('action') in ('store_true','store_false'):
            if value == (contract[key]['action']=='store_true'):
                command.append('--'+key)
        else:
            command.append('--'+key+'='+str(value))
    return cwd, command, args


def check_inputs(profile, cwd, args):
    missing = []
    for key in profile.get('input_args', []):
        value = args.get(key)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        path = path if path.is_absolute() else cwd/path
        if not path.exists():
            missing.append(str(path.resolve()))
    if missing:
        raise FileNotFoundError('Missing inputs; run the prerequisite stage or import frozen inputs:\n'+'\n'.join(missing))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',nargs='?')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--check',action='store_true',help='Validate declared input files without loading a model')
    args=parser.parse_args(argv)
    catalog=json.loads((ROOT/'configs/experiments.json').read_text())
    if args.list:
        for name,p in catalog.items(): print(f'{name:38} {p["description"]}')
        return 0
    if args.profile not in catalog:
        parser.error('Choose a profile from --list')
    overrides={}
    for value in args.set:
        key,sep,val=value.partition('=')
        if not sep: parser.error('--set requires KEY=VALUE')
        overrides[key]=val
    profile=catalog[args.profile]
    cwd,command,values=make_command(profile,overrides)
    print('Working directory:',cwd)
    print('Command:',shlex.join(command))
    print('Scope:',profile.get('scope','regenerated method run; compare input hashes before claiming exact paper reproduction'))
    if args.check or args.execute:
        check_inputs(profile,cwd,values)
    if not args.execute:
        return 0
    env=os.environ.copy()
    paths=[str(ROOT/'implementation'/f/'src') for f in FAMILIES]
    env['PYTHONPATH']=os.pathsep.join(paths+([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('MPLBACKEND','Agg')
    env.setdefault('PYTHONUNBUFFERED','1')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record_dir=ROOT/'run_records'/f'{args.profile}_{stamp}'
    record_dir.mkdir(parents=True)
    packages={}
    for name in ['torch','transformers','numpy','scipy','datasets','matplotlib','pandas','accelerate']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    record={'profile':args.profile,'cwd':str(cwd),'command':command,'args':values,'packages':packages,'python':sys.version,'started_utc':stamp,'exit_code':None}
    record_path=record_dir/'invocation.json'
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    try:
        code=subprocess.run(command,cwd=cwd,env=env).returncode
    except KeyboardInterrupt:
        code=130
    record['exit_code']=code
    record_path.write_text(json.dumps(record,indent=2)+'\n')
    return code


if __name__=='__main__':
    raise SystemExit(main())
