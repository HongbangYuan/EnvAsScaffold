"""Static release checks. Does not import or execute research code."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL = ROOT / 'agent_system/environments/env_package/bfcl/berkeley_function_call_leaderboard'
SCIWORLD = ROOT / 'agent_system/environments/env_package/sciworld'
CATEGORIES = ('base', 'long_context', 'miss_func', 'miss_param')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deny-pattern', action='append', default=[], help='Additional private markers; not saved in report')
    parser.add_argument('--report', type=Path, default=ROOT / 'outputs/static_audit.json')
    args = parser.parse_args()
    errors, warnings = [], []
    counts = {'python_files': 0, 'shell_files': 0, 'yaml_files': 0, 'toml_files': 0, 'archives': 0}
    module_roots = {'verl': ROOT, 'agent_system': ROOT, 'gigpo': ROOT,
                    'bfcl_eval': BFCL, 'scienceworld': SCIWORLD / 'ScienceWorld'}
    private_pattern = re.compile('|'.join([
        r'/' + r'Users/[^/\s]+/', r'/' + r'mnt/hdfs/',
        r'byted' + r'-wandb', r'(?i:wandb_base_url)\s*=\s*[\"\x27]?https?://(?!api\.wandb\.ai)',
        r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
        r'\bsk-[A-Za-z0-9_-]{24,}',
        *args.deny_pattern,
    ]), re.IGNORECASE)
    provenance = json.loads((ROOT / 'docs/bfcl_data_provenance.json').read_text())
    public_fixtures = {str((BFCL / 'bfcl_eval/data' / entry['path']).relative_to(ROOT)): entry['sha256']
                       for entry in provenance['files']}
    credential = re.compile(r'''(?i)(?:api_key|access_token|auth_token|password)\s*=\s*["']([^"']{12,})["']''')
    try:
        import yaml
        parse_yaml = yaml.safe_load
    except ImportError:
        try:
            from ruamel.yaml import YAML
            parse_yaml = YAML(typ='safe').load
        except ImportError:
            parse_yaml = None
            warnings.append('YAML parser unavailable; YAML parsing skipped')
    try:
        import tomllib
    except ImportError:
        tomllib = None
        warnings.append('TOML parser unavailable; TOML parsing skipped (requires Python 3.11+)')
    for path in sorted(ROOT.rglob('*')):
        relative = str(path.relative_to(ROOT))
        if any(part in ('.git', '__pycache__', '.venv') for part in path.relative_to(ROOT).parts):
            continue
        if path.is_symlink():
            errors.append(f'{relative}: symlink is not allowed in release')
            continue
        if not path.is_file() or path.resolve() == args.report.resolve():
            continue
        if path.suffix in ('.pyc', '.ipynb', '.log') or path.name in ('.DS_Store', '.env'):
            errors.append(f'{relative}: generated/private artifact')
        try:
            text = path.read_text()
        except UnicodeError:
            text = None
        if text is not None:
            for index, line in enumerate(text.splitlines(), 1):
                if private_pattern.search(line):
                    errors.append(f'{relative}:{index}: private marker (value withheld)')
                # Comments are not executable credentials; placeholder values are also allowed.
                match = credential.search(line) if not line.lstrip().startswith('#') else None
                known_fixture = provenance.get('repository') == 'https://github.com/ShishirPatil/gorilla' and provenance.get('revision') == '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8' and relative in public_fixtures and hashlib.sha256(path.read_bytes()).hexdigest() == public_fixtures[relative]
                if match and not known_fixture and not any(word in match[1].lower() for word in ('example', 'your_', 'dummy', 'placeholder', 'token-abc123')):
                    errors.append(f'{relative}:{index}: possible credential literal (value withheld)')
        try:
            if path.suffix == '.py':
                tree = ast.parse(text, filename=relative)
                compile(tree, relative, 'exec')
                counts['python_files'] += 1
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'breakpoint':
                        errors.append(f'{relative}:{node.lineno}: active breakpoint')
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                        modules = [node.module]
                    else:
                        continue
                    for module in modules:
                        root = module_roots.get(module.split('.')[0])
                        if root:
                            target = root / module.replace('.', '/')
                            if not target.is_dir() and not target.with_suffix('.py').is_file():
                                errors.append(f'{relative}:{node.lineno}: missing local module {module}')
            elif path.suffix == '.sh':
                result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
                if result.returncode:
                    errors.append(f'{relative}: shell syntax error')
                counts['shell_files'] += 1
            elif path.suffix in ('.yaml', '.yml') and parse_yaml:
                parse_yaml(text)
                counts['yaml_files'] += 1
            elif path.suffix == '.toml' and tomllib:
                tomllib.loads(text)
                counts['toml_files'] += 1
            elif path.suffix in ('.jar', '.zip'):
                with zipfile.ZipFile(path) as archive:
                    if archive.testzip() is not None:
                        errors.append(f'{relative}: corrupt archive')
                    for name in archive.namelist():
                        payload = archive.read(name).decode('utf-8', errors='ignore')
                        if private_pattern.search(name + payload):
                            errors.append(f'{relative}: private marker in archive member')
                counts['archives'] += 1
        except Exception as exc:
            errors.append(f'{relative}: {type(exc).__name__}: static parse/check failed')
    splits = {}
    for path in (SCIWORLD / 'variations_idx').glob('*.json'):
        data = json.loads(path.read_text())
        train, test = set(map(tuple, data['train'])), set(map(tuple, data['test']))
        if train & test:
            errors.append(f'{path.name}: train/test overlap')
        splits[path.stem] = {'train': len(train), 'test': len(test), 'overlap': len(train & test)}
    data_root = BFCL / 'bfcl_eval/data'
    mapping_path = BFCL / 'bfcl_eval/constants/executable_backend_config.py'
    tree = ast.parse(mapping_path.read_text())
    docs_mapping = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MULTI_TURN_FUNC_DOC_FILE_MAPPING' for t in node.targets))
    bfcl_counts = {}
    for category in CATEGORIES:
        name = f'multi_turn_{category}'
        expected = {f'{name}_{index}' for index in range(200)}
        for prefix in ('', 'possible_answer/'):
            path = data_root / f'{prefix}BFCL_v4_{name}.json'
            if not path.is_file():
                errors.append(f'BFCL {prefix}{name}: missing data')
                continue
            entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            ids = [entry['id'] for entry in entries]
            if len(ids) != 200 or set(ids) != expected:
                errors.append(f'BFCL {prefix}{name}: wrong IDs or duplicates')
            if not prefix:
                for entry in entries:
                    primary = {'GorillaFileSystem', 'VehicleControlAPI'} if int(entry['id'].rsplit('_', 1)[1]) < 100 else {'TradingBot', 'TravelAPI'}
                    opposite = {'TradingBot', 'TravelAPI', 'GorillaFileSystem', 'VehicleControlAPI'} - primary
                    classes = set(entry['involved_classes'])
                    if not classes & primary or classes & opposite:
                        errors.append(f'BFCL {entry["id"]}: unexpected train/test domain')
                    for cls in classes:
                        filename = docs_mapping.get(cls)
                        if not filename or not (data_root / 'multi_turn_func_doc' / filename).is_file():
                            errors.append(f'BFCL {entry["id"]}: missing tool definition')
                        elif not (BFCL / 'bfcl_eval/eval_checker/multi_turn_eval/func_source_code' / filename.replace('.json', '.py')).is_file():
                            errors.append(f'BFCL {entry["id"]}: missing dynamic tool backend')
                bfcl_counts[category] = len(entries)
    provenance = json.loads((ROOT / 'docs/bfcl_data_provenance.json').read_text())
    for entry in provenance['files']:
        path = data_root / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            errors.append(f'BFCL {entry["path"]}: provenance checksum mismatch')
    report = {'mode': 'static only; no research modules, training, environments or installers executed',
              'counts': counts, 'sciworld_splits': splits, 'bfcl_category_counts': bfcl_counts,
              'errors': sorted(set(errors)), 'warnings': warnings}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
