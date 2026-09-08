"""Import an experiment's BFCL data snapshot without copying any logs or code."""
import argparse
import hashlib
import json
from pathlib import Path

CATEGORIES = ('base', 'long_context', 'miss_func', 'miss_param')
TOOLS = ('gorilla_file_system', 'math_api', 'message_api', 'posting_api',
         'ticket_api', 'trading_bot', 'travel_booking', 'vehicle_control')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True, help='Directory containing BFCL_v4_*.json')
    parser.add_argument('--revision', required=True, help='Public revision or neutral snapshot label; no private paths')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / 'agent_system/environments/env_package/bfcl/berkeley_function_call_leaderboard/bfcl_eval/data'
    paths = [f'{prefix}BFCL_v4_multi_turn_{category}.json'
             for category in CATEGORIES for prefix in ('', 'possible_answer/')]
    paths += [f'multi_turn_func_doc/{tool}.json' for tool in TOOLS]
    pending = []
    for relative in paths:
        payload = (args.source_dir / relative).read_bytes()
        entries = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if 'multi_turn_func_doc/' not in relative:
            category = Path(relative).stem.removeprefix('BFCL_v4_')
            expected = {f'{category}_{index}' for index in range(200)}
            ids = [entry['id'] for entry in entries]
            if len(ids) != 200 or set(ids) != expected:
                raise ValueError(f'Unexpected IDs in {relative}; expected exactly indices 0..199')
        pending.append((relative, payload))
    # Validate all files before replacing the bundled public fallback.
    for relative, payload in pending:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    provenance = {'revision': args.revision, 'status': 'User-provided experiment snapshot',
                  'files': [{'path': relative, 'sha256': hashlib.sha256(payload).hexdigest()}
                            for relative, payload in pending]}
    (root / 'docs/bfcl_data_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Imported 16 validated BFCL data files; run scripts/static_audit.py before publication.')


if __name__ == '__main__':
    main()
