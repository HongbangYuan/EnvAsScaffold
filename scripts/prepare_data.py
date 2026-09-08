"""Create text-only scheduling rows; tasks and rewards come from the environments."""
import argparse
from pathlib import Path
from datasets import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('data/text'))
    parser.add_argument('--train-size', type=int, default=16)
    parser.add_argument('--val-size', type=int, default=128)
    args = parser.parse_args()
    if min(args.train_size, args.val_size) < 1:
        parser.error('Dataset sizes must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, count in [('train', args.train_size), ('test', args.val_size)]:
        rows = [{'data_source': 'text', 'prompt': [{'role': 'user', 'content': ''}],
                 'ability': 'agent', 'extra_info': {'split': split, 'index': index}}
                for index in range(count)]
        Dataset.from_list(rows).to_parquet(str(args.output_dir / f'{split}.parquet'))


if __name__ == '__main__':
    main()
