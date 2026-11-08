#!/usr/bin/env python3
"""Compare the corrected official pair scripts against a saved single-copy run."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    c0 = 'data/iter0/0_chosen_judgement.json'
    f0 = 'data/iter0/0_future_judgement.json'
    c1 = 'data/iter1/1_chosen_judgement.json'
    r1 = 'data/iter1/1_rejected_judgement.json'
    f1 = 'data/iter1/1_future_judgement.json'
    cases = [('iter0_D1', 'past', c0, c0, None, [], 'data/iter0/D1/train.json', 5000),
             ('iter0_D2', 'future', c0, c0, f0, [], 'data/iter0/D2/train.json', 5000),
             ('iter1_D1', 'past', c1, r1, None, [], 'data/iter1/D1/train.json', 5000)]
    for model, flags, count in [('M2', [], 5000), ('M2_gt_keep', ['--strict'], 5000),
                                ('M2_ge_filter', ['--filter'], 4770), ('M2_gt_filter', ['--strict', '--filter'], 4770)]:
        cases.append((model, 'future', c1, r1, f1, flags, f'data/four-settings/{model}/D2/train.json', count))
    report = {'status': 'PASS', 'work': str(args.work), 'pairs': [], 'loading': [],
              'versions': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'trl', 'datasets', 'accelerate']}}
    with tempfile.TemporaryDirectory(prefix='tsr-official-pairs-') as tmp:
        for name, mode, chosen, rejected, future, flags, saved, count in cases:
            target = Path(tmp) / (name + '.jsonl')
            command = [sys.executable, str(ROOT / f'settings/tsr/reformat2DPO_{mode}.py'),
                       '--input_path_chosen', str(args.work / chosen), '--input_path_rejected', str(args.work / rejected),
                       '--output_path', str(target), *flags]
            if future:
                command += ['--input_path_future', str(args.work / future)]
            subprocess.run(command, check=True, capture_output=True)
            payload = target.read_bytes()
            assert payload == (args.work / saved).read_bytes(), name
            assert len(payload.splitlines()) == count
            report['pairs'].append({'name': name, 'rows': count, 'byte_identical': True,
                                    'sha256': hashlib.sha256(payload).hexdigest()})

    # Exercise the actual loader against real legacy dirs containing official/train.json.
    package = types.ModuleType('_tsr_archive_alignment')
    package.__path__ = [str(ROOT / 'alignment-handbook/src/alignment')]
    sys.modules[package.__name__] = package
    data = importlib.import_module(package.__name__ + '.data')
    for relative, count in [('data/iter0/D1', 5000), ('data/iter0/D2', 5000),
                            ('data/iter1/D1', 5000), ('data/four-settings/M2_ge_filter/D2', 4770)]:
        path = args.work / relative
        dataset = data.get_datasets({str(path): 1.0}, splits=['train'],
                                   columns_to_keep=['prompt', 'chosen', 'rejected'],
                                   shuffle=False, local_train_file='train.json')['train']
        assert len(dataset) == count
        report['loading'].append({'directory': relative, 'loaded_rows': len(dataset),
                                   'global_batch_32_steps': count // 32})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
