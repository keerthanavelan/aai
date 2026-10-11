"""Read completed experiment statistics. Standard library only; never trains."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent


def load_report(path):
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    records = report.get('records', [])
    if not records or any('test_metrics' not in r for r in records):
        raise ValueError(f'{path.name}: not a completed experiment report')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, help='One specific experiment JSON')
    args = parser.parse_args()
    paths = [args.file] if args.file else sorted(
        (ROOT / 'results').glob('experiment_*.json'),
        key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise SystemExit('No experiment_*.json found in results/. No training was started.')
    reports = []
    for path in paths:
        try:
            reports.append((path, load_report(path)))
        except (ValueError, OSError) as exc:
            print(f'Skipping {path}: {exc}')
    if not reports:
        raise SystemExit('No readable completed experiment report found.')
    print('COMPLETED REPORTS (newest file first; configurations are kept separate)')
    for path, report in reports:
        cfg = report.get('config', {})
        print(f"{path.name}: seeds={report.get('seeds')}, population={cfg.get('population')}, "
              f"iterations={cfg.get('iterations')}, train_rows={cfg.get('train_rows')}, "
              f"objective={cfg.get('objective', 'accuracy')}")
    for path, report in reports:
        print(f'\nREPORT: {path.name}')
        cfg = report.get('config', {})
        if cfg.get('train_rows', 0) <= 1000 and cfg.get('iterations', 0) <= 2:
            print('SMALL VERIFICATION BUDGET: do not present this as the final experiment.')
        print('Test metrics: mean +/- sample SD in percentage points; SD unavailable for one run.')
        print(f"{'Method':12} {'Runs':>4} {'Accuracy':>17} {'Attack recall':>17} {'F1':>17} {'False alarm rate':>17} {'Features':>10}")
        for method in ['full', 'chi_square', 'ga', 'pso']:
            rows = [r for r in report['records'] if r['algorithm'] == method]
            if not rows:
                continue
            def summary(key):
                values = [100 * r['test_metrics'][key] for r in rows]
                deviation = f'{stdev(values):.2f}' if len(values) > 1 else 'n/a'
                return f'{mean(values):.2f} +/- {deviation}'
            print(f"{method:12} {len(rows):4} {summary('accuracy'):>17} {summary('recall'):>17} "
                  f"{summary('f1'):>17} {summary('false_alarm_rate'):>17} "
                  f"{mean(r['feature_count'] for r in rows):10.1f}")
            chosen = min(rows, key=lambda r: r['best_fitness'])
            identifier = chosen.get('model_id', 'not saved')
            exists = (ROOT / 'models' / f'{identifier}.joblib').is_file()
            print(f"  Validation-selected representative: seed={chosen.get('seed', 'baseline')}, "
                  f"fitness={chosen['best_fitness']:.6f}, model={identifier}, model_file_present={exists}")
            print(f"  Selected inputs: {', '.join(chosen['selected_features'])}")
            print(f"  Representative test: missed attacks={chosen['test_metrics']['missed_attacks']}, "
                  f"false alarms={chosen['test_metrics']['false_alarms']}")
        print('No files were changed and no model was trained.')


if __name__ == '__main__':
    main()
