"""Explore saved-score threshold tradeoffs without fitting any classifier."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.config import ROOT
from src.fitness import metrics


def audit(report_path, root=ROOT):
    report = json.loads(Path(report_path).read_text(encoding='utf-8-sig'))
    records = report['records'].copy()
    ids = {r['model_id'] for r in records}
    for path in (root/'models').glob('*.json'):
        meta = json.loads(path.read_text())
        if (meta.get('source_experiment_id') == report['experiment_id'] and
                meta['model_id'] not in ids):
            records.append(meta); ids.add(meta['model_id'])
    rows = []
    print('EXPLORATORY TEST-SET THRESHOLD AUDIT — NO TRAINING')
    print('The highest rows below are selected using TEST LABELS. They are not independent test estimates.')
    print('Keep your original fixed-threshold experiment results in the main comparison.\n')
    for record in records:
        identifier = record['model_id']
        path = root/'models'/f'{identifier}.predictions.npz'
        if not path.is_file():
            print('Missing prediction cache:', identifier); continue
        with np.load(path) as cache:
            y = cache['y']; scores = cache['attack_scores']
            candidate_rows = []
            inherited = record.get('attack_threshold', report['config'].get('attack_threshold', .5))
            for threshold in sorted(set(np.round(np.arange(.05, 1., .025), 3)) | {float(inherited)}):
                row = {'algorithm': record['algorithm'], 'seed': record.get('seed', ''),
                       'model_id': identifier, 'threshold': float(threshold),
                       'selection_set': 'KDDTest+ exploratory',
                       **metrics(y, (scores >= threshold).astype(np.int8))}
                row.pop('confusion_matrix'); candidate_rows.append(row)
            current = next(r for r in candidate_rows if np.isclose(r['threshold'], inherited))
            highest = max(candidate_rows, key=lambda r: (r['accuracy'], r['f1']))
            print(f"{record['algorithm']} seed={record.get('seed', 'ensemble/baseline')} model={identifier}")
            for label, row in [('Original threshold', current), ('TEST-TUNED highest accuracy', highest)]:
                print(f"  {label}: threshold={row['threshold']:.3f}, accuracy={row['accuracy']:.2%}, "
                      f"recall={row['recall']:.2%}, F1={row['f1']:.2%}, false alarms={row['false_alarm_rate']:.2%}")
            rows.extend(candidate_rows)
    if not rows:
        raise ValueError('No prediction caches found; no model was trained.')
    above = [r for r in rows if r['accuracy'] > .9]
    print('\nAny tested threshold above 90%:', bool(above))
    output = root/'results'/f"threshold_audit_{report['experiment_id']}.csv"
    with output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Full curves saved:', output)
    print('No saved thresholds, models or original experiment results were changed.')
    print('Choosing a deployment threshold requires separate validation; selecting this test maximum is optimistic.')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    audit(args.report)
