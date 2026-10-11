"""Build and evaluate a fixed ensemble; no fit(), GA or PSO search is run."""
import argparse
import json
import time
import uuid
from pathlib import Path
import joblib
import numpy as np
from src.config import ROOT
from src.preprocess import FEATURE_NAMES, NUMERIC, CATEGORICAL, transform_features
from src.saved_ensemble import SavedSubsetEnsemble
from src.evaluate import classify, security_breakdown
from src.fitness import metrics
from src.storage import atomic_json


def build(report_path, root=ROOT, method='ga'):
    report = json.loads(Path(report_path).read_text(encoding='utf-8-sig'))
    records = sorted([r for r in report['records'] if r['algorithm'] == method],
                     key=lambda r: r['seed'])
    if len(records) < 2:
        raise ValueError('At least two completed models of the same method are required.')
    threshold = float(report['config'].get('attack_threshold', .5))
    print(f'Combining ALL {len(records)} {method.upper()} runs equally, threshold={threshold}.', flush=True)
    print('No training. No test-based weight, member or threshold tuning.', flush=True)
    members = []
    score_arrays = []
    truth = None
    for record in records:
        identifier = record['model_id']
        print('Loading saved model ' + identifier, flush=True)
        saved = joblib.load(root / 'models' / f'{identifier}.joblib')
        if not np.isclose(saved.get('attack_threshold', .5), threshold):
            raise ValueError('Member threshold differs from experiment configuration.')
        if members and any(not np.array_equal(a, b) for a, b in zip(
                members[0]['encoder'].categories_, saved['encoder'].categories_)):
            raise ValueError('Member encoders differ; cannot share transformed columns.')
        members.append(saved)
        with np.load(root / 'models' / f'{identifier}.predictions.npz') as data:
            if truth is None:
                truth = data['y'].copy()
            elif not np.array_equal(truth, data['y']):
                raise ValueError('Member test rows or labels differ.')
            score_arrays.append(data['attack_scores'].copy())
    scores = np.mean(score_arrays, axis=0)
    predictions = (scores >= threshold).astype(np.int8)
    test_metrics = metrics(truth, predictions)
    encoder = members[0]['encoder']
    union_columns = sorted(set(int(c) for member in members for c in member['columns']))
    selected = [name for name in FEATURE_NAMES if any(
        name in member['selected_features'] for member in members)]
    identifier = uuid.uuid4().hex
    model = SavedSubsetEnsemble(members, union_columns)
    saved = {'model': model, 'encoder': encoder, 'columns': union_columns,
             'selected_features': selected, 'feature_names': FEATURE_NAMES,
             'algorithm': method + '_ensemble', 'model_id': identifier,
             'attack_threshold': threshold}
    print('Checking the saved ensemble against cached member predictions...', flush=True)
    bundle = joblib.load(root / 'results' / 'prepared_data.joblib')
    if not np.array_equal(truth, bundle['y_test']):
        raise ValueError('Prepared test labels differ from recorded predictions.')
    raw = bundle['raw_test']
    check = transform_features(raw.iloc[:128], encoder)[:, union_columns]
    if not np.allclose(model.predict_proba(check)[:, 1], scores[:128], atol=1e-12):
        raise ValueError('Saved member scores do not match prediction on prepared test rows.')
    sample = raw.iloc[:min(1000, len(raw))]
    encoded = transform_features(sample, encoder)[:, union_columns]
    model.predict_proba(encoded)
    classifier_times = []; pipeline_times = []
    for _ in range(3):
        start = time.perf_counter(); model.predict_proba(encoded)
        classifier_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        model.predict_proba(transform_features(sample, encoder)[:, union_columns])
        pipeline_times.append(time.perf_counter() - start)
    timing = {'batch_size': len(sample), 'repeats': 3,
              'classifier_ms_per_connection': float(np.median(classifier_times)*1000/len(sample)),
              'pipeline_ms_per_connection': float(np.median(pipeline_times)*1000/len(sample)),
              'note': 'Amortized batch latency. Five forests may cost more than one.'}
    # Exercise the same classifier entry point used by Flask before saving.
    response = classify([raw.iloc[0].to_dict()], saved)
    if not np.isclose(response['predictions'][0]['attack_score'], scores[0]):
        raise ValueError('API prediction does not match cached ensemble score.')
    offset = len(NUMERIC); groups = {name: [i] for i, name in enumerate(NUMERIC)}
    for name, cats in zip(CATEGORICAL, encoder.categories_):
        groups[name] = list(range(offset, offset + len(cats))); offset += len(cats)
    column_importance = dict(zip(union_columns, model.feature_importances_))
    importances = sorted([{'feature': name, 'importance': float(sum(
        column_importance.get(c, 0.) for c in groups[name]))} for name in selected],
        key=lambda row: -row['importance'])
    meta = {'model_id': identifier, 'algorithm': method + '_ensemble',
            'source_experiment_id': report['experiment_id'],
            'member_model_ids': [r['model_id'] for r in records],
            'attack_threshold': threshold, 'selected_features': selected,
            'feature_count': len(selected), 'config': report['config'],
            'validation_fitness': None, 'test_metrics': test_metrics,
            'prediction_timing': timing, 'feature_importances': importances,
            'security_breakdown': security_breakdown(bundle, predictions),
            'test_security_constraints_met': (
                test_metrics['recall'] >= report['config'].get('min_attack_recall', 0)
                if report['config'].get('min_attack_recall') is not None else True) and (
                test_metrics['false_alarm_rate'] <= report['config'].get('max_false_alarm_rate', 1)
                if report['config'].get('max_false_alarm_rate') is not None else True),
            'evaluation_note': 'Exploratory ensemble added after inspecting this test set. '
                               'Uniform weights and inherited threshold are fixed before computing '
                               'ensemble test metrics. Not a new independent held-out evaluation.'}
    joblib.dump(saved, root / 'models' / f'{identifier}.joblib', compress=3)
    atomic_json(root / 'models' / f'{identifier}.json', meta)
    np.savez_compressed(root / 'models' / f'{identifier}.predictions.npz',
                        y=truth, predictions=predictions, attack_scores=scores)
    atomic_json(root / 'results' / f'ensemble_{identifier}.json', meta)
    print('\nENSEMBLE TEST RESULTS')
    for key in ['accuracy', 'recall', 'f1', 'false_alarm_rate']:
        print(f'{key}: {test_metrics[key]:.2%}')
    print(f'Original input features needed: {len(selected)} (union across members)')
    print('Saved model ID: ' + identifier)
    print('Target above 90% reached: ' + str(test_metrics['accuracy'] > .90))
    print('Restart app.py to refresh the model list; existing APIs can use this model ID.')
    print(meta['evaluation_note'])
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--method', choices=['ga', 'pso'], default='ga')
    args = parser.parse_args()
    build(args.report, method=args.method)
