"""Bounded RF tuning of a frozen PSO mask; never changes the active detector."""
import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

NOTE = ('Exploratory RF tuning after prior KDDTest+ inspection. The frozen mask and '
        'validation split were used earlier; this is not nested validation or a fresh '
        'independent test. Test metrics never select the candidate or threshold. '
        'The active detector and original experiment remain unchanged.')
THRESHOLDS = [.005, .01, .025, .05, .075, .1, .15, .2, .3, .4, .5]


def worker(request_path):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from src.fitness import attack_type_weights, metrics
    from src.preprocess import transform_features
    from src.evaluate import security_breakdown
    from src.storage import atomic_json
    request = json.loads(Path(request_path).read_text())
    root = Path(request['root'])
    bundle = joblib.load(root/'results'/'prepared_data.joblib')
    saved = joblib.load(root/'models'/f"{request['source_model_id']}.joblib")
    parameters = request['parameters']
    model = RandomForestClassifier(**parameters, random_state=42, n_jobs=-1)
    if request['phase'] == 'validation':
        indices = np.arange(len(bundle['y_full']))
        ti, vi = train_test_split(indices, test_size=.2, random_state=42,
                                 stratify=bundle['y_full'])
        # Match the existing scorer's encoding and split exactly.
        from src.preprocess import mask_to_columns
        columns = mask_to_columns(request['mask'], bundle['feature_groups'])
        types = np.asarray(bundle['train_attack_types'])
        weights = attack_type_weights(types[ti], bundle['y_train'], request['weight_mode'])
        model.fit(bundle['X_train'][:, columns], bundle['y_train'], sample_weight=weights)
        scores = model.predict_proba(bundle['X_val'][:, columns])[:, list(model.classes_).index(1)]
        rows = []
        for threshold in THRESHOLDS:
            predictions = (scores >= threshold).astype(np.int8)
            measured = metrics(bundle['y_val'], predictions)
            macro = float(np.mean([np.mean(predictions[types[vi] == name] == 1)
                          for name in sorted(set(types[vi]) - {'normal'})]))
            rows.append({'threshold': threshold, 'attack_macro_recall': macro,
                         'validation_metrics': measured})
        atomic_json(Path(request['output']), {'parameters': parameters, 'rows': rows})
        return
    # This phase starts only after the validation choice has been written to disk.
    x = transform_features(bundle['raw_train'], saved['encoder'])[:, saved['columns']]
    weights = attack_type_weights(np.asarray(bundle['train_attack_types']),
                                 bundle['y_full'], request['weight_mode'])
    model.fit(x, bundle['y_full'], sample_weight=weights)
    test = transform_features(bundle['raw_test'], saved['encoder'])[:, saved['columns']]
    scores = model.predict_proba(test)[:, list(model.classes_).index(1)]
    threshold = request['threshold']
    predictions = (scores >= threshold).astype(np.int8)
    measured = metrics(bundle['y_test'], predictions)
    identifier = request['model_id']
    saved.update(model=model, model_id=identifier, algorithm='pso_rf_tuned',
                 attack_threshold=threshold)
    folder = root/'models'
    # Save metadata last so partially written models are not advertised by the API.
    joblib.dump(saved, folder/f'{identifier}.joblib', compress=3)
    np.savez_compressed(folder/f'{identifier}.predictions.npz', y=bundle['y_test'],
                        predictions=predictions, attack_scores=scores)
    meta = json.loads((folder/f"{request['source_model_id']}.json").read_text())
    times = []
    batch = test[:1000]
    model.predict_proba(batch)
    for _ in range(3):
        started = time.perf_counter(); model.predict_proba(batch)
        times.append(time.perf_counter()-started)
    pipeline_times = []
    frame = bundle['raw_test'].iloc[:1000]
    for _ in range(3):
        started = time.perf_counter()
        model.predict_proba(transform_features(frame, saved['encoder'])[:, saved['columns']])
        pipeline_times.append(time.perf_counter()-started)
    meta.update(model_id=identifier, algorithm='pso_rf_tuned', attack_threshold=threshold,
                source_model_id=request['source_model_id'], rf_parameters=parameters,
                validation_calibration=request['selected'], evaluation_note=NOTE,
                test_metrics=measured, security_breakdown=security_breakdown(bundle, predictions),
                test_security_constraints_met=bool(measured['recall']>=.98 and measured['false_alarm_rate']<=.02),
                validation_fitness=None,
                prediction_timing={'batch_size':len(batch), 'repeats':3,
                    'classifier_ms_per_connection':float(np.median(times)*1000/len(batch)),
                    'pipeline_ms_per_connection':float(np.median(pipeline_times)*1000/len(batch)),
                    'note':'Amortized batch latency, not isolated single-connection latency.'})
    # Leaf size is recorded explicitly in rf_parameters (not part of SearchConfig).
    meta['config'] = {**meta['config'], 'final_trees':parameters['n_estimators'],
                      'max_depth':parameters['max_depth'],
                      'rf_max_features':parameters['max_features'], 'attack_threshold':threshold}
    groups = bundle['feature_groups']
    importance = dict(zip(saved['columns'], map(float, model.feature_importances_)))
    meta['feature_importances'] = sorted([
        {'feature':name, 'importance':sum(importance.get(c, 0.) for c in groups[name])}
        for name in saved['selected_features']], key=lambda row:-row['importance'])
    atomic_json(folder/f'{identifier}.json', meta)
    atomic_json(Path(request['output']), meta)


def bounded_worker(request, folder, seconds):
    path = folder/f"request_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(request), encoding='utf-8')
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                       check=True, timeout=max(1, seconds))
        return json.loads(Path(request['output']).read_text())
    except subprocess.TimeoutExpired:
        print('Time limit reached; worker stopped. Active detector unchanged.', flush=True)
    except subprocess.CalledProcessError:
        print('Worker failed; active detector unchanged. See traceback above.', flush=True)
    return None


def tune(root, minutes):
    from src.storage import atomic_json
    started = time.monotonic()
    deadline = started + minutes*60
    tuning_deadline = started + minutes*60*.65
    policy = json.loads((root/'detector_policy.json').read_text())
    source = policy['model_id']
    source_meta = json.loads((root/'models'/f'{source}.json').read_text())
    if not (root/'models'/f'{source}.joblib').is_file():
        raise ValueError('Run activate_detector.py first.')
    cfg = source_meta['config']
    names = source_meta['selected_features']
    from src.preprocess import FEATURE_NAMES
    # Preflight feature identity; only 41 original feature groups, no encoded-bit masks.
    mask = [int(name in names) for name in FEATURE_NAMES]
    identifier = uuid.uuid4().hex
    folder = root/'results'/f'rf_tuning_{identifier}'
    folder.mkdir(parents=True, exist_ok=False)
    base = {'n_estimators':cfg['final_trees'], 'max_depth':cfg['max_depth'],
            'max_features':cfg['rf_max_features'], 'min_samples_leaf':1}
    candidates = [base,
                  {**base, 'n_estimators':300},
                  {**base, 'min_samples_leaf':2},
                  {**base, 'max_features':.75, 'min_samples_leaf':2},
                  {**base, 'max_features':'sqrt', 'min_samples_leaf':2}]
    print(f'Budget: {minutes:g} minutes including final training. Fixed inputs: {len(names)}.', flush=True)
    print('Validation rule: FAR<=2%, recall>=98%; attack-type macro recall>=98%; maximize F1, '
          'then attack recall, then minimize false alarms. No test-based selection.', flush=True)
    completed = []
    for index, parameters in enumerate(candidates):
        remaining = tuning_deadline-time.monotonic()
        if remaining <= 1: break
        print(f'Validation trial {index+1}/{len(candidates)}: {parameters}', flush=True)
        request = {'root':str(root), 'source_model_id':source, 'mask':mask,
                   'weight_mode':cfg['weight_mode'], 'parameters':parameters,
                   'phase':'validation', 'output':str(folder/f'trial_{index}.json')}
        trial = bounded_worker(request, folder, min(remaining, minutes*60*.2))
        if trial is not None: completed.append({'trial_index':index, **trial})
        atomic_json(folder/'progress.json', {'completed':completed, 'note':NOTE})
    feasible = [{'trial_index':trial['trial_index'], 'parameters':trial['parameters'], **row}
                for trial in completed for row in trial['rows']
                if row['validation_metrics']['false_alarm_rate']<=.02
                and row['validation_metrics']['recall']>=.98 and row['attack_macro_recall']>=.98]
    if not feasible:
        print('No completed candidate met validation gates. Keep your current detector.'); return
    def rank(row):
        return (row['validation_metrics']['f1'], row['validation_metrics']['recall'],
                -row['validation_metrics']['false_alarm_rate'])
    # Lower complexity wins exact metric ties because candidates are ordered baseline first.
    selected = max(feasible, key=rank)
    incumbent = {'attack_macro_recall':policy['validation_calibration']['attack_macro_recall'],
                 'validation_metrics':policy['validation_calibration']['validation_metrics']}
    atomic_json(folder/'selection.json', {'selected':selected, 'incumbent':incumbent,
                'completed_trials':len(completed), 'thresholds':THRESHOLDS, 'note':NOTE})
    print('Frozen validation selection:',json.dumps(selected),flush=True)
    if rank(selected) <= rank(incumbent):
        print('No improvement under the predeclared validation rule. No final fit needed.')
        print('KEEP your current detector. Results:',folder); return
    if deadline-time.monotonic()<=1:
        print('No final-training budget remains. Keep your current detector.'); return
    print('One final fit; test metrics are computed only now. Default detector stays unchanged.',flush=True)
    request = {'root':str(root), 'source_model_id':source, 'weight_mode':cfg['weight_mode'],
               'parameters':selected['parameters'], 'phase':'final', 'selected':selected,
               'threshold':selected['threshold'], 'model_id':identifier,
               'output':str(folder/'candidate.json')}
    candidate = bounded_worker(request, folder, deadline-time.monotonic())
    if candidate is None:return
    print('\nCANDIDATE TEST RESULTS — exploratory')
    for key in ['accuracy','recall','f1','false_alarm_rate']:
        print(f"{key}: {candidate['test_metrics'][key]:.2%} (current {source_meta['test_metrics'][key]:.2%})")
    print('Candidate model ID:',identifier)
    print('Default detector unchanged. Report:',folder/'candidate.json')
    print(NOTE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=float, default=50)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    else:
        if not 1 <= args.minutes <= 60: parser.error('--minutes must be between 1 and 60')
        tune(args.root.resolve(), args.minutes)
