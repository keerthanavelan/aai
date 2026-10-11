"""Calibrate each frozen GA/PSO mask on training validation, then refit one detector."""
import argparse
import json
import time
import uuid
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import ROOT, SearchConfig
from src.preprocess import mask_to_columns
from src.fitness import make_forest, attack_type_weights, metrics
from src.evaluate import train_final
from src.storage import atomic_json


def calibrate(report_path, root=ROOT, trees=150):
    started=time.perf_counter()
    report=json.loads(Path(report_path).read_text(encoding='utf-8-sig'))
    output=root/'results'/f"reduced_calibration_{report['experiment_id']}.json"
    if output.is_file():
        saved=json.loads(output.read_text());print('Already completed:',output)
        print(json.dumps(saved['test_metrics'],indent=2));print('Model ID:',saved['model_id']);return saved
    records=[r for r in report['records'] if r['algorithm'] in ['ga','pso']]
    if not records:raise ValueError('No completed GA/PSO masks found.')
    bundle=joblib.load(root/'results'/'prepared_data.joblib')
    if 'train_attack_types' not in bundle:raise ValueError('Prepared training attack types required.')
    cfg=SearchConfig.parse({**report['config'],'trees':trees,'final_trees':trees})
    source=np.arange(len(bundle['y_full']))
    ti,vi=train_test_split(source,test_size=.2,random_state=42,stratify=bundle['y_full'])
    types=np.asarray(bundle['train_attack_types']);valtypes=types[vi]
    weights=attack_type_weights(types[ti],bundle['y_train'],cfg.weight_mode)
    choices=[]
    print('Recalibrating frozen masks; GA/PSO searches are not repeated.',flush=True)
    print('Train/validation only: maximize attack-type macro recall, then overall recall, then specificity.',flush=True)
    print('Fixed validation gates: false alarms <=2%, attack recall >=98%.',flush=True)
    for index,record in enumerate(records):
        print(f'Validation RF {index+1}/{len(records)}: {record["algorithm"]}, seed={record["seed"]}',flush=True)
        columns=mask_to_columns(record['best_mask'],bundle['feature_groups'])
        model=make_forest(cfg,trees)
        model.fit(bundle['X_train'][:,columns],bundle['y_train'],sample_weight=weights)
        scores=model.predict_proba(bundle['X_val'][:,columns])[:,1]
        for threshold in [.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5]:
            predictions=(scores>=threshold).astype(np.int8)
            m=metrics(bundle['y_val'],predictions)
            macro=float(np.mean([np.mean(predictions[valtypes==name]==1)
                                for name in sorted(set(valtypes)-{'normal'})]))
            choices.append({'record_index':index,'algorithm':record['algorithm'],'seed':record['seed'],
                            'threshold':threshold,'attack_macro_recall':macro,
                            'feature_count':record['feature_count'],'validation_metrics':m})
    feasible=[c for c in choices if c['validation_metrics']['false_alarm_rate']<=.02
              and c['validation_metrics']['recall']>=.98]
    if not feasible:raise ValueError('No candidate meets fixed validation security gates.')
    chosen=max(feasible,key=lambda c:(c['attack_macro_recall'],c['validation_metrics']['recall'],
                                     -c['validation_metrics']['false_alarm_rate'],-c['feature_count']))
    identifier=uuid.uuid4().hex
    note=('Recall-focused validation calibration of previously frozen masks. No test labels used '
          'to select mask or threshold in this step. Exploratory revision after prior KDDTest+ '
          'inspection; masks were previously optimized using this validation split, so this '
          'is not nested validation or a fresh independent test. No global-optimum claim.')
    atomic_json(root/'results'/f'reduced_calibration_selection_{identifier}.json',
                {'source_experiment_id':report['experiment_id'],'chosen':chosen,'candidates':choices,
                 'rule':'Lexicographic macro recall, micro recall, specificity, feature count; '
                        'validation FAR<=2% and recall>=98%.','note':note})
    original=records[chosen['record_index']]
    current=SearchConfig.parse({**cfg.to_dict(),'seed':original['seed'],
                                'attack_threshold':chosen['threshold']})
    result={'algorithm':original['algorithm']+'_calibrated','seed':original['seed'],
            'best_mask':original['best_mask'],'selected_features':original['selected_features'],
            'feature_count':original['feature_count'],'best_fitness':original['best_fitness'],
            'original_search_fitness':original['best_fitness'],'config':current.to_dict(),
            'validation_calibration':chosen,'evaluation_note':note,
            'source_experiment_id':report['experiment_id'],'source_model_id':original['model_id']}
    print(f'Frozen choice: {original["algorithm"]} seed={original["seed"]}, '
          f'threshold={chosen["threshold"]}, features={original["feature_count"]}',flush=True)
    print('One final RF fit on all KDDTrain+, then evaluate KDDTest+.',flush=True)
    train_final(bundle,result,current,root)
    atomic_json(output,result)
    meta_path=root/'models'/f'{result["model_id"]}.json'
    meta=json.loads(meta_path.read_text())
    meta.update(source_experiment_id=report['experiment_id'],validation_calibration=chosen,
                evaluation_note=note,source_model_id=original['model_id'],
                original_search_fitness=original['best_fitness'])
    atomic_json(meta_path,meta)
    print('\nCALIBRATED RF TEST RESULTS')
    for key in ['accuracy','recall','f1','false_alarm_rate']:
        print(f'{key}: {result["test_metrics"][key]:.2%}')
    print('Missed attacks:',result['test_metrics']['missed_attacks'])
    print('Saved model ID:',result['model_id'])
    print('Threshold:',chosen['threshold'])
    print('Above 90% test accuracy:',result['test_metrics']['accuracy']>.9)
    print(f'Elapsed: {time.perf_counter()-started:.1f} seconds')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--trees',type=int,default=150)
    args=parser.parse_args()
    if not 5<=args.trees<=200:parser.error('trees must be 5–200.')
    calibrate(args.report,trees=args.trees)
