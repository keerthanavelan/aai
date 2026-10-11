import time
import uuid
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from .preprocess import CATEGORICAL, NUMERIC, FEATURE_NAMES, transform_features, mask_to_columns
from .fitness import metrics,make_forest,attack_type_weights
from .storage import atomic_json


def train_final(bundle,result,cfg,root):
    start=time.perf_counter()
    encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
    encoder.fit(bundle['raw_train'][CATEGORICAL])
    groups={name:[i] for i,name in enumerate(NUMERIC)}
    offset=len(NUMERIC)
    for name,categories in zip(CATEGORICAL,encoder.categories_):
        groups[name]=list(range(offset,offset+len(categories)));offset+=len(categories)
    columns=mask_to_columns(result['best_mask'],groups)
    x=transform_features(bundle['raw_train'],encoder)[:,columns]
    test=transform_features(bundle['raw_test'],encoder)[:,columns]
    model=make_forest(cfg,cfg.final_trees)
    model.fit(x,bundle['y_full'],sample_weight=attack_type_weights(
        bundle.get('train_attack_types'),bundle['y_full'],cfg.weight_mode))
    attack_column=list(model.classes_).index(1)
    test_scores=model.predict_proba(test)[:,attack_column]
    predictions=(test_scores>=cfg.attack_threshold).astype(np.int8)
    test_metrics=metrics(bundle['y_test'],predictions)
    model_id=uuid.uuid4().hex
    saved={'model':model,'encoder':encoder,'columns':columns,
           'selected_features':result['selected_features'],'feature_names':FEATURE_NAMES,
           'algorithm':result['algorithm'],'model_id':model_id,'attack_threshold':cfg.attack_threshold}
    folder=root/'models';folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(saved,folder/f'{model_id}.joblib',compress=3)
    batch=bundle['raw_test'].iloc[:min(1000,len(bundle['raw_test']))]
    def end_to_end():
        return model.predict_proba(transform_features(batch,encoder)[:,columns])[:,attack_column]>=cfg.attack_threshold
    encoded_batch=transform_features(batch,encoder)[:,columns]
    model.predict_proba(encoded_batch);end_to_end()  # Warm up both timing paths.
    clf_times=[];pipeline_times=[]
    for _ in range(5):
        t=time.perf_counter();model.predict_proba(encoded_batch)[:,attack_column]>=cfg.attack_threshold;clf_times.append(time.perf_counter()-t)
        t=time.perf_counter();end_to_end();pipeline_times.append(time.perf_counter()-t)
    timing={'batch_size':len(batch),'repeats':5,
            'classifier_ms_per_connection':float(np.median(clf_times)*1000/len(batch)),
            'pipeline_ms_per_connection':float(np.median(pipeline_times)*1000/len(batch)),
            'note':'Amortized batch latency, not isolated single-connection latency.'}
    attack_report=security_breakdown(bundle,predictions)
    test_constraints_met=(cfg.min_attack_recall is None or test_metrics['recall']>=cfg.min_attack_recall) and \
                         (cfg.max_false_alarm_rate is None or test_metrics['false_alarm_rate']<=cfg.max_false_alarm_rate)
    importance_by_column={column:float(value) for column,value in zip(columns,model.feature_importances_)}
    importances=sorted([{'feature':name,'importance':sum(importance_by_column.get(c,0.0) for c in groups[name])}
                        for name in result['selected_features']],key=lambda row:-row['importance'])
    meta={'model_id':model_id,'algorithm':result['algorithm'],'attack_threshold':cfg.attack_threshold,
          'selected_features':result['selected_features'],'feature_count':result['feature_count'],
          'validation_fitness':result['best_fitness'],'test_metrics':test_metrics,
          'prediction_timing':timing,'security_breakdown':attack_report,
          'feature_importances':importances,'test_security_constraints_met':test_constraints_met,
          'config':cfg.to_dict()}
    atomic_json(folder/f'{model_id}.json',meta)
    result.update(model_id=model_id,test_metrics=test_metrics,prediction_timing=timing,
                  security_breakdown=attack_report,
                  feature_importances=importances,
                  test_security_constraints_met=test_constraints_met,
                  final_training_and_evaluation_seconds=time.perf_counter()-start)
    np.savez_compressed(folder/f'{model_id}.predictions.npz',
                        y=bundle['y_test'],predictions=predictions,attack_scores=test_scores)
    return result


def validate_connections(records,saved):
    if not isinstance(records,list) or not 1<=len(records)<=5000:
        raise ValueError('connections must contain 1 to 5000 objects.')
    rows=[];warnings=set()
    category_defaults={name:str(values[0]) for name,values in zip(CATEGORICAL,saved['encoder'].categories_)}
    known={name:set(map(str,values)) for name,values in zip(CATEGORICAL,saved['encoder'].categories_)}
    for index,record in enumerate(records):
        if not isinstance(record,dict):raise ValueError('Each connection must be an object.')
        unknown=set(record)-set(FEATURE_NAMES)
        if unknown:raise ValueError('Unknown input fields: '+', '.join(sorted(unknown)))
        missing=set(saved['selected_features'])-set(record)
        if missing:raise ValueError(f'Connection {index}: missing selected inputs: '+', '.join(sorted(missing)))
        row={}
        for name in NUMERIC:
            value=record.get(name,0)
            if isinstance(value,bool):raise ValueError(f'{name} must be numeric.')
            try:value=float(value)
            except (TypeError,ValueError):raise ValueError(f'{name} must be numeric.')
            if not np.isfinite(value) or value<0:raise ValueError(f'{name} must be finite and nonnegative.')
            if ('rate' in name or name in ['land','logged_in','root_shell','is_host_login','is_guest_login']) and value>1:
                raise ValueError(f'{name} must be between 0 and 1.')
            row[name]=value
        for name in CATEGORICAL:
            value=record.get(name,category_defaults[name])
            if not isinstance(value,str) or not value.strip():raise ValueError(f'{name} must be a nonempty string.')
            row[name]=value.strip()
            if name in saved['selected_features'] and row[name] not in known[name]:
                warnings.add(f'Unknown {name} category encoded as all zero indicators.')
        rows.append(row)
    return pd.DataFrame(rows,columns=FEATURE_NAMES),sorted(warnings)


def classify(records,saved,threshold=None):
    frame,warnings=validate_connections(records,saved)
    start=time.perf_counter()
    x=transform_features(frame,saved['encoder'])[:,saved['columns']]
    probabilities=saved['model'].predict_proba(x)
    attack_column=list(saved['model'].classes_).index(1)
    threshold=saved.get('attack_threshold',0.5) if threshold is None else threshold
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not 0.05<=threshold<=0.95:
        raise ValueError('threshold must be between 0.05 and 0.95.')
    predictions=(probabilities[:,attack_column]>=threshold).astype(np.int8)
    seconds=time.perf_counter()-start
    return {'model_id':saved['model_id'],'algorithm':saved['algorithm'],
            'selected_features':saved['selected_features'],'warnings':warnings,
            'prediction_seconds':seconds,'threshold':threshold,
            'predictions':[{'index':i,'label':'attack' if int(p) else 'normal',
                            'attack_score':float(probabilities[i,attack_column])}
                           for i,p in enumerate(predictions)],
            'score_note':'Random Forest probability estimate; not calibrated certainty.'}


def security_breakdown(bundle,predictions):
    types=bundle.get('test_attack_types')
    training=set(bundle.get('train_attack_types',[]))
    if types is None:return {'available':False,'reason':'Re-run preprocessing to include attack-type labels.'}
    rows=[]
    for name in sorted(set(types)):
        if name=='normal':continue
        selected=types==name
        rows.append({'attack_type':name,'connections':int(selected.sum()),
                     'detected':int((predictions[selected]==1).sum()),
                     'missed':int((predictions[selected]==0).sum()),
                     'recall':float(np.mean(predictions[selected]==1)),
                     'present_in_training':name in training})
    output={'available':True,'attack_types':rows}
    for label,selected in [('seen_attack_types',np.array([t!='normal' and t in training for t in types])),
                           ('unseen_attack_types',np.array([t!='normal' and t not in training for t in types]))]:
        output[label]={'connections':int(selected.sum()),
                       'recall':float(np.mean(predictions[selected]==1)) if selected.any() else None}
    return output


def threshold_sweep(saved,root):
    data=np.load(root/'models'/f"{saved['model_id']}.predictions.npz")
    return {'model_id':saved['model_id'],'evaluation_set':'KDDTest+',
            'rows':[{'threshold':float(t),**metrics(data['y'],(data['attack_scores']>=t).astype(np.int8))}
                    for t in np.arange(.05,1,.05)],
            'note':'Exploratory test-set tradeoff. Choose deployment thresholds on validation data, not this curve.'}
