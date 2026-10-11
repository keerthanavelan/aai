"""Choose RF weighting and threshold from training/validation only."""
from dataclasses import replace
import uuid
import numpy as np
from sklearn.model_selection import train_test_split
from .fitness import make_forest,attack_type_weights,metrics
from .storage import atomic_json


def calibrate(bundle,cfg,root,progress=print):
    source=np.arange(len(bundle['y_full']))
    ti,vi=train_test_split(source,test_size=.2,random_state=42,stratify=bundle['y_full'])
    types=np.asarray(bundle['train_attack_types']);training=types[ti];validation=types[vi]
    choices=[]
    for mode in ['attack_type_sqrt','attack_type_equal']:
        current=replace(cfg,weight_mode=mode)
        progress('Validation calibration: '+mode)
        model=make_forest(current,current.final_trees)
        model.fit(bundle['X_train'],bundle['y_train'],
                  sample_weight=attack_type_weights(training,bundle['y_train'],mode))
        scores=model.predict_proba(bundle['X_val'])[:,list(model.classes_).index(1)]
        for threshold in [.05,.1,.15,.2,.25,.3,.35,.4,.5]:
            predictions=(scores>=threshold).astype(np.int8)
            m=metrics(bundle['y_val'],predictions)
            macro=float(np.mean([np.mean(predictions[validation==t]==1)
                                 for t in set(validation) if t!='normal']))
            quality=.5*macro+.25*m['recall']+.25*(1-m['false_alarm_rate'])
            choices.append({'weight_mode':mode,'threshold':threshold,'quality':quality,
                            'attack_macro_recall':macro,**m})
    feasible=[c for c in choices if c['false_alarm_rate']<=.005]
    if not feasible:raise ValueError('No calibration setting meets the validation false-alarm limit of 0.5%.')
    chosen=max(feasible,key=lambda c:(c['quality'],c['threshold']))
    revised=replace(cfg,weight_mode=chosen['weight_mode'],attack_threshold=chosen['threshold'])
    atomic_json(root/'results'/f'calibration_{uuid.uuid4().hex}.json',
                {'selected':chosen,'candidates':choices,'config':revised.to_dict(),
                 'selection_set':'Training split and validation split of KDDTrain+ only; no KDDTest+ selection.',
                 'limits':'Selected on a random within-training split; unseen traffic may behave differently.'})
    progress(f"Chosen on validation: {chosen['weight_mode']}, threshold={chosen['threshold']}, "
             f"attack macro recall={chosen['attack_macro_recall']:.2%}, false alarms={chosen['false_alarm_rate']:.2%}")
    return revised
