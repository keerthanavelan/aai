import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from .preprocess import mask_to_columns
from .config import parse_mask


class Cancelled(Exception):
    pass


def metrics(y, predictions):
    tn,fp,fn,tp=confusion_matrix(y,predictions,labels=[0,1]).ravel()
    return {'accuracy': float(accuracy_score(y,predictions)),
            'precision': float(precision_score(y,predictions,zero_division=0)),
            'recall': float(recall_score(y,predictions,zero_division=0)),
            'f1': float(f1_score(y,predictions,zero_division=0)),
            'confusion_matrix': [[int(tn),int(fp)],[int(fn),int(tp)]],
            'false_alarm_rate':float(fp/(tn+fp)) if tn+fp else 0.0,
            'missed_attack_rate':float(fn/(tp+fn)) if tp+fn else 0.0,
            'false_alarms':int(fp),'missed_attacks':int(fn)}


def sampled_indices(y, limit):
    indices = np.arange(len(y))
    if len(indices) <= limit:
        return indices
    chosen,_ = train_test_split(indices,train_size=limit,stratify=y,random_state=42)
    return chosen


class FitnessEvaluator:
    def __init__(self,bundle,cfg,cancel=None):
        self.bundle,self.cfg,self.cancel = bundle,cfg,cancel
        ti=sampled_indices(bundle['y_train'],cfg.train_rows)
        vi=sampled_indices(bundle['y_val'],cfg.validation_rows)
        self.X,self.y=bundle['X_train'][ti],bundle['y_train'][ti]
        self.V,self.v=bundle['X_val'][vi],bundle['y_val'][vi]
        self.cache={}
        self.calls=0
        self.fits=0

    def evaluate(self,mask):
        if self.cancel and self.cancel.is_set():
            raise Cancelled('Cancelled between classifier evaluations.')
        mask=parse_mask(mask)  # Baselines need not obey search feature bounds.
        self.calls+=1
        key=tuple(mask.tolist())
        if key in self.cache:
            return self.cache[key].copy()
        columns=mask_to_columns(mask,self.bundle['feature_groups'])
        start=time.perf_counter()
        model=RandomForestClassifier(n_estimators=self.cfg.trees,max_depth=self.cfg.max_depth,
                                      random_state=42,n_jobs=-1)
        model.fit(self.X[:,columns],self.y)
        attack_column=list(model.classes_).index(1)
        result=metrics(self.v,(model.predict_proba(self.V[:,columns])[:,attack_column]
                              >=self.cfg.attack_threshold).astype(np.int8))
        count=int(mask.sum())
        base=float(self.cfg.accuracy_weight*(1-result['accuracy'])+
                          (1-self.cfg.accuracy_weight)*count/41)
        violations=[]
        if self.cfg.min_attack_recall is not None and result['recall']<self.cfg.min_attack_recall:
            violations.append(self.cfg.min_attack_recall-result['recall'])
        if self.cfg.max_false_alarm_rate is not None and result['false_alarm_rate']>self.cfg.max_false_alarm_rate:
            violations.append(result['false_alarm_rate']-self.cfg.max_false_alarm_rate)
        result.update(fitness=(1+sum(violations)+base if violations else base),
                      base_fitness=base,security_feasible=not violations,feature_count=count,
                      evaluation_seconds=time.perf_counter()-start)
        self.cache[key]=result
        self.fits+=1
        return result.copy()
