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


def sampled_indices(y, limit,types=None):
    indices = np.arange(len(y))
    if len(indices) <= limit:
        return indices
    if types is not None:
        rng=np.random.default_rng(42)
        mandatory=[]
        for name in sorted(set(types)):
            group=np.flatnonzero(types==name)
            mandatory.extend(rng.choice(group,min(8,len(group)),replace=False).tolist())
        mandatory=np.array(mandatory,dtype=int)
        if len(mandatory)>limit:return rng.choice(mandatory,limit,replace=False)
        remainder=np.setdiff1d(indices,mandatory)
        return np.r_[mandatory,rng.choice(remainder,limit-len(mandatory),replace=False)]
    chosen,_ = train_test_split(indices,train_size=limit,stratify=y,random_state=42)
    return chosen


class FitnessEvaluator:
    def __init__(self,bundle,cfg,cancel=None):
        self.bundle,self.cfg,self.cancel = bundle,cfg,cancel
        full_types=bundle.get('train_attack_types')
        train_types=val_types=None
        if full_types is not None:
            source=np.arange(len(bundle['y_full']))
            tr,va=train_test_split(source,test_size=.2,random_state=42,stratify=bundle['y_full'])
            train_types=np.asarray(full_types)[tr];val_types=np.asarray(full_types)[va]
        if (cfg.objective=='attack_aware' or cfg.weight_mode!='none') and full_types is None:
            raise ValueError('Re-run preprocessing: attack-type labels are required for this configuration.')
        typed=cfg.objective=='attack_aware'
        ti=sampled_indices(bundle['y_train'],cfg.train_rows,train_types if typed else None)
        vi=sampled_indices(bundle['y_val'],cfg.validation_rows,val_types if typed else None)
        self.X,self.y=bundle['X_train'][ti],bundle['y_train'][ti]
        self.V,self.v=bundle['X_val'][vi],bundle['y_val'][vi]
        self.train_types=train_types[ti] if train_types is not None else None
        self.val_types=val_types[vi] if val_types is not None else None
        self.weights=attack_type_weights(self.train_types,self.y,cfg.weight_mode)
        self.allowed=np.ones(41,dtype=bool)
        if cfg.exclude_constant:
            spread=np.ptp(bundle['X_train'],axis=0)
            self.allowed=np.array([np.any(spread[bundle['feature_groups'][name]]>0)
                                   for name in bundle['feature_names']])
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
        model=make_forest(self.cfg,self.cfg.trees)
        model.fit(self.X[:,columns],self.y,sample_weight=self.weights)
        attack_column=list(model.classes_).index(1)
        predictions=(model.predict_proba(self.V[:,columns])[:,attack_column]>=self.cfg.attack_threshold).astype(np.int8)
        result=metrics(self.v,predictions)
        if self.val_types is not None:
            result['attack_macro_recall']=float(np.mean([
                np.mean(predictions[self.val_types==t]==1) for t in set(self.val_types) if t!='normal']))
        quality=result['accuracy']
        if self.cfg.objective=='attack_aware':
            quality=.5*result['attack_macro_recall']+.25*result['recall']+.25*(1-result['false_alarm_rate'])
        count=int(mask.sum())
        base=float(self.cfg.accuracy_weight*(1-quality)+
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

    def ranked_masks(self):
        model=make_forest(self.cfg,self.cfg.trees)
        model.fit(self.X,self.y,sample_weight=self.weights)
        scores=np.array([model.feature_importances_[self.bundle['feature_groups'][name]].sum()
                         for name in self.bundle['feature_names']])
        order=[i for i in np.argsort(-scores) if self.allowed[i]]
        masks=[]
        for size in [12,20,30,len(order)]:
            size=max(self.cfg.min_features,min(size,self.cfg.max_features,len(order)))
            mask=np.zeros(41,dtype=np.int8);mask[order[:size]]=1
            if not any(np.array_equal(mask,m) for m in masks):masks.append(mask)
        return masks


def make_forest(cfg,trees):
    return RandomForestClassifier(n_estimators=trees,max_depth=cfg.max_depth,
                                  max_features=cfg.rf_max_features,random_state=42,n_jobs=-1)


def attack_type_weights(types,y,mode):
    if mode=='none':return None
    weights=np.ones(len(y))
    names,counts=np.unique(types[y==1],return_counts=True)
    lookup=dict(zip(names,counts))
    exponent=1.0 if mode=='attack_type_equal' else .5
    weights[y==1]=[lookup[t]**(-exponent) for t in types[y==1]]
    weights[y==1]/=weights[y==1].mean()
    return weights
