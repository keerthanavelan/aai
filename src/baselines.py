import time
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler


def baseline_masks(evaluator,k):
    x=evaluator.X.copy()
    numeric=len(evaluator.bundle['numeric_features'])
    scaler=MinMaxScaler(clip=True)
    x[:,:numeric]=scaler.fit_transform(x[:,:numeric])
    scores,_=chi2(x,evaluator.y)
    scores=np.nan_to_num(scores,nan=0.0,posinf=0.0,neginf=0.0)
    names=evaluator.bundle['feature_names']
    grouped=np.array([scores[evaluator.bundle['feature_groups'][name]].sum() for name in names])
    order=np.argsort(-grouped,kind='stable')
    mask=np.zeros(41,dtype=np.int8);mask[order[:k]]=1
    ranking=[{'feature':names[i],'score':float(grouped[i]),'selected':bool(mask[i])} for i in order]
    return np.ones(41,dtype=np.int8),mask,ranking


def run_baselines(evaluator,cfg):
    start=time.perf_counter()
    full,chi,ranking=baseline_masks(evaluator,cfg.chi_k)
    ranking_seconds=time.perf_counter()-start
    output=[]
    for name,mask in [('full',full),('chi_square',chi)]:
        t=time.perf_counter();score=evaluator.evaluate(mask)
        output.append({'algorithm':name,'best_mask':mask.tolist(),
                       'selected_features':[n for n,b in zip(evaluator.bundle['feature_names'],mask) if b],
                       'feature_count':int(mask.sum()),'feature_reduction_percent':100*(41-int(mask.sum()))/41,
                       'best_fitness':score['fitness'],'validation_metrics':score,
                       'search_seconds':(ranking_seconds if name=='chi_square' else 0.0),
                       'baseline_evaluation_seconds':time.perf_counter()-t,
                       'ranking':ranking if name=='chi_square' else [],'config':cfg.to_dict()})
    return output
