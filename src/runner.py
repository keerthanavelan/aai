import argparse
import json
import uuid
import time
import numpy as np
import joblib
from .config import ROOT,SearchConfig
from .fitness import FitnessEvaluator,Cancelled
from .ga import run_ga
from .pso import run_pso
from .baselines import run_baselines
from .evaluate import train_final
from .statistics import compare_statistics
from .storage import atomic_json


def experiment(bundle,cfg,seeds,root=ROOT,progress=None,cancel=None):
    if not isinstance(seeds,list) or not 2<=len(seeds)<=10 or len(set(seeds))!=len(seeds):
        raise ValueError('seeds must contain 2 to 10 distinct integers.')
    for seed in seeds:SearchConfig.parse({**cfg.to_dict(),'seed':seed})
    identifier=uuid.uuid4().hex;records=[];start=time.perf_counter()
    root=ROOT if root is None else root
    def notify(message):
        if progress:progress(message)
        if cancel and cancel.is_set():raise Cancelled('Experiment cancelled between stages.')
    notify({'stage':'baselines','completed_runs':0,'total_runs':2*len(seeds)+2})
    for result in run_baselines(FitnessEvaluator(bundle,cfg,cancel),cfg):
        notify({'stage':'final_training','algorithm':result['algorithm'],'completed_runs':len(records)})
        train_final(bundle,result,cfg,root);records.append(result)
        atomic_json(root/'results'/f'{identifier}_{result["algorithm"]}.json',result)
    for seed in seeds:
        current=SearchConfig.parse({**cfg.to_dict(),'seed':seed})
        for name,algorithm in [('ga',run_ga),('pso',run_pso)]:
            notify({'stage':'search','algorithm':name,'seed':seed,'completed_runs':len(records)})
            evaluator=FitnessEvaluator(bundle,current,cancel)
            def callback(row):
                notify({'stage':'search','algorithm':name,'seed':seed,
                        'completed_runs':len(records),'latest':row})
            result=algorithm(evaluator,current,callback)
            notify({'stage':'final_training','algorithm':name,'seed':seed,'completed_runs':len(records)})
            train_final(bundle,result,current,root);records.append(result)
            atomic_json(root/'results'/f'{identifier}_{name}_{seed}.json',result)
    summaries=[]
    for method in ['full','chi_square','ga','pso']:
        items=[r for r in records if r['algorithm']==method]
        stats={}
        values={'accuracy':[r['test_metrics']['accuracy'] for r in items],
                'f1':[r['test_metrics']['f1'] for r in items],
                'recall':[r['test_metrics']['recall'] for r in items],
                'features':[r['feature_count'] for r in items],
                'search_seconds':[r['search_seconds'] for r in items],
                'pipeline_ms_per_connection':[r['prediction_timing']['pipeline_ms_per_connection'] for r in items]}
        for key,arr in values.items():
            stats[key]={'mean':float(np.mean(arr)),
                        'std':float(np.std(arr,ddof=1)) if len(arr)>1 else None}
        frequencies=np.mean(np.array([r['best_mask'] for r in items]),axis=0)
        summaries.append({'algorithm':method,'runs':len(items),'metrics':stats,
                          'feature_selection_frequency':[
                              {'feature':name,'frequency':float(frequency)}
                              for name,frequency in zip(bundle['feature_names'],frequencies)]})
    notify({'stage':'statistics','completed_runs':len(records)})
    report={'experiment_id':identifier,'config':cfg.to_dict(),'seeds':seeds,
            'records':records,'summaries':summaries,
            'statistics':compare_statistics(root,records),
            'elapsed_seconds':time.perf_counter()-start}
    atomic_json(root/'results'/f'experiment_{identifier}.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
    parser.add_argument('--population',type=int,default=16)
    parser.add_argument('--iterations',type=int,default=15)
    parser.add_argument('--train-rows',type=int,default=10000)
    parser.add_argument('--validation-rows',type=int,default=5000)
    args=parser.parse_args()
    cfg=SearchConfig.parse({'population':args.population,'iterations':args.iterations,
                            'train_rows':args.train_rows,'validation_rows':args.validation_rows})
    path=ROOT/'results'/'prepared_data.joblib'
    if not path.is_file():raise SystemExit('Run python src/preprocess.py first.')
    report=experiment(joblib.load(path),cfg,args.seeds,progress=lambda p:print(
        p.get('stage'),p.get('algorithm',''),p.get('seed',''),
        p.get('latest',{}).get('iteration',''),flush=True))
    print(json.dumps(report['summaries'],indent=2))
