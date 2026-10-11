"""Small real-data end-to-end check. Creates real demo models/results, not report-grade experiments."""
import json
import tempfile
import time
from pathlib import Path
import joblib
import numpy as np
from src.config import ROOT,SearchConfig
from src.fitness import FitnessEvaluator
from src.ga import run_ga
from src.pso import run_pso
from src.baselines import run_baselines
from src.evaluate import train_final
from src.runner import experiment
from src.service import ProjectService


def check(root=ROOT):
    bundle=joblib.load(root/'results'/'prepared_data.joblib')
    cfg=SearchConfig.parse({'population':4,'iterations':2,'train_rows':1000,
                            'validation_rows':500,'trees':8,'final_trees':8,'max_depth':8})
    results=[]
    for name,fn in [('ga',run_ga),('pso',run_pso)]:
        evaluator=FitnessEvaluator(bundle,cfg)
        result=fn(evaluator,cfg)
        repeat=fn(FitnessEvaluator(bundle,cfg),cfg)
        assert result['best_mask']==repeat['best_mask'],'Seed reproducibility failed'
        history=[h['best_fitness'] for h in result['history']]
        assert all(a>=b-1e-12 for a,b in zip(history,history[1:])),'Best-so-far regression'
        assert 1<=result['feature_count']<=41
        train_final(bundle,result,cfg,root);results.append(result)
        print(name,'search + final model passed',flush=True)
    for result in run_baselines(FitnessEvaluator(bundle,cfg),cfg):
        train_final(bundle,result,cfg,root);results.append(result)
    assert results[-1]['feature_count']==12
    service=ProjectService(root)
    identifier=results[0]['model_id'];sample=service.sample('normal')
    prediction=service.predict({'model_id':identifier,'connection':sample['connection']})
    assert prediction['predictions'][0]['label'] in ['normal','attack']
    selected=set(results[0]['selected_features'])
    excluded=next((n for n in bundle['numeric_features'] if n not in selected),None)
    if excluded:
        original=sample['connection'][excluded]
        replacement=1-original if 'rate' in excluded else original+1
        if excluded in ['land','logged_in','root_shell','is_host_login','is_guest_login']:replacement=1-original
        response=service.what_if({'model_ids':[identifier],'connection':sample['connection'],
                                  'changes':{excluded:replacement}})
        assert response['comparisons'][0]['attack_score_change']==0.0,'Excluded input affected prediction'
    replay=service.replay(identifier,0,3);assert len(replay['predictions'])==3
    challenge=service.new_challenge(identifier)
    assert 'actual_label' not in challenge and 'machine' not in challenge
    answer=service.answer_challenge(challenge['challenge_id'],'attack');assert 'correct' in answer
    assert len(service.threshold_curve(identifier)['rows'])==19
    assert results[0]['security_breakdown']['available']
    print('Prediction, what-if, replay, challenge, thresholds, attack-type breakdown passed',flush=True)
    # An impossible pair of security constraints must be reported, not silently accepted.
    strict=SearchConfig.parse({**cfg.to_dict(),'min_attack_recall':1.0,'max_false_alarm_rate':0.0})
    score=FitnessEvaluator(bundle,strict).evaluate(np.ones(41,dtype=np.int8))
    if not score['security_feasible']:assert score['fitness']>1
    report=experiment(bundle,SearchConfig.parse({**cfg.to_dict(),'iterations':1}),[1,2],root)
    assert len(report['records'])==6
    assert len(report['statistics']['paired_stratified_test_bootstrap'])==3
    print('Repeated experiments + statistical comparisons passed',flush=True)
    return {'checks':'passed','real_test_connections':len(bundle['y_test']),
            'smoke_config':cfg.to_dict(),'models_created':len(service.models()),
            'note':'Tiny search settings verify operation; they are not final project results.'}


if __name__=='__main__':
    if not (ROOT/'results'/'prepared_data.joblib').is_file():raise SystemExit('Run preprocessing first.')
    print(json.dumps(check(),indent=2))
