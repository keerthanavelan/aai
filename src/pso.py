import time
import numpy as np
from scipy.special import expit
from .config import initial_positions, repair
from .search_common import snapshot, stopping, finish


def run_pso(evaluator,cfg,progress=None,initial_masks=None):
    rng=np.random.default_rng(cfg.seed)
    positions=initial_positions(cfg,rng,initial_masks)
    velocity=rng.uniform(-1,1,positions.shape)
    start=time.perf_counter()
    scores=[evaluator.evaluate(p) for p in positions]
    personal=positions.copy()
    personal_fitness=np.array([s['fitness'] for s in scores])
    index=int(np.argmin(personal_fitness))
    best,best_score=positions[index].copy(),scores[index].copy()
    history=[];stale=0;reason=None
    for iteration in range(cfg.iterations+1):
        row=snapshot(iteration,positions,scores,best,best_score,evaluator,start,
                     {'mean_absolute_velocity':float(np.abs(velocity).mean()),
                      'personal_best_fitness':personal_fitness.tolist()})
        history.append(row)
        if progress:progress(row)
        reason=stopping(cfg,best_score,stale,start)
        if reason or iteration==cfg.iterations:break
        velocity=cfg.inertia*velocity+cfg.cognitive*rng.random(positions.shape)*(personal-positions)\
                 +cfg.social*rng.random(positions.shape)*(best-positions)
        velocity=np.clip(velocity,-cfg.velocity_limit,cfg.velocity_limit)
        positions=(rng.random(positions.shape)<expit(velocity)).astype(np.int8)
        positions=np.array([repair(p,rng,cfg) for p in positions])
        scores=[evaluator.evaluate(p) for p in positions]
        for j,score in enumerate(scores):
            if score['fitness']<personal_fitness[j]:
                personal[j]=positions[j].copy();personal_fitness[j]=score['fitness']
        index=int(np.argmin([s['fitness'] for s in scores]))
        if scores[index]['fitness']<best_score['fitness']-1e-12:
            best,best_score=positions[index].copy(),scores[index].copy();stale=0
        else:stale+=1
    return finish('pso',cfg,best,best_score,history,start,reason or 'iteration_limit',
                  evaluator.bundle['feature_names'])
