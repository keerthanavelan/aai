import time
import numpy as np
from .config import initial_positions, repair
from .search_common import snapshot, stopping, finish


def run_ga(evaluator,cfg,progress=None,initial_masks=None):
    start=time.perf_counter()
    rng=np.random.default_rng(cfg.seed)
    supplied=list(initial_masks or [])
    if cfg.seed_ranked:supplied+=(evaluator.ranked_masks()[:max(0,cfg.population-len(supplied))])
    positions=initial_positions(cfg,rng,supplied,evaluator.allowed)
    scores=[evaluator.evaluate(p) for p in positions]
    index=int(np.argmin([s['fitness'] for s in scores]))
    best,best_score=positions[index].copy(),scores[index].copy()
    history=[];stale=0;reason=None
    for generation in range(cfg.iterations+1):
        row=snapshot(generation,positions,scores,best,best_score,evaluator,start)
        history.append(row)
        if progress:progress(row)
        reason=stopping(cfg,best_score,stale,start)
        if reason or generation==cfg.iterations:
            break
        children=[best.copy()]  # Elitism preserves the best known solution.
        def parent():
            choices=rng.choice(cfg.population,cfg.tournament_size,replace=False)
            winner=min(choices,key=lambda j:scores[j]['fitness'])
            return positions[winner].copy()
        while len(children)<cfg.population:
            a,b=parent(),parent()
            if rng.random()<cfg.crossover_rate:
                cut=int(rng.integers(1,41))
                a,b=np.r_[a[:cut],b[cut:]],np.r_[b[:cut],a[cut:]]
            for child in [a,b]:
                flips=rng.random(41)<cfg.mutation_rate
                child[flips]=1-child[flips]
                children.append(repair(child,rng,cfg,evaluator.allowed))
                if len(children)==cfg.population:break
        positions=np.array(children)
        scores=[evaluator.evaluate(p) for p in positions]
        index=int(np.argmin([s['fitness'] for s in scores]))
        if scores[index]['fitness'] < best_score['fitness']-1e-12:
            best,best_score=positions[index].copy(),scores[index].copy();stale=0
        else:stale+=1
    return finish('ga',cfg,best,best_score,history,start,reason or 'iteration_limit',
                  evaluator.bundle['feature_names'])
