import time
import numpy as np


def snapshot(iteration,positions,scores,best,best_score,evaluator,start,extra=None):
    row={'iteration':iteration,'best_fitness':best_score['fitness'],
         'validation_accuracy':best_score['accuracy'],'validation_f1':best_score['f1'],
         'feature_count':int(best.sum()),'best_mask':best.tolist(),
         'mean_fitness':float(np.mean([s['fitness'] for s in scores])),
         'diversity':len({tuple(p.tolist()) for p in positions})/len(positions),
         'elapsed_seconds':time.perf_counter()-start,
         'evaluation_calls':evaluator.calls,'classifier_fits':evaluator.fits,
         'attack_recall':best_score.get('recall'),
         'attack_macro_recall':best_score.get('attack_macro_recall'),
         'false_alarm_rate':best_score.get('false_alarm_rate'),
         'security_feasible':best_score.get('security_feasible',True),
         'candidates':[{'mask':p.tolist(),'fitness':s['fitness'],
                        'accuracy':s['accuracy'],'feature_count':s['feature_count']}
                       for p,s in zip(positions,scores)]}
    row.update(extra or {})
    return row


def stopping(cfg,best,stale,start):
    if cfg.target_accuracy is not None and best['accuracy'] >= cfg.target_accuracy and best.get('security_feasible',True):
        return 'target_accuracy'
    if cfg.target_fitness is not None and best['fitness'] <= cfg.target_fitness and best.get('security_feasible',True):
        return 'target_fitness'
    if cfg.patience and stale >= cfg.patience:
        return 'no_improvement'
    if cfg.max_seconds and time.perf_counter()-start >= cfg.max_seconds:
        return 'time_budget'
    return None


def finish(algorithm,cfg,best,score,history,start,reason,names):
    return {'algorithm':algorithm,'seed':cfg.seed,'config':cfg.to_dict(),
            'best_mask':best.tolist(),
            'selected_features':[n for n,b in zip(names,best) if b],
            'feature_count':int(best.sum()),'feature_reduction_percent':100*(41-int(best.sum()))/41,
            'best_fitness':score['fitness'],'validation_metrics':score,
            'history':history,'completed_iterations':history[-1]['iteration'],
            'stop_reason':reason,'search_seconds':time.perf_counter()-start,
            'global_optimum_guaranteed':False}
