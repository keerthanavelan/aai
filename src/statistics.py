import numpy as np
from scipy.stats import wilcoxon


def binary_f1(y,p):
    tp=np.count_nonzero((y==1)&(p==1));fp=np.count_nonzero((y==0)&(p==1))
    fn=np.count_nonzero((y==1)&(p==0));den=2*tp+fp+fn
    return 2*tp/den if den else 0.0


def compare_statistics(root,records,bootstrap_repeats=300):
    ga=[r for r in records if r['algorithm']=='ga']
    pso=[r for r in records if r['algorithm']=='pso']
    matched=sorted(set(r['seed'] for r in ga)&set(r['seed'] for r in pso))
    paired={}
    for metric in ['best_fitness','test_f1']:
        a=[];b=[]
        for seed in matched:
            g=next(r for r in ga if r['seed']==seed);p=next(r for r in pso if r['seed']==seed)
            a.append(g['best_fitness'] if metric=='best_fitness' else g['test_metrics']['f1'])
            b.append(p['best_fitness'] if metric=='best_fitness' else p['test_metrics']['f1'])
        differences=np.array(a)-np.array(b)
        if len(differences)<2:
            paired[metric]={'available':False,'reason':'At least two matched seeds needed.'};continue
        if np.allclose(differences,0):stat,pvalue=0.0,1.0
        else:
            outcome=wilcoxon(differences,method='auto');stat,pvalue=float(outcome.statistic),float(outcome.pvalue)
        paired[metric]={'n_pairs':len(matched),'statistic':stat,'p_value':pvalue,
                        'p_value_bonferroni_two_metrics':min(1.0,pvalue*2),
                        'mean_ga_minus_pso':float(differences.mean())}
    # Freeze one model per method using validation fitness, never test metrics.
    chosen={}
    for record in records:
        method=record['algorithm']
        if method not in chosen or record['best_fitness']<chosen[method]['best_fitness']:
            chosen[method]=record
    full=chosen.get('full');bootstrap=[]
    if full:
        baseline=np.load(root/'models'/f"{full['model_id']}.predictions.npz")
        y=baseline['y'];bpred=baseline['predictions']
        negatives=np.flatnonzero(y==0);positives=np.flatnonzero(y==1)
        for method,record in chosen.items():
            if method=='full':continue
            candidate=np.load(root/'models'/f"{record['model_id']}.predictions.npz")['predictions']
            rng=np.random.default_rng(123);acc=[];f1=[]
            for _ in range(bootstrap_repeats):
                idx=np.r_[rng.choice(negatives,len(negatives),replace=True),
                          rng.choice(positives,len(positives),replace=True)]
                acc.append(float(np.mean(candidate[idx]==y[idx])-np.mean(bpred[idx]==y[idx])))
                f1.append(float(binary_f1(y[idx],candidate[idx])-binary_f1(y[idx],bpred[idx])))
            bootstrap.append({'method':method,'reference':'full','model_id':record['model_id'],
                              'accuracy_difference':float(np.mean(candidate==y)-np.mean(bpred==y)),
                              'accuracy_difference_ci95':np.quantile(acc,[.025,.975]).tolist(),
                              'f1_difference':binary_f1(y,candidate)-binary_f1(y,bpred),
                              'f1_difference_ci95':np.quantile(f1,[.025,.975]).tolist(),
                              'bootstrap_repeats':bootstrap_repeats})
    return {'ga_vs_pso_paired':paired,'paired_stratified_test_bootstrap':bootstrap,
            'limits':'Seeds measure optimizer variability on a fixed split, not independent datasets. '
                     'Small seed counts have low power. Bootstrap intervals assume exchangeable '
                     'test connections within each class; they do not prove deployment performance.'}
