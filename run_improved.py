"""A single command for the revised, attack-aware experiment."""
import argparse
import json
import joblib
from src.config import ROOT,SearchConfig
from src.preprocess import prepare_data
from src.runner import experiment
from src.calibration import calibrate


def improved_config(population=16,iterations=15):
    return SearchConfig.parse({
        'population':population,'iterations':iterations,'train_rows':20000,
        'validation_rows':25195,'trees':40,'final_trees':150,
        'max_depth':None,'rf_max_features':0.5,'weight_mode':'attack_type_sqrt',
        'objective':'attack_aware','accuracy_weight':0.98,'attack_threshold':0.5,
        'min_features':8,'max_features':35,'exclude_constant':True,'seed_ranked':True,
        'min_attack_recall':0.98,'max_false_alarm_rate':0.02,
    })


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
    parser.add_argument('--population',type=int,default=16)
    parser.add_argument('--iterations',type=int,default=15)
    args=parser.parse_args()
    path=ROOT/'results'/'prepared_data.joblib'
    if not path.is_file():prepare_data()
    bundle=joblib.load(path)
    if 'train_attack_types' not in bundle:
        prepare_data();bundle=joblib.load(path)
    cfg=improved_config(args.population,args.iterations)
    cfg=calibrate(bundle,cfg,ROOT)
    print('ATTACK-AWARE REVISION: original completed results remain unchanged.',flush=True)
    report=experiment(bundle,cfg,args.seeds,ROOT,lambda p:print(
        p.get('stage'),p.get('algorithm',''),p.get('seed',''),
        p.get('latest',{}).get('iteration',''),flush=True))
    print('\nFINAL TEST RESULTS (GA/PSO: mean across seeds)')
    for row in report['summaries']:
        m=row['metrics']
        print(f"{row['algorithm']}: accuracy={m['accuracy']['mean']:.2%}, "
              f"attack recall={m['recall']['mean']:.2%}, F1={m['f1']['mean']:.2%}, "
              f"features={m['features']['mean']:.1f}")
    print('\nReport: results/experiment_'+report['experiment_id']+'.json')
