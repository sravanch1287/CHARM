"""Command-line interface for plotting published results and running new experiments."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def fresh_directory(path):
    path=Path(path).resolve()
    if path.exists() and any(path.iterdir()):raise FileExistsError(f'Output is not empty: {path}. Choose another output directory.')
    path.mkdir(parents=True,exist_ok=True)
    return path

def check_data(path):
    data=Path(path).resolve();file=data/'pathTable.csv'
    if not file.is_file():raise FileNotFoundError(f'Missing {file}; see data/README.md')
    expected=json.loads((ROOT/'configs/input_data.json').read_text())['sha256']
    if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Input hash does not match the audited Boston instance')
    return data

def run_module(module,args,env,log):
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    if log.exists():raise FileExistsError(f'Log already exists: {log}; use a new workspace')
    print(f'Running {module}; log: {log}',flush=True)
    with log.open('x',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-m','charm.'+module,*map(str,args)],env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{module} failed; inspect {log}')

def main():
    parser=argparse.ArgumentParser(prog='python -m charm',description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['reproduce','figures','tables','verify']:
        p=sub.add_parser(name,help='Regenerate/check results using bundled numerical data or new runs')
        p.add_argument('--output',default='outputs/paper' if name=='reproduce' else f'outputs/{name}')
        p.add_argument('--from-runs',type=Path,help='Use a full new experiment workspace instead of bundled paper data')
    for name in ['train','evaluate','posttraining','tier2','phy']:
        p=sub.add_parser(name)
        p.add_argument('--data-dir',required=True,type=Path)
        p.add_argument('--workspace',type=Path,default=Path('runs/reproduction'))
        p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        if name in ['train','evaluate']:
            p.add_argument('--method',choices=['EPA','DPA','DPA2','SLAPP'],default='SLAPP')
            p.add_argument('--seed',type=int,choices=range(1,6),default=1)
            p.add_argument('--all',action='store_true',help='Run all three methods and five seeds')
            p.add_argument('--episodes',type=int,default=800 if name=='train' else 50)
            p.add_argument('--steps',type=int,default=80)
        if name=='tier2':p.add_argument('--dry-count',action='store_true')
    a=parser.parse_args()
    if a.command in ['reproduce','figures','tables','verify']:
        from charm.publication import figures,tables,validate_paper_data
        output=fresh_directory(a.output);report={}
        if a.from_runs is None:validate_paper_data()
        if a.command in ['reproduce','figures']:report['figures']=figures(output,a.from_runs)
        if a.command in ['reproduce','tables','verify']:report['tables']=tables(output,a.from_runs)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));print('Outputs:',output)
        if report.get('tables',{}).get('failed'):raise SystemExit(1)
        return
    data=check_data(a.data_dir);workspace=a.workspace.resolve()
    # Never target the external input directory or any ancestor containing it.
    if data==workspace or data.is_relative_to(workspace) or workspace.is_relative_to(data):
        raise ValueError('Data and output directories must be separate')
    if any(getattr(a,k,1)<=0 for k in ['episodes','steps']):raise ValueError('Episodes and steps must be positive')
    env=os.environ.copy();env.update({'CHARM_WORKSPACE':str(workspace),'CHARM_DATA_DIR':str(data),'CHARM_DEVICE':a.device,
        'CHARM_PHY_LEGACY':'1','CHARM_STANDARD_PPO':'1','CHARM_RAW_NRF_LAGRANGIAN':'1','CHARM_DEPLOY_LOGIT_CLAMP':'8',
        'CHARM_ENTROPY_WEIGHT':'0.00001','CHARM_TORCH_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MPLBACKEND':'Agg','PYTHONUNBUFFERED':'1',
        'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
    env['PYTHONPATH']=str(ROOT)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if a.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    if a.command=='train':
        config=json.loads((ROOT/'configs/training.json').read_text())
        methods=['EPA','DPA2','SLAPP'] if a.all else [{'DPA':'DPA2'}.get(a.method,a.method)]
        for method in methods:
            for seed in (range(1,6) if a.all else [a.seed]):
                name=f'{method.lower()}_seed{seed}';run=workspace/'runs'/name
                if run.exists():raise FileExistsError(f'Training run already exists: {run}')
                run.mkdir(parents=True)
                this=dict(config,method=method,seed=seed,episodes=a.episodes,episode_length=a.steps)
                (run/'experiment_config.json').write_text(json.dumps(this,indent=2)+'\n')
                job=env.copy();job.update({'CHARM_RUN_DIR':str(run),'CHARM_SEED':str(seed),'CHARM_PA':str({'EPA':4,'DPA2':3,'SLAPP':1}[method]),
                    'CHARM_MAX_EP':str(a.episodes),'CHARM_EP_LEN':str(a.steps),'CHARM_C':'5','CHARM_D':'8','CHARM_NRF':'3',
                    'CHARM_BASE_SAFETY':'50','CHARM_BASE_NRF':'40','CHARM_BASE_VALIDITY':'20','CHARM_BASE_MOVEMENT':'10',
                    'CHARM_POST200_LR_FACTOR':'0.1','CHARM_FULL_CHECKPOINT_FREQ':'25','CHARM_SAVE_FREQ':'2000',
                    'CHARM_RESUME_CHECKPOINT':'','CHARM_PENALTY_SCHEDULE':'conditional','CHARM_QUIET':'1'})
                run_module('train',[],job,workspace/'logs'/f'train_{name}.log')
    elif a.command=='evaluate':
        args=['--episodes',a.episodes,'--steps',a.steps]
        if not a.all:args+=['--method',{'DPA':'DPA2'}.get(a.method,a.method),'--train-seed',a.seed]
        run_module('evaluate',args,env,workspace/'logs/evaluate.log')
    elif a.command=='posttraining':
        env['CHARM_TABLE5_OUT']=str(workspace/'phy/table5_posttraining_u4_8')
        run_module('evaluate_posttraining',[],env,workspace/'logs/posttraining.log')
    elif a.command=='tier2':
        args=['--dry-count'] if a.dry_count else ['--output-dir',workspace/'tier2/results','--parity-samples','100']
        run_module('tier2',args,env,workspace/'logs/tier2.log')
    else:
        for module in ['validate_sinr','validate_iterative','evaluate_sinr','evaluate_iterative']:
            run_module(module,[],env,workspace/'logs'/f'{module}.log')

if __name__=='__main__':
    try:main()
    except (FileNotFoundError,FileExistsError,ValueError,RuntimeError) as exc:
        print('Error:',exc,file=sys.stderr);raise SystemExit(2)
