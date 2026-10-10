"""Regenerate accepted-paper numeric figures and tables from small, auditable inputs."""
from pathlib import Path
import csv,gzip,hashlib,json,pickle,statistics
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METHODS=('EPA','DPA2','SLAPP')
LABELS={'EPA':'SafeDeploy + EPA','DPA2':'SafeDeploy + DPA','SLAPP':'SafeDeploy + SLAPP (CHARM)'}
COLORS={'EPA':'#0000FF','DPA2':'#008000','SLAPP':'#FF0000'}
SCENARIOS=('Low_Demand','Balanced','High_Demand','Mixed_stress','Extreme_high')

def validate_paper_data():
    manifest=json.loads((ROOT/'paper_data/provenance.json').read_text())
    for name,expected in manifest['files'].items():
        actual=hashlib.sha256((ROOT/'paper_data'/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Paper data hash mismatch: '+name)
    return manifest

def training_frame(workspace=None):
    if workspace is None:
        validate_paper_data()
        frame=pd.read_csv(ROOT/'paper_data/training_metrics.csv.gz')
    else:
        rows=[]
        for method in METHODS:
            for seed in range(1,6):
                path=Path(workspace)/'runs'/f'{method.lower()}_seed{seed}'/'logs/experiences_train_log.pkl'
                # Only load local experiment files that the user trusts.
                with path.open('rb') as f:raw=pickle.load(f)
                for ep,steps in enumerate(raw['episode_projection_stats']):
                    flags=np.asarray([[float(st[k])==0 for k in ['safety_deployments_added','validity_reassignments','nrf_violations_fixed']] for st in steps])
                    rows.append(dict(method=method,seed=seed,episode=ep,convergence=raw['convergence_history'][ep],sum_rate=raw['episode_sum_rates'][ep],qos=raw['episode_qos_satisfactions'][ep],safety=flags[:,0].mean(),validity=flags[:,1].mean(),nrf=flags[:,2].mean(),joint=flags.all(axis=1).mean()))
        frame=pd.DataFrame(rows)
    for method in METHODS:
        for seed in range(1,6):
            subset=frame[(frame.method==method)&(frame.seed==seed)].sort_values('episode')
            if not np.array_equal(subset.episode.to_numpy(),np.arange(801)):
                raise ValueError(f'Expected complete episode indices 0..800 for {method} seed {seed}')
    if len(frame)!=12015 or frame.isna().any().any():raise ValueError('Invalid training table')
    return frame

def figure_series(frame,metric,method):
    traces=[]
    for seed in range(1,6):
        subset=frame[(frame.method==method)&(frame.seed==seed)].sort_values('episode')
        traces.append(subset[metric].rolling(20,center=True,min_periods=1).mean().to_numpy())
    traces=np.asarray(traces)
    return traces.mean(axis=0),traces.std(axis=0,ddof=1)

def figures(output,workspace=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame=training_frame(workspace)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'pdf.fonttype':42})
    plans=[('figure5_training',[('convergence','Convergence score'),('sum_rate','Sum rate (Gbps)'),('qos','QoS satisfaction')]),
           ('figure6_compliance',[('safety','Stage 1 safety compliance'),('validity','Stage 2 validity compliance'),('nrf','Stage 3 NRF compliance'),('joint','Joint raw-action compliance')])]
    source_rows=[]
    for stem,panels in plans:
        fig,axes=plt.subplots(1,len(panels),figsize=(12.6,3.5),squeeze=False)
        for index,(metric,label) in enumerate(panels):
            ax=axes[0,index]
            for method in METHODS:
                mean,sd=figure_series(frame,metric,method);x=np.arange(len(mean))
                lo=np.maximum(mean-sd,0);hi=mean+sd
                if metric!='sum_rate':hi=np.minimum(hi,1)
                ax.plot(x,mean,color=COLORS[method],label=LABELS[method],lw=1.8)
                ax.fill_between(x,lo,hi,color=COLORS[method],alpha=.15,lw=0)
                source_rows.extend({'figure':stem,'metric':metric,'method':method,'episode':int(ep),'mean':float(m),'sample_sd':float(s)} for ep,m,s in zip(x,mean,sd))
            ax.set(xlabel='Episode',ylabel=label,xlim=(0,800),xticks=[0,200,400,600,800])
            if metric!='sum_rate':ax.set_ylim((.8,1.02) if metric=='validity' else (0,1.02))
            ax.grid(alpha=.25);ax.set_title(f'({chr(97+index)})',loc='left')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.88))
        for ext in ['png','pdf']:fig.savefig(output/f'{stem}.{ext}',dpi=250,bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(source_rows).to_csv(output/'figure_source_data.csv',index=False)
    return {'figures':2,'training_rows':len(frame),'series_rows':len(source_rows),'data':'bundled published-run metrics' if workspace is None else 'new training runs'}

def table_inputs(workspace=None):
    if workspace is None:
        validate_paper_data()
        return json.loads((ROOT/'paper_data/posttraining_metrics.json').read_text()),json.loads((ROOT/'paper_data/evaluation_means.json').read_text())
    base=Path(workspace)
    t5=json.loads((base/'phy/table5_posttraining_u4_8/TABLE5_data.json').read_text())
    rows=[]
    for p in sorted((base/'evaluation/five_demand_five_seed/models').glob('*/episodes.csv')):
        data=list(csv.DictReader(p.open()))
        for scenario in SCENARIOS:
            block=[r for r in data if r['scenario']==scenario]
            if len(block)!=50:raise ValueError('Paper tables require 50 episodes per scenario: '+str(p))
            rows.append({'method':block[0]['method'],'seed':int(block[0]['train_seed']),'scenario':scenario,**{k:statistics.mean(float(r[k]) for r in block) for k in ['sum_rate_gbps','qos_satisfaction','joint_raw_success','nrf_fixes','dus_moved']}})
    return t5,rows

def tables(output,workspace=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    expected=json.loads((ROOT/'configs/accepted_checks.json').read_text());t5,evals=table_inputs(workspace)
    rows=[]
    def append(table,method,scenario,metric,values,pair):
        scale=100 if metric in ['qos_satisfaction','joint_raw_success'] else 1
        mean=statistics.mean(values)*scale;sd=statistics.stdev(values)*scale
        check=all(abs(value-float(text))<=.5*10**(-len(text.partition('.')[2]))+1e-9 for value,text in zip([mean,sd],pair))
        rows.append(dict(table=table,method=method,scenario=scenario,metric=metric,mean=mean,sample_sd=sd,expected_mean=pair[0],expected_sd=pair[1],pass_printed_precision=check))
    for metric,methods in expected['table5'].items():
        for method,pair in methods.items():append('table5',method,'U[4,8]',metric,t5['metrics'][method][{'nrf_fixes':'nrf_activity'}.get(metric,metric)]['per_training_seed'],pair)
    for table in ['table6','table7']:
        for item in expected[table]:
            group=[x for x in evals if x['method']==item['method'] and (table=='table7' or x['scenario']==item['scenario'])]
            if {x['seed'] for x in group}!={1,2,3,4,5}:raise ValueError('Incomplete evaluation seed coverage')
            for metric,pair in item['metrics'].items():
                values=[statistics.mean(x[metric] for x in group if x['seed']==seed) for seed in range(1,6)]
                append(table,item['method'],item.get('scenario','all'),metric,values,pair)
    frame=pd.DataFrame(rows);frame.to_csv(output/'numerical_verification.csv',index=False)
    for table,block in frame.groupby('table',sort=False):
        block.to_csv(output/f'{table}.csv',index=False)
        def cell(row):
            dp=len(row.expected_mean.partition('.')[2]);ds=len(row.expected_sd.partition('.')[2])
            return f'{row.mean:.{dp}f} $\\pm$ {row.sample_sd:.{ds}f}'
        display={'EPA':'EPA','DPA2':'DPA','SLAPP':'SLAPP','SINR_FP':'SINR-PA','SLAPP_ITER':'SLAPP-Iter'}
        labels={'sum_rate_gbps':'Sum rate (Gbps)','qos_satisfaction':r'QoS satisfaction (\%)','convergence_score':'Convergence score','joint_raw_success':r'Joint compliance (\%)','nrf_fixes':'NRF fixes per decision','dus_moved':'DUs moved per decision'}
        if table=='table5':
            methods=['EPA','DPA2','SLAPP','SINR_FP','SLAPP_ITER']
            lines=[r'\begin{tabular}{lccccc}',r'\hline','Metric & '+' & '.join(display[m] for m in methods)+r' \\',r'\hline']
            for metric in labels:
                cells=[cell(next(r for r in block.itertuples() if r.method==m and r.metric==metric)) for m in methods]
                lines.append(labels[metric]+' & '+' & '.join(cells)+r' \\')
        else:
            metrics=['sum_rate_gbps','qos_satisfaction','joint_raw_success','nrf_fixes','dus_moved']
            lines=[r'\begin{tabular}{llccccc}' if table=='table6' else r'\begin{tabular}{lccccc}',r'\hline',('Scenario & ' if table=='table6' else '')+'Method & '+ ' & '.join(labels[m] for m in metrics)+r' \\',r'\hline']
            for scenario in (SCENARIOS if table=='table6' else ['all']):
                for method in METHODS:
                    cells=[cell(next(r for r in block.itertuples() if r.method==method and r.scenario==scenario and r.metric==metric)) for metric in metrics]
                    prefix=(scenario.replace('_',' ')+' & ') if table=='table6' else ''
                    lines.append(prefix+display[method]+' & '+' & '.join(cells)+r' \\')
        lines.extend([r'\hline',r'\end{tabular}']);(output/f'{table}.tex').write_text('\n'.join(lines)+'\n')
    result={'printed_scalar_checks':len(rows)*2,'passed':int(frame.pass_printed_precision.sum())*2,'failed':int((~frame.pass_printed_precision).sum())*2}
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
