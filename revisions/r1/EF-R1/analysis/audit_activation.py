"""Independently evaluate ALL feasible corridors for each archived port plan.

No source data or archived results are modified. The pruning and aggregate
range rule reproduce the reference methanol implementation (k=1).
"""
from pathlib import Path
import sys, json, hashlib, shutil
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '_deps'))
import pandas as pd
import numpy as np

ROOT = HERE.parents[2]
ARCHIVE = HERE / 'inputs' if (HERE / 'inputs' / 'data').exists() else ROOT / '06_Github'
DATA = ARCHIVE / 'data' / 'derived'
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)
P = DATA / 'processed_v5'
Q = DATA / 'policy_analysis_v5'
R = DATA / 'milp_results_v5'

def load():
    c = pd.read_parquet(P/'corridors_container.parquet')
    p = pd.read_parquet(P/'candidate_ports_container.parquet').drop_duplicates('port_code')
    m = pd.read_parquet(P/'M_ell_container.parquet')
    gov = pd.read_parquet(Q/'governance_corridors_container.parquet')
    c['W'] = c.E_e*c.eta_green_methanol*c.alpha_e
    xy=p.set_index('port_code')[['port_lon','port_lat']].to_dict('index')
    coords=c.set_index('corridor_id')[['origin_lon','origin_lat','dest_lon','dest_lat']].to_dict('index')
    def deviation(row):
        q=coords.get(row.corridor_id); t=xy.get(row.intermediate_port)
        if q is None or t is None:return 1e9
        return ((t['port_lon']-(q['origin_lon']+q['dest_lon'])/2)**2+(t['port_lat']-(q['origin_lat']+q['dest_lat'])/2)**2)**0.5
    m['deviation']=m.apply(deviation,axis=1)
    m=m.sort_values('deviation').groupby(['corridor_id','fuel']).head(15)
    feasible=set(p.loc[p.is_feasible,'port_code'])
    mids={k:list(g.intermediate_port) for k,g in m[(m.fuel=='green_methanol') & m.intermediate_port.isin(feasible)].groupby('corridor_id')}
    return c,p,m,gov,mids,feasible

def closure(c, selected, mids, feasible):
    selected=set(selected)&feasible
    ok=c.origin_port.isin(selected)&c.dest_port.isin(selected)
    for i,row in c[c.mean_distance>8000].iterrows():
        k=max(1,int(row.k_e_green_methanol))
        ok.loc[i] &= sum(p in selected for p in mids.get(row.corridor_id,[]))>=k
    return ok

def main():
    c,p,m,g,mids,feasible=load()
    print('CORRIDOR', c.columns.tolist()); print('GOV',g.columns.tolist())
    plans={'Global':R/'selected_ports_container.csv','EU':Q/'meU_selected_ports_container.csv','BRI':Q/'meB_selected_ports_container.csv','BRI_open':Q/'bri_view_control_selected_ports_container.csv','EU_forced':Q/'meU_forced_selected_ports_container.csv','Global_forced':Q/'fueleu_selected_ports_container.csv'}
    g=g.set_index('corridor_id')
    c['EU_scope']=c.corridor_id.map(g.eu_ets_endpoint).fillna(False).astype(bool)
    c['BRI_scope']=c.corridor_id.map(g.is_bri).fillna(False).astype(bool)
    rows=[]
    for name,f in plans.items():
        s=pd.read_csv(f).port_code
        ok=closure(c,s,mids,feasible)
        act=c[ok]
        act.to_csv(OUT/f'closure_{name}.csv',index=False)
        rows.append({'plan':name,'n_ports':len(s),'n_enabled':int(ok.sum()),'n_positive':int((act.W>0).sum()),'global_Mt':act.W.sum()/1e6,'EU_Mt':act.loc[act.EU_scope,'W'].sum()/1e6,'BRI_Mt':act.loc[act.BRI_scope,'W'].sum()/1e6,'cost_MUSD':p.loc[p.port_code.isin(s),'cost_p'].sum()})
    out=pd.DataFrame(rows)
    out['loss_pct']=(1-out.global_Mt/out.iloc[0].global_Mt)*100
    out.to_csv(OUT/'archived_plan_closure.csv',index=False)
    print(out.to_string(index=False))
    # Record the actual grid, not the stale pre-cleanup JSON statistics.
    grid=pd.read_parquet(DATA/'h3_grid_v5'/'h3_emission_grid_container.parquet')
    print('GRID',len(grid),grid.columns.tolist())
    (OUT/'grid_audit.json').write_text(json.dumps({'n_cells':len(grid),'columns':grid.columns.tolist()},indent=2))

if __name__=='__main__':main()
