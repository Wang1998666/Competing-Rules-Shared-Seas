"""R1 experiments with consistent energy/adoption and one charge per port.

The price experiment is a coordinated net-benefit optimization, not a market
equilibrium or a simulation of the IMO tiered fuel-intensity instrument.
"""
from recompute_rule_comparison import literal_set
from audit_activation import *
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack

def main():
    c,p,m,g,mids,feasible=load(); c=c.reset_index(drop=True)
    p=p[p.is_feasible].reset_index(drop=True)
    eu=literal_set('03c_h3_policy_coverage.py','EU_EEA_COUNTRIES')
    bri=literal_set('08_bri_analysis.py','BRI_MARITIME_COUNTRIES')
    country=p.set_index('port_code').country_code
    nP=len(p);nC=len(c);n=nP+nC;pidx={v:i for i,v in enumerate(p.port_code)}
    ri=[];ci=[];vv=[];lo=[];hi=[]
    def add(d,lb=-np.inf,ub=0):
        row=len(lo)
        for j,v in d.items():ri.append(row);ci.append(j);vv.append(v)
        lo.append(lb);hi.append(ub)
    for j,row in c.iterrows():
        z=nP+j
        if row.origin_port not in pidx or row.dest_port not in pidx:
            add({z:1});continue
        for ep in {row.origin_port,row.dest_port}:add({z:1,pidx[ep]:-1})
        if row.mean_distance>8000:
            d={z:max(1,int(row.k_e_green_methanol))}
            for mid in mids.get(row.corridor_id,[]):d[pidx[mid]]=d.get(pidx[mid],0)-1
            add(d)
    A=coo_matrix((vv,(ri,ci)),shape=(len(lo),n)).tocsc()
    integ=np.r_[np.ones(nP),np.zeros(nC)];W=c.W.to_numpy()/1e6
    cost=p.cost_p.to_numpy();B=cost.sum()*.2
    def solve(weights,scope='G',domain=None,costscale=1,budget=B,port_charge=False,tie=True):
        Am=A;lom=np.array(lo);him=np.array(hi)
        if budget is not None:
            Am=vstack([A,coo_matrix(np.r_[cost*costscale,np.zeros(nC)].reshape(1,-1))],format='csc')
            lom=np.r_[lom,-np.inf];him=np.r_[him,budget]
        lb=np.zeros(n);ub=np.ones(n)
        if domain is not None:ub[:nP]=p.country_code.isin(domain).astype(float)
        primary=np.r_[-cost*costscale if port_charge else np.zeros(nP),weights]
        res=milp(-primary,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(Am,lom,him),options={'time_limit':180,'mip_rel_gap':1e-6})
        if res.x is None:raise RuntimeError(res.message)
        pv=-res.fun;pgap=res.mip_gap
        if scope!='G' and tie:
            Am=vstack([Am,coo_matrix(primary.reshape(1,-1))],format='csc')
            res=milp(-np.r_[np.zeros(nP),W],integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(Am,np.r_[lom,pv-1e-6],np.r_[him,np.inf]),options={'time_limit':180,'mip_rel_gap':1e-5})
        if res.x is None:raise RuntimeError(res.message)
        selected=p.loc[res.x[:nP]>.5,'port_code']
        if port_charge:
            active=(res.x[nP:]>.5)&(W>0)
        else:active=closure(c,selected,mids,feasible)&(W>0)
        return {'global_Mt':float(W[active].sum()),'n_ports':len(selected),'n_corridors':int(active.sum()),'cost_MUSD':float(cost[p.port_code.isin(selected)].sum()*costscale),'primary_value':float(pv),'primary_gap':float(pgap),'final_gap':float(res.mip_gap),'status':res.message}
    rows=[]
    for frac in [.10,.20,.30]:
        for scope in ['G','EU','BRI']:
            flag=np.ones(nC) if scope=='G' else (c.origin_port.map(country).isin(eu if scope=='EU' else bri)|c.dest_port.map(country).isin(eu if scope=='EU' else bri)).to_numpy()
            r=solve(W*flag,scope, bri if scope=='BRI' else None,budget=cost.sum()*frac)
            r.update(experiment='budget',scope=scope,value=frac);rows.append(r);print(r,flush=True)
    for scale in [.5,1,2]:
        r=solve(W,costscale=scale);r.update(experiment='fixed_absolute_budget_cost_scale',scope='G',value=scale);rows.append(r)
    # A narrower footprint sensitivity, not a claim to reconstruct a complete
    # legally authoritative BRI membership series.
    narrow=bri-{'IT','TW','IN'}
    flag=(c.origin_port.map(country).isin(narrow)|c.dest_port.map(country).isin(narrow)).to_numpy()
    for closed in [True,False]:
        r=solve(W*flag,'BRI',narrow if closed else None)
        r.update(experiment='narrower_network',scope='closed' if closed else 'open',value='exclude IT TW IN');rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'R1_sensitivity.csv',index=False)
    # Same abatement share appears in the fuel bill and in the emissions benefit.
    bp=pd.read_csv(Q/'breakeven_prices_container.csv').set_index('corridor_id')
    energy=c.corridor_id.map(bp.energy_kwh_e).to_numpy()
    alpha=c.alpha_e.to_numpy()
    market=[];floors=[]
    for methanol in [700,1200,1700]:
        premium=energy*alpha*(3.6/.50)/1000*(methanol/19.9-600/40.5)/1e6
        floor=np.divide(premium,W,out=np.full(nC,np.nan),where=W>0)
        floors.append({'methanol_usd_t':methanol,'min':float(np.nanmin(floor)),'median':float(np.nanmedian(floor)),'p25':float(np.nanquantile(floor,.25)),'p75':float(np.nanquantile(floor,.75))})
        for carbon in [100,380,700,1000,2000,5000]:
            for capital in ['constrained','unconstrained']:
                r=solve(carbon*W-premium,port_charge=True,budget=B if capital=='constrained' else None,tie=False)
                r.update(methanol_usd_t=methanol,carbon_usd_t=carbon,capital=capital);market.append(r)
                print('PRICE',methanol,carbon,capital,r['global_Mt'],flush=True)
    pd.DataFrame(market).to_csv(OUT/'R1_price_experiment.csv',index=False)
    pd.DataFrame(floors).to_csv(OUT/'R1_fuel_only_thresholds.csv',index=False)
    pd.DataFrame({'corridor_id':c.corridor_id,'energy_kwh':energy,'alpha':alpha,'W_t':W*1e6}).to_csv(OUT/'R1_energy_audit.csv',index=False)

if __name__=='__main__':main()
