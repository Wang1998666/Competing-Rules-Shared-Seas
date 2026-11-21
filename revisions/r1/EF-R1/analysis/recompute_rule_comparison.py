"""R1 independent MILP: common scoring plus a transparent tie-breaking rule.

First maximize regional benefit. Among plans attaining that benefit to
1e-6 Mt, maximize global benefit. Uses scipy/HiGHS and the archived input
tables, without overwriting the original analysis. Run audit_activation.py
first. All objectives are Mt CO2e/year, costs M USD/year.
"""
from audit_activation import *
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack
import time, ast

def literal_set(filename, name):
    tree=ast.parse((ARCHIVE/'code'/'calc_v5'/filename).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)

def main():
    c,p,m,g,mids,feasible=load()
    p=p[p.is_feasible].reset_index(drop=True)
    c=c.reset_index(drop=True)
    eu=literal_set('03c_h3_policy_coverage.py','EU_EEA_COUNTRIES')
    bri=literal_set('08_bri_analysis.py','BRI_MARITIME_COUNTRIES')
    country=p.set_index('port_code').country_code
    c['EU']=c.origin_port.map(country).isin(eu)|c.dest_port.map(country).isin(eu)
    c['BRI']=c.origin_port.map(country).isin(bri)|c.dest_port.map(country).isin(bri)
    pidx={x:i for i,x in enumerate(p.port_code)}
    nP=len(p); nC=len(c); n=nP+nC
    ri=[]; ci=[]; vv=[]; lo=[]; hi=[]
    def add(coef,lb=-np.inf,ub=0):
        r=len(lo)
        for col,v in coef.items():ri.append(r);ci.append(col);vv.append(v)
        lo.append(lb);hi.append(ub)
    for j,row in c.iterrows():
        zj=nP+j
        if row.origin_port not in pidx or row.dest_port not in pidx:
            add({zj:1},ub=0);continue
        for ep in {row.origin_port,row.dest_port}:add({zj:1,pidx[ep]:-1})
        if row.mean_distance>8000:
            k=max(1,int(row.k_e_green_methanol)); coef={zj:k}
            for mid in mids.get(row.corridor_id,[]):
                coef[pidx[mid]]=coef.get(pidx[mid],0)-1
            add(coef)
    B=p.cost_p.sum()*.20
    add(dict(enumerate(p.cost_p)),ub=B)
    A=coo_matrix((vv,(ri,ci)),shape=(len(lo),n)).tocsc()
    integ=np.r_[np.ones(nP),np.zeros(nC)]
    W=c.W.to_numpy()/1e6
    out=[]
    variants=[('Global','G',False,False),('EU','EU',False,False),('BRI','BRI',True,False),('BRI_open','BRI',False,False),('Global_forced','G',False,True),('EU_forced','EU',False,True)]
    for name,scope,closed,force in variants:
        lb=np.zeros(n);ub=np.ones(n)
        if closed:ub[:nP]=p.country_code.isin(bri).astype(float)
        if force:
            eu_codes=set(p.loc[p.country_code.isin(eu),'port_code'])
            counts=pd.concat([c.loc[c.origin_port.isin(eu_codes),'origin_port'],c.loc[c.dest_port.isin(eu_codes),'dest_port']]).value_counts()
            ranked=[x for x in counts.index if x in pidx]
            cum=np.cumsum([p.cost_p.iloc[pidx[x]] for x in ranked])
            nforce=min(100,len(ranked),max(int((cum<=.8*B).sum()),1))
            for x in ranked[:nforce]:lb[pidx[x]]=1
        primary=W*(1 if scope=='G' else c[scope].to_numpy())
        obj=np.r_[np.zeros(nP),-primary]
        t=time.time()
        print('SOLVING',name,'primary',flush=True)
        res=milp(obj,integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(A,lo,hi),options={'time_limit':240,'mip_rel_gap':1e-6})
        if res.x is None:raise RuntimeError((name,res.message))
        primary_val=-res.fun
        primary_gap=float(res.mip_gap)
        primary_bound=-float(res.mip_dual_bound)
        primary_status=res.message
        if scope!='G':
            print('SOLVING',name,'secondary',primary_val,flush=True)
            Aprim=coo_matrix(np.r_[np.zeros(nP),primary].reshape(1,-1)).tocsc()
            secondary=milp(np.r_[np.zeros(nP),-W],integrality=integ,bounds=Bounds(lb,ub),constraints=LinearConstraint(vstack([A,Aprim],format='csc'),np.r_[lo,primary_val-1e-6],np.r_[hi,np.inf]),options={'time_limit':240,'mip_rel_gap':1e-5})
            if secondary.x is None:raise RuntimeError((name,secondary.message))
            res=secondary
        selected=p.loc[res.x[:nP]>.5,'port_code']
        ok=closure(c,selected,mids,feasible)
        act=c[ok & (c.W>0)]
        row={'plan':name,'global_Mt':act.W.sum()/1e6,'EU_Mt':act.loc[act.EU,'W'].sum()/1e6,'BRI_Mt':act.loc[act.BRI,'W'].sum()/1e6,'n_ports':len(selected),'n_positive_corridors':len(act),'cost_MUSD':p.loc[p.port_code.isin(selected),'cost_p'].sum(),'budget_MUSD':B,'primary_incumbent_Mt':primary_val,'primary_upper_bound_Mt':primary_bound,'primary_gap':primary_gap,'final_gap':float(res.mip_gap),'primary_status':primary_status,'final_status':res.message,'elapsed_s':time.time()-t}
        assert row['cost_MUSD']<=B+1e-5
        if scope!='G':assert row[scope+'_Mt']>=primary_val-2e-6
        out.append(row)
        pd.DataFrame({'port_code':selected}).to_csv(OUT/f'R1_ports_{name}.csv',index=False)
        act.to_csv(OUT/f'R1_corridors_{name}.csv',index=False)
        pd.DataFrame(out).to_csv(OUT/'R1_rule_comparison.csv',index=False)
        (OUT/'R1_rule_comparison.json').write_text(json.dumps(out,indent=2),encoding='utf8')
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
