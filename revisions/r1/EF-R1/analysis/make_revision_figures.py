"""Publication figures from audited tables; unchanged R0 maps are copied.
No values are digitized or invented. PDFs are vector except retained maps.
"""
from audit_activation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
import shapefile

DEST=HERE.parent
MAIN=DEST/'figures_main';SUPP=DEST/'figures_supporting'
ORIG=HERE/'inputs'/'reference_figures' if (HERE/'inputs'/'reference_figures').exists() else DEST.parent/'EF-R0'/'fig'
BLUE='#3564A5';GOLD='#C08C24';GREEN='#27806B';GREY='#737B85';RED='#B54B45'
# Match the original figures; fail explicitly if Times New Roman is unavailable.
for font_file in Path('C:/Windows/Fonts').glob('times*.ttf'):
    font_manager.fontManager.addfont(str(font_file))
font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.serif':['Times New Roman'],
    'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
    'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
    'mathtext.fallback':None,'font.size':10,'axes.labelsize':10,'axes.titlesize':11,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white'})

def save(fig,path):
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)

def main():
    MAIN.mkdir(parents=True,exist_ok=True)
    SUPP.mkdir(parents=True,exist_ok=True)
    for number,old in [(1,3),(2,4),(3,5),(4,6),(5,7),(7,14)]:
        if (SUPP/f'Figure_S{number:02}.pdf').exists():continue
        shutil.copy2(ORIG/f'Fig_{old:02}.png',SUPP/f'Figure_S{number:02}.png')
    # A plain-language diagram introducing objectives before notation.
    fig,ax=plt.subplots(figsize=(7.4,5.2));ax.set(xlim=(0,10),ylim=(0,10));ax.axis('off')
    def box(x,y,w,h,title,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=color,edgecolor='#B9C4CD',linewidth=.8))
        ax.text(x+w/2,y+h-.35,title,ha='center',va='top',weight='bold',fontsize=10)
        ax.text(x+w/2,y+h/2-.15,text,ha='center',va='center',fontsize=9.2,linespacing=1.45)
    box(.4,8.0,9.2,1.5,'Common observational network','Vessel activity  →  corridor emissions  →  eligible ports and costs','#EDF2F5')
    boxes=[(.4,'Global plan','Value reductions\non every corridor\n\nBuild at any feasible port','#E6F2EE'),(3.6,'EU priority plan','Value reductions\non EU connected routes\n\nBuild at any feasible port','#EAF0F8'),(6.8,'BRI network plan','Value reductions\non BRI linked routes\n\nBuild inside the network','#F7F0DF')]
    for x,title,txt,color in boxes:
        box(x,3.8,2.8,3.1,title,txt,color)
        ax.add_patch(FancyArrowPatch((x+1.4,7.95),(x+1.4,7.05),arrowstyle='-|>',mutation_scale=14,color=GREY))
        ax.add_patch(FancyArrowPatch((x+1.4,3.65),(5,2.7),arrowstyle='-|>',mutation_scale=14,color=GREY))
    box(.4,.8,9.2,1.8,'Common evaluation','Count every route supported by each port plan\nCompare worldwide and regional benefits at the same budget','#EDF2F5')
    ax.text(5,.15,'Control: allow the BRI plan to build outside its network, keeping its objective.',ha='center',fontsize=9)
    save(fig,MAIN/'Figure_01')

    c,p,m,g,mids,feasible=load()
    # Nonexclusive geographic overlap, avoiding the language of legal rule vacuums.
    eu=g.eu_ets_endpoint.astype(bool);bri=g.is_bri.astype(bool)
    masks=[eu&~bri,eu&bri,bri&~eu,~eu&~bri]
    names=['EU connected only','Both footprints','BRI linked only','Outside both']
    colors=[BLUE,GREEN,GOLD,GREY]
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.4),gridspec_kw={'width_ratios':[1,1.1]})
    for ax,values,title,xlabel in [(axs[0],[int(x.sum()) for x in masks],'a  Corridor counts','Number of corridors'),(axs[1],[g.loc[x,'E_e'].sum()/g.E_e.sum()*100 for x in masks],'b  Emission shares','Share of corridor emissions (%)')]:
        ax.barh(names,values,color=colors,height=.62);ax.invert_yaxis();ax.set_title(title,loc='left');ax.set_xlabel(xlabel)
        for j,v in enumerate(values):ax.text(v+max(values)*.025,j,f'{v:,.0f}' if ax==axs[0] else f'{v:.1f}%',va='center',fontsize=9)
        ax.set_xlim(0,max(values)*1.22);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axs[1].set_yticklabels([]);fig.tight_layout();save(fig,SUPP/'Figure_S05')

    a=pd.read_csv(OUT/'archived_plan_closure.csv').set_index('plan').loc[['Global','EU','BRI','BRI_open']]
    vals=a[['global_Mt','EU_Mt','BRI_Mt']].to_numpy()
    fig,ax=plt.subplots(figsize=(7.4,3.5));im=ax.imshow(vals,cmap='Blues',vmin=0,vmax=105,aspect='auto')
    ax.set_xticks(range(3),['Worldwide','EU connected','BRI linked']);ax.set_yticks(range(4),['Global plan','EU priority','Closed BRI network','Open BRI control'])
    ax.tick_params(length=0);ax.set_xlabel('Scope used to evaluate the same port plan')
    for i in range(4):
        for j in range(3):ax.text(j,i,f'{vals[i,j]:.2f}',ha='center',va='center',color='white' if vals[i,j]>60 else '#1F2933',fontsize=12,weight='bold')
        ax.text(2.75,i,f"{a.loss_pct.iloc[i]:.1f}% loss",va='center',fontsize=10,color=GREY)
    ax.set_xlim(-.5,3.5);ax.set_title('Enabled annual reductions (Mt CO₂e)',loc='left',pad=12)
    fig.tight_layout();save(fig,MAIN/'Figure_05')

    # Endpoint reliance: accessible labels and no claim of geopolitical control.
    # Reconstruct the complete endpoint table: the archived CSV contains only
    # the first 30 unsorted port records, not the highest-ranked ports.
    active=pd.read_csv(OUT/'closure_Global.csv')
    endpoints=pd.concat([active[['origin_port','W']].rename(columns={'origin_port':'port_code'}),active[['dest_port','W']].rename(columns={'dest_port':'port_code'})])
    fullhub=endpoints.groupby('port_code',as_index=False).W.sum()
    fullhub['HCS_pct']=100*fullhub.W/fullhub.W.sum()
    port_meta=pd.read_parquet(Q/'governance_ports_container.parquet')[['port_code','port_name_cn','country_code','camp']].drop_duplicates('port_code')
    fullhub=fullhub.merge(port_meta,on='port_code',how='left').sort_values('HCS_pct',ascending=False)
    fullhub.to_csv(OUT/'R1_endpoint_reliance.csv',index=False)
    hub=fullhub.head(20)
    regional=fullhub.groupby('camp').HCS_pct.sum().reindex(['BRI','Other','EU'])
    top20=hub.HCS_pct.sum()
    port_names={'SGSGP':'Singapore','KRBUS':'Busan','MAZAN':'Tanger Med','MYTPP':'Tanjung Pelepas','CNNBO':'Ningbo','NLROT':'Rotterdam','MYPKE':'Port Klang','CNYTN':'Yantian','LKCOL':'Colombo','CNQIN':'Qingdao','ESALG':'Algeciras','HKHKG':'Hong Kong','CNWGG':'Waigaoqiao','TWKAO':'Kaohsiung','USNWK':'Newark','MXMAN':'Manzanillo','ESVAL':'Valencia','BRSTS':'Santos','CNNSA':'Nansha'}
    port_names.update({'MATAN':'Tangier','CNYSN':'Shanghai Yangshan','CNWGQ':'Shanghai Waigaoqiao','USNYK':'New York'})
    fig,axs=plt.subplots(1,2,figsize=(7.4,5.6),gridspec_kw={'width_ratios':[1.45,1]})
    hcolors=hub.camp.map({'BRI':GOLD,'EU':BLUE,'Other':GREY})
    labels=[port_names.get(x,x) for x in hub.port_code]
    axs[0].barh(labels,hub.HCS_pct,color=hcolors,height=.72);axs[0].invert_yaxis();axs[0].set_xlabel('Endpoint reliance share (%)');axs[0].set_title('a  Twenty largest port shares',loc='left')
    axs[0].tick_params(axis='y',labelsize=8);axs[0].grid(axis='x',alpha=.18);axs[0].set_axisbelow(True)
    axs[1].bar(['BRI linked','Other','EU'],regional.to_numpy(),color=[GOLD,GREY,BLUE],width=.65)
    axs[1].set_ylim(0,65);axs[1].set_ylabel('Endpoint reliance share (%)');axs[1].set_title('b  Regional attribution',loc='left');axs[1].tick_params(axis='x',rotation=25)
    for i,v in enumerate(regional):axs[1].text(i,v+1,f'{v:.1f}%',ha='center')
    axs[1].text(.5,.93,f'Top 20 ports: {top20:.1f}%',transform=axs[1].transAxes,ha='center',fontsize=10,weight='bold')
    fig.tight_layout(w_pad=2);save(fig,MAIN/'Figure_08')

    prices=pd.read_csv(OUT/'R1_price_experiment.csv')
    fig,ax=plt.subplots(figsize=(7.4,4.0))
    for fuel,color in [(700,GREEN),(1200,BLUE),(1700,RED)]:
        for cap,ls in [('constrained','-'),('unconstrained','--')]:
            s=prices[(prices.methanol_usd_t==fuel)&(prices.capital==cap)]
            ax.plot(s.carbon_usd_t,s.global_Mt,ls,color=color,marker='o' if cap=='constrained' else None,markersize=4,label=f'Methanol USD {fuel:,}/t' if cap=='constrained' else None)
    ax.axvline(380,color=GREY,lw=.9);ax.text(390,7,'380',color=GREY,fontsize=9)
    ax.set_xscale('log');ax.set_xticks([100,380,700,1000,2000,5000],['100','380','700','1,000','2,000','5,000'])
    ax.set_ylim(-3,115);ax.set_ylabel('Annual reduction (Mt CO₂e)');ax.set_xlabel('Uniform carbon value (USD/t CO₂e)')
    ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=9,frameon=False)
    ax.text(.99,.02,'Solid: reference budget\nDashed: no budget ceiling',transform=ax.transAxes,ha='right',va='bottom',fontsize=9)
    fig.tight_layout();save(fig,MAIN/'Figure_10')

    # Port-only maps: no invented great-circle route geometry.
    land=HERE/'inputs'/'basemap'/'ne_110m_land.shp'
    if not land.exists():land=Path('C:/Users/mingW/.local/share/cartopy/shapefiles/natural_earth/physical/ne_110m_land.shp')
    pieces=[]
    for s in shapefile.Reader(str(land)).shapes():
        starts=list(s.parts)+[len(s.points)]
        for i in range(len(starts)-1):pieces.append(Polygon(s.points[starts[i]:starts[i+1]],closed=True))
    fig,axs=plt.subplots(3,1,figsize=(7.4,7.8))
    for ax,name,color in zip(axs,['Global','EU','BRI'],[GREEN,BLUE,GOLD]):
        ax.add_collection(PatchCollection(pieces,facecolor='#E8E6E1',edgecolor='#B1B4B6',linewidth=.25))
        src=R/'selected_ports_container.csv' if name=='Global' else Q/f"me{'U' if name=='EU' else 'B'}_selected_ports_container.csv"
        selected=pd.read_csv(src).port_code
        pts=p[p.port_code.isin(selected)]
        ax.scatter(pts.port_lon,pts.port_lat,s=12,color=color,edgecolors='white',linewidths=.25,zorder=3)
        row=a.loc[name];ax.set_title(f"{name if name=='Global' else ('EU priority' if name=='EU' else 'Closed BRI network')}   |   {len(selected)} ports   |   {row.global_Mt:.2f} Mt worldwide",loc='left',fontsize=10)
        ax.set(xlim=(-180,180),ylim=(-58,80));ax.set_facecolor('#F3F7FA');ax.set_xticks([-120,-60,0,60,120]);ax.set_yticks([-30,0,30,60]);ax.tick_params(labelsize=8)
    fig.tight_layout();save(fig,SUPP/'Figure_S06')

    front=pd.read_csv(Q/'moo_pareto_container_bri_share.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.4))
    axs[0].plot(front.bri_share_pct,front.eff_obj_Mt,'o-',color=GREEN);axs[0].set_ylabel('Worldwide reduction (Mt CO₂e)')
    axs[1].plot(front.bri_share_pct,front.vac_obj_Mt,'o-',color=BLUE);axs[1].set_ylabel('Reduction outside proxy coverage (Mt)')
    for ax in axs:ax.set_xlabel('Required BRI linked benefit share (%)');ax.grid(alpha=.18)
    axs[0].set_title('a  Aggregate benefit',loc='left');axs[1].set_title('b  Distributional consequence',loc='left');fig.tight_layout();save(fig,SUPP/'Figure_S08')

    sens=pd.read_csv(OUT/'R1_sensitivity.csv');fig,axs=plt.subplots(1,2,figsize=(7.4,3.5))
    for name,color in [('G',GREEN),('EU',BLUE),('BRI',GOLD)]:
        s=sens[(sens.experiment=='budget')&(sens.scope==name)]
        axs[0].plot(s.value.astype(float)*100,s.global_Mt,'o-',label={'G':'Global','EU':'EU priority','BRI':'Closed BRI'}[name],color=color)
    axs[0].set(xlabel='Budget fraction (%)',ylabel='Worldwide reduction (Mt CO₂e)',title='a  Budget sensitivity');axs[0].legend(frameon=False,fontsize=9)
    s=sens[sens.experiment=='fixed_absolute_budget_cost_scale'];axs[1].bar(['Half','Reference','Double'],s.global_Mt,color=[GREEN,BLUE,RED]);axs[1].set(xlabel='Port cost level at fixed USD budget',ylabel='Worldwide reduction (Mt CO₂e)',title='b  Cost level sensitivity',ylim=(0,120))
    for i,v in enumerate(s.global_Mt):axs[1].text(i,v+2,f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout();save(fig,SUPP/'Figure_S09')

    fig,axs=plt.subplots(1,2,figsize=(7.4,3.4))
    floor=pd.read_csv(OUT/'R1_fuel_only_thresholds.csv');axs[0].plot(floor.methanol_usd_t,floor['median'],'o-',color=BLUE);axs[0].axhline(380,color=GREY,ls='--');axs[0].set(xlabel='Methanol price (USD/t fuel)',ylabel='Fuel only threshold (USD/t CO₂e)',title='a  Consistent energy accounting')
    s=prices[(prices.carbon_usd_t==380)&(prices.capital=='constrained')]
    axs[1].bar([str(x) for x in s.methanol_usd_t],s.global_Mt,color=[GREEN,BLUE,RED]);axs[1].set(xlabel='Methanol price (USD/t fuel)',ylabel='Annual reduction (Mt CO₂e)',title='b  Deployment at USD 380/t CO₂e',ylim=(0,110))
    for i,v in enumerate(s.global_Mt):axs[1].text(i,v+2,f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout();save(fig,SUPP/'Figure_S10')

if __name__=='__main__':
    main()
    # Preserve the author's final main-text map selection after regeneration.
    from build_targeted_revision import restore_main_figures
    restore_main_figures()
    from make_revision_maps import main as draw_maps
    draw_maps()
