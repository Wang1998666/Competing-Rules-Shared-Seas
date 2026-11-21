from pathlib import Path
import sys, json, hashlib, argparse
sys.dont_write_bytecode = True
import map_base as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib import patheffects
import pymupdf as fitz

ROOT=Path(__file__).resolve().parent
TABLES=ROOT/'panel_data'
TABLES.mkdir(exist_ok=True)
m.plt.rcParams.update({'axes.labelcolor':m.INK,'text.color':m.INK,'xtick.color':m.MUTED,'ytick.color':m.MUTED})
GRID='#D5DEE1'; PALE='#E4ECEB'; INK=m.INK; SOFT='#DAE8E6'
CATS=['EU','BRI','Other']; NAMES=['EU','BRI linked','Other']
PLANS=['Global','EU','BRI']; PLAN_LABEL=['Global','EU priority','Closed BRI']
CLASS=[('EU_ETS','EU connected',m.EU),('BRI_only','BRI linked only',m.BRI),('FuelEU_exposed','Near EU',m.NEAR),('Vacuum','Outside footprints',m.OTHER)]
META={}
def table(key,frame):
    frame.to_csv(TABLES/(key+'.csv'),index=False,float_format='%.10g')
def new(height):return plt.figure(figsize=(8.1,height))
def title(fig,x,y,letter,label):
    fig.text(x,y,letter,fontsize=10.7,weight='bold',color=INK,va='center')
    fig.text(x+.032,y,label,fontsize=10,weight='bold',va='center',color=INK)
def chart(fig,box,grid='x'):
    ax=fig.add_axes(box);ax.set_axisbelow(True)
    ax.grid(axis=grid,color=GRID,lw=.45,alpha=.8)
    for key,spine in ax.spines.items():
        spine.set_visible(key=='bottom');spine.set_color('#A6B4BA')
    ax.tick_params(length=0,pad=3,labelsize=8.2)
    return ax
def key(fig,colors,labels,xy,styles=None,size=8.5):
    styles=styles or ['o']*len(colors)
    handles=[Line2D([],[],ls='',marker=shape,markersize=5.4,markerfacecolor=color,
         markeredgecolor=color if shape=='x' else 'white',markeredgewidth=.65) for color,shape in zip(colors,styles)]
    fig.legend(handles,labels,loc='center left',bbox_to_anchor=xy,ncol=len(labels),frameon=False,
        handlelength=.85,handletextpad=.34,columnspacing=1.1,fontsize=size,borderaxespad=0)
def region(ax,name):
    x0,x1,y0,y1=m.REGIONS[name]
    ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],transform=m.PC,color=INK,lw=.65,ls=(0,(3,2)),zorder=9)
def save(fig,id,caption,panels,zooms):
    if id in ['2','4','S6']:
        caption+=' The fine blue-gray line in regional panels marks the 200 m depth contour (Natural Earth).'
    folder=m.SI if id.startswith('S') else m.MAIN
    stem='Figure_'+('S' if id.startswith('S') else '')+f'{int(id.strip("S")):02d}'
    for ext in ['pdf','png']:
        fig.savefig(folder/(stem+'.'+ext),dpi=400 if ext=='pdf' else 350,facecolor='white')
    plt.close(fig)
    META[id]={'file':str((folder/(stem+'.pdf')).relative_to(ROOT).as_posix()),'caption':caption,'map_panels':panels,'enlargements':zooms}
    print('Saved '+stem,flush=True)
def selection_legend(fig,xy):
    handles=[Line2D([],[],marker='o',ms=2.6,ls='',color='#98A8AD'),Line2D([],[],color='#437976',lw=.65)]
    fig.legend(handles,['Feasible port','Shown supported route'],loc='center left',bbox_to_anchor=xy,ncol=2,frameon=False,fontsize=7.8,handlelength=1.1,handletextpad=.32,columnspacing=1,borderaxespad=0)
def fit_outer(ax):ax.set_xlim(left=0);ax.xaxis.set_major_locator(MaxNLocator(4,integer=True))
def quantiles(ax,samples,colors,labels):
    data=[]
    for row,(vals,col,name) in enumerate(zip(samples,colors,labels)):
        lo,q1,med,q3,hi=np.quantile(vals,[.1,.25,.5,.75,.9])
        ax.hlines(row,lo,hi,color=col,lw=1);ax.plot([q1,q3],[row,row],color=col,lw=5,alpha=.52,solid_capstyle='butt')
        ax.scatter(med,row,s=18,color=col,edgecolors='white',linewidths=.5,zorder=3)
        data.append([name,lo,q1,med,q3,hi])
    ax.set_yticks(range(len(labels)),labels);ax.set_ylim(len(labels)-.48,-.5)
    ax.set_xscale('log');ax.tick_params(which='minor',length=0)
    return pd.DataFrame(data,columns=['group','p10','p25','median','p75','p90'])
def selected(ax,name):
    valid=m.CAND[m.CAND.is_feasible]
    m.points(ax,valid,'#7E939D',1.8,alpha=.30,edge='none',zorder=3)
    frame=m.PORTS[m.PORTS.port_code.isin(m.PLANS[name])]
    for camp,col in m.CAMP.items():
        m.points(ax,frame[frame.camp==camp],col,8,alpha=.79)
def paths(ax,frame,limit,weight,col):
    ids=m.route_ids(frame,limit,weight)
    m.routes(ax,ids,col,.58,.49)
def concentration(ax,weights,mark,xlabel):
    weights=np.sort(np.array(weights))[::-1]
    shares=100*np.cumsum(weights)/weights.sum();ranks=np.arange(1,len(weights)+1)
    assert np.isclose(shares[-1],100)
    ax.fill_between(ranks,0,shares,color='#567583',alpha=.13)
    ax.plot(ranks,shares,color='#345463',lw=1.65)
    ax.hlines(shares[mark-1],1,mark,color='#718993',lw=.58,ls=(0,(2,2)))
    ax.vlines(mark,0,shares[mark-1],color='#718993',lw=.58,ls=(0,(2,2)))
    ax.scatter(mark,shares[mark-1],s=28,color='#345463',edgecolors='white',linewidths=.65,zorder=3)
    ax.annotate(f'Top {mark}: {shares[mark-1]:.1f}%',(mark,shares[mark-1]),xytext=(7,-12),textcoords='offset points',fontsize=8.5,color=INK)
    ax.set_xscale('log');ax.set_xlim(1,len(weights));ax.set_ylim(0,102);ax.set_yticks([0,50,100],['0','50','100%'])
    ticks=[v for v in [1,10,100,1000] if v<=len(weights)]
    ax.set_xticks(ticks,labels=[f'{v:,}' for v in ticks]);ax.tick_params(which='minor',length=0)
    ax.set_xlabel(xlabel,fontsize=8.4)
    return pd.DataFrame({'rank':ranks,'cumulative_pct':shares})
def fig9():
    fig=new(5.40);title(fig,.032,.952,'a','Reliance on shared ports');title(fig,.639,.952,'c','Twenty largest port shares')
    world=m.map_ax(fig,[.032,.49,.54,.40])
    data=m.HUB.sort_values('HCS_pct',ascending=False)
    for camp,col in m.CAMP.items():
        block=data[data.camp==camp];area=np.maximum(1.4,block.HCS_pct*23)
        world.scatter(block.port_lon,block.port_lat,s=area,c=col,alpha=.65,edgecolors='white',linewidths=.27,transform=m.PC,zorder=5)
        sel=block[block.port_code.isin(m.PLANS['Global'])]
        world.scatter(sel.port_lon,sel.port_lat,s=np.maximum(1.4,sel.HCS_pct*23),facecolors='none',edgecolors=col,linewidths=.53,transform=m.PC,zorder=6)
    ranks={code:rank for rank,code in enumerate(data.port_code,1)}
    for code,delta in [('SGSGP',(9,-16)),('MATAN',(-25,-15)),('KRBUS',(11,12))]:
        row=data[data.port_code==code].iloc[0];coord=world.projection.transform_point(row.port_lon,row.port_lat,m.PC)
        label=world.annotate(f'{ranks[code]}  {m.PORT_NAMES[code]}',coord,xytext=delta,textcoords='offset points',fontsize=7.4,
            arrowprops=dict(arrowstyle='-',lw=.45,color=INK),zorder=10)
        label.set_path_effects([patheffects.withStroke(linewidth=1.8,foreground='white')])
    handles=[Line2D([],[],ls='',marker='o',markersize=np.sqrt(23*v),mfc='none',mec=m.MUTED,mew=.6) for v in [1,5,10]]
    fig.legend(handles,['1%','5%','10%'],loc='center left',bbox_to_anchor=(.035,.49),ncol=3,fontsize=7.8,frameon=False,columnspacing=.7)
    group_shares=data.groupby('camp').HCS_pct.sum().reindex(CATS)
    assert np.isclose(group_shares.sum(),100)
    key(fig,list(m.CAMP.values()),[f'{name}  {share:.1f}%' for name,share in zip(NAMES,group_shares)],(.032,.418))
    table('09a_group_shares',group_shares.rename('reliance_pct').reset_index())
    title(fig,.032,.32,'b','Cumulative share of endpoint reliance')
    ax=chart(fig,[.09,.09,.46,.175],grid='y')
    table('09b_port_concentration',concentration(ax,data.loc[data.HCS_pct>0,'HCS_pct'],20,'Port rank (log scale)'))
    top=data.head(20).iloc[::-1].copy();ax=chart(fig,[.753,.122,.196,.755])
    for idx,row in enumerate(top.itertuples()):
        color=m.CAMP[row.camp]
        ax.barh(idx,row.HCS_pct,height=.51,color=color,alpha=.66 if idx!=19 else .98)
        ax.text(row.HCS_pct+.18,idx,f'{row.HCS_pct:.2f}',va='center',fontsize=7.3,color=INK)
    top['name']=top.port_code.map(m.PORT_NAMES);assert top.name.notna().all()
    ax.set_yticks(range(20),top.name,fontsize=7.7);ax.set_ylim(-.6,19.7);ax.set_xlim(0,13.9);ax.set_xticks([0,5,10]);ax.set_xlabel('Endpoint reliance (%)',fontsize=8.3)
    for pos,code in enumerate(top.port_code):
        ax.text(-.61,pos,f'{ranks[code]:02d}',transform=ax.get_yaxis_transform(),ha='right',
                va='center',fontsize=7.1,color=m.MUTED)
    table('09c_port_ranking',top[['port_code','name','camp','HCS_pct']].iloc[::-1])
    save(fig,'9','Endpoint reliance in the global plan. (a) Symbol area gives each port’s share, with a minimum visible size and an outline for selected ports. (b) Cumulative reliance across ports with positive shares, ranked from largest to smallest; the horizontal axis is logarithmic. (c) The twenty largest shares. Numbers beside the three map labels match their ranks in panel c. Geographic group colors are consistent throughout; each route’s benefit is divided equally between its endpoints.',1,0)
def map_port(ax,code,name,offset):
    row=m.PORTS.loc[m.PORTS.port_code==code].iloc[0]
    coord=ax.projection.transform_point(row.port_lon,row.port_lat,m.PC)
    ax.scatter(*coord,s=8,facecolors='white',edgecolors=INK,linewidths=.65,zorder=10)
    label=ax.annotate(name,coord,xytext=offset,textcoords='offset points',fontsize=7.2,
        ha='left' if offset[0]>0 else 'right',va='center',color=INK,zorder=11,
        arrowprops=dict(arrowstyle='-',color=INK,lw=.48,shrinkA=1,shrinkB=2))
    label.set_path_effects([patheffects.withStroke(linewidth=1.8,foreground='#FBFBF9')])
def distance_bar(ax,km):
    from pyproj import Geod
    west,east,south,north=ax.get_extent(m.PC)
    left=west+.055*(east-west);bottom=south+.062*(north-south)
    geod=Geod(ellps='WGS84')
    end,_,_=geod.fwd(left,bottom,90,km*1000)
    ax.plot([left,end],[bottom,bottom],color=INK,lw=1.65,zorder=12,
            path_effects=[patheffects.withStroke(linewidth=2.9,foreground='white'),patheffects.Normal()])
    text=ax.annotate(f'{km} km',((left+end)/2,bottom),xytext=(0,3),textcoords='offset points',
            ha='center',color=INK,fontsize=6.8,zorder=12)
    text.set_path_effects([patheffects.withStroke(linewidth=1.7,foreground='white')])
def fig2():
    fig=new(6.02)
    title(fig,.033,.966,'a','Container-shipping emissions, 2024')
    fig.text(.967,.966,f'{m.GRID.emission_tco2e.sum()/1e6:,.1f} Mt CO₂e  ·  Global inventory',
        fontsize=8.5,ha='right',va='center',color=m.MUTED)
    world=m.map_ax(fig,[.035,.491,.93,.435]);raster=m.heat_raster()
    assert m.CHECKS['display_grid_conserves_emissions']
    m.HEAT=LinearSegmentedColormap.from_list('shipping_emissions',['#E4ECED','#B6CED0','#62A7AB','#D8B46C','#B56F3D','#743D39'])
    mesh=m.heat(world,raster)
    views=[('Europe',500,[('NLROT','Rotterdam',(6,5)),('ESALG','Algeciras',(5,-5))]),
           ('Malacca Strait',250,[('MYPKE','Port Klang',(-7,6)),('SGSGP','Singapore',(7,-6))]),
           ('East Asia',300,[('CNYSN','Yangshan',(-6,-5)),('KRBUS','Busan',(-6,8))])]
    records=[]
    for idx,(name,km,ports) in enumerate(views):
        letter=chr(98+idx);m.region_box(world,name,letter)
        left=.032+idx*.323
        title(fig,left,.448,letter,name)
        ax=m.map_ax(fig,[left,.145,.291,.264],name)
        m.heat(ax,raster)
        ax.add_feature(m.LAND50,facecolor='none',edgecolor='#8A999B',linewidth=.22,zorder=4)
        for code,label,offset in ports:map_port(ax,code,label,offset)
        distance_bar(ax,km)
        records.append([letter,name,*m.REGIONS[name],km])
    colorbar=fig.colorbar(mesh,cax=fig.add_axes([.246,.084,.508,.014]),orientation='horizontal',extend='both')
    colorbar.set_ticks([10,100,1000,10000,100000],labels=['10','100','1,000','10,000','100,000'])
    colorbar.ax.tick_params(length=2,labelsize=7.9)
    colorbar.set_label('Annual emissions per 0.25° display cell (t CO₂e)',fontsize=8.5,labelpad=2)
    table('02_regional_windows',pd.DataFrame(records,columns=['panel','region','west','east','south','north','scale_bar_km']))
    save(fig,'2','Annual emissions from the 2024 H3 shipping inventory: (a) global overview; (b–d) Europe, the Malacca Strait and East Asia. The dashed windows on the overview identify the three regional panels. All maps use identical 0.25° display aggregation and the same logarithmic color scale. Named ports are geographic landmarks, not individual emission estimates. Scale bars are local to each panel.',4,3)
def fig3():
    fig=new(5.45)
    stats=[]
    for idx,(tag,label,col) in enumerate(CLASS):
        x=.035+.49*(idx%2);y=.70-.32*(idx//2)
        title(fig,x,y+.245,chr(97+idx),label)
        ax=m.map_ax(fig,[x,y,.44,.203]);block=m.GOV[m.GOV.gov_type==tag]
        ep=set(block.origin_port)|set(block.dest_port);m.points(ax,m.PORTS[m.PORTS.port_code.isin(ep)],col,2.1,.28,edge='none',zorder=3)
        paths(ax,block,60,'E_e',col)
        fig.text(x+.437,y+.245,f'{len(block):,} corridors',ha='right',va='center',fontsize=7.7,color=m.MUTED)
        stats.append((label,len(block),100*len(block)/len(m.GOV),100*block.E_e.sum()/m.GOV.E_e.sum()))
    title(fig,.035,.305,'e','Corridor share and emission share')
    ax=chart(fig,[.24,.096,.67,.17]);df=pd.DataFrame(stats,columns=['group','corridors','corridor_pct','emission_pct'])
    for idx,row in enumerate(df.itertuples()):
        col=CLASS[idx][2];ax.plot([row.corridor_pct,row.emission_pct],[idx,idx],color=col,lw=1.8,alpha=.65)
        ax.scatter(row.corridor_pct,idx,s=25,facecolors='white',edgecolors=col,linewidths=1,zorder=3)
        ax.scatter(row.emission_pct,idx,s=29,c=col,edgecolors='white',linewidths=.4,zorder=4)
        ax.text(max(row.emission_pct,row.corridor_pct)+1.2,idx,f'{row.emission_pct:.1f}%',va='center',fontsize=7.8)
    ax.set_ylim(3.55,-.55);ax.set_xlim(0,67);ax.set_xticks([0,20,40,60]);ax.set_yticks(range(4),df.group)
    fig.text(.24,.033,'Open: share of corridors     •     Filled: share of emissions',fontsize=8.3,color=m.MUTED)
    assert np.isclose(df.emission_pct.sum(),100) and df.corridors.sum()==len(m.GOV)
    table('03e_classification',df)
    save(fig,'3','Four mutually exclusive corridor classes are shown at the same extent (up to 60 high-emission archived routes per group; dots show endpoints in the full class). Panel e contrasts each class’s share of all corridors with its share of total corridor emissions. The near-EU designation is a proximity proxy.',4,0)
def fig4():
    fig=new(4.35);title(fig,.035,.941,'a','Overlap of EU and BRI footprints')
    world=m.map_ax(fig,[.032,.36,.574,.51])
    title(fig,.035,.357,'b','Approaches to Europe')
    local=m.map_ax(fig,[.035,.075,.48,.252],'Europe')
    groups=m.overlap_groups()
    for ax in (world,local):
        for idx in [3,0,2,1]:
            tag,block,col=groups[idx];m.routes(ax,m.route_ids(block,75),col,.69 if idx==1 else .44,.77 if idx==1 else .39)
    m.region_box(world,'Europe','b')
    map_port(local,'NLROT','Rotterdam',(4,5))
    map_port(local,'ESALG','Algeciras',(3,-5))
    title(fig,.63,.943,'c','Corridors and emissions')
    frame=pd.DataFrame([(name,len(block),100*block.E_e.sum()/m.GOV.E_e.sum(),col) for name,block,col in groups],columns=['group','corridors','emission_pct','color'])
    for idx,row in enumerate(frame.itertuples()):
        ypos=.749-.182*idx
        fig.text(.65,ypos+.053,row.group,fontsize=9,color=row.color,weight='bold')
        fig.text(.65,ypos,f'{row.corridors:,} routes',fontsize=8.2)
        fig.text(.946,ypos,f'{row.emission_pct:.1f}%',ha='right',fontsize=8.5,color=INK)
        ax=fig.add_axes([.65,ypos-.039,.298,.018]);ax.axis('off');ax.set_xlim(0,100);ax.barh([0],100,color='#E7EBEC');ax.barh([0],row.emission_pct,color=row.color,alpha=.86)
    fig.text(.65,.134,'Bars: share of corridor emissions',fontsize=8.1,color=m.MUTED)
    key(fig,[gr[2] for gr in groups],[gr[0] for gr in groups],(.035,.036),size=8.2)
    table('04c_overlap',frame.drop(columns='color'));assert frame.corridors.sum()==len(m.GOV) and np.isclose(frame.emission_pct.sum(),100)
    save(fig,'4','Nonexclusive EU and BRI footprints shown as four disjoint combinations. The world map and European enlargement display up to 75 high-emission archived routes per combination. Counts and bars refer to the complete corridor inventory; the BRI footprint is an infrastructure scenario.',2,1)
def fig6():
    fig=new(5.47)
    stats=[]
    for idx,name in enumerate(PLANS):
        y=.66-.29*idx;title(fig,.03,y+.282,chr(97+idx),PLAN_LABEL[idx]+' plan')
        ax=m.map_ax(fig,[.029,y+.035,.486,.219])
        closure=pd.read_csv(m.OUT/f'closure_{name}.csv');positive=closure[closure.W>0]
        paths(ax,positive,60,'W','#437976');selected(ax,name)
        sub=m.PORTS[m.PORTS.port_code.isin(m.PLANS[name])];counts=sub.camp.value_counts()
        assert len(sub)==len(m.PLANS[name])==counts.sum()
        fig.text(.518,y+.282,f'{len(sub)} ports  ·  {len(positive):,} routes',fontsize=7.9,color=m.MUTED,ha='right',va='center')
        stats.append([name,len(sub),len(positive),m.STATS.loc[name,'global_Mt'],*[int(counts.get(gr,0)) for gr in CATS]])
    frame=pd.DataFrame(stats,columns=['plan','ports','positive_routes','annual_Mt','EU','BRI','Other'])
    title(fig,.57,.942,'d','Worldwide reduction')
    ax=chart(fig,[.70,.606,.243,.237])
    for idx,val in enumerate(frame.annual_Mt):
        ax.barh(idx,val,color=['#315C6E','#6797A3','#9CBAC0'][idx],height=.53)
        ax.text(val+2,idx,f'{val:.2f}',va='center',fontsize=8)
    ax.set_xlim(0,125);ax.set_ylim(2.5,-.5);ax.set_yticks(range(3),PLAN_LABEL);ax.set_xticks([0,50,100]);ax.set_xlabel('Mt CO₂e / yr',fontsize=8.5)
    title(fig,.57,.476,'e','Selected ports by group')
    ax=chart(fig,[.70,.167,.253,.218])
    for idx,row in frame.iterrows():
        left=0
        for camp in CATS:
            val=row[camp];ax.barh(idx,val,left=left,height=.55,color=m.CAMP[camp],alpha=.9)
            if val>=20:ax.text(left+val/2,idx,str(val),ha='center',va='center',color='white',fontsize=7.6)
            left+=val
    ax.set_ylim(2.5,-.5);ax.set_yticks(range(3),PLAN_LABEL);ax.set_xlabel('Number of ports',fontsize=8.4);ax.set_xlim(0,245);ax.set_xticks([0,100,200])
    key(fig,list(m.CAMP.values()),NAMES,(.565,.071))
    selection_legend(fig,(.029,.071))
    table('06de_plan_comparison',frame)
    save(fig,'6','The three port investment plans at the same global extent. Colored points are selected ports; faint points are feasible candidates. Up to 60 supported archived routes per plan are ranked by reduction benefit. Panel d compares worldwide reduction on a common basis; panel e counts the selected ports by geographic group. Map annotations give total ports and positive-benefit supported routes.',3,0)
def figS1():
    import shutil
    approved=ROOT.parent/'EF-Figures-v4-S1'
    for suffix in ['pdf','png']:
        source=approved/f'Figure_S01.{suffix}'
        target=m.SI/source.name
        shutil.copyfile(source,target)
        assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
    for name in ['grid_display_scope.csv','S1_ports.csv']:
        shutil.copyfile(approved/name,TABLES/name)
    caption=(approved/'Caption_S1.txt').read_text(encoding='utf8').strip().removeprefix('Figure S1. ')
    META['S1']={'file':'figures_supporting/Figure_S01.pdf','caption':caption,'map_panels':2,'enlargements':0}
def figS2():
    fig=new(4.7);title(fig,.031,.935,'a','Near-EU route proximity');title(fig,.516,.935,'b','Ports changing selection')
    first=m.map_ax(fig,[.03,.57,.45,.31]);second=m.map_ax(fig,[.516,.57,.45,.31])
    sample=m.GOV[m.GOV.corridor_id.isin(m.PATHS)].nlargest(150,'E_e')
    cmap=LinearSegmentedColormap.from_list('prox',['#A8C4CA','#79639D','#5A386D'])
    for row in sample.itertuples():m.routes(first,[row.corridor_id],cmap(float(row.fueleu_max)),.57,.76)
    changes=pd.read_csv(m.DATA/'policy_analysis_v5/fueleu_port_changes_container.csv')
    assert not changes.port_code.duplicated().any()
    for status,col,marker in [('added',m.BRI,'o'),('removed',m.OTHER,'x')]:
        m.points(second,changes[changes.change==status],col,9,.81,marker=marker)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap),cax=fig.add_axes([.114,.543,.26,.012]),orientation='horizontal')
    cb.set_ticks([0,.5,1],labels=['0','0.5','1']);cb.ax.tick_params(length=2,labelsize=7)
    key(fig,[m.BRI,m.OTHER],[f'Added ({(changes.change=="added").sum()})',f'Removed ({(changes.change=="removed").sum()})'],(.585,.55),['o','x'])
    title(fig,.033,.44,'c','Port changes by region')
    frame=changes.groupby(['continent','change']).size().unstack(fill_value=0).sort_values(by=['added','removed'],ascending=False)
    ax=chart(fig,[.211,.118,.69,.26])
    pos=np.arange(len(frame));ax.barh(pos,frame.added,color=m.BRI,alpha=.78,height=.64);ax.barh(pos,-frame.removed,color=m.OTHER,alpha=.78,height=.64)
    ax.axvline(0,color=INK,lw=.55);ax.set_xlim(-58,91);ax.set_xticks([-40,-20,0,20,40,60,80],['40','20','0','20','40','60','80'])
    ax.set_yticks(pos,frame.index);ax.set_ylim(len(pos)-.45,-.55);ax.set_xlabel('Removed  ←     Number of ports     →  Added',fontsize=8.6)
    for idx,(_,row) in enumerate(frame.iterrows()):
        if row.added:ax.text(row.added+1,idx,str(row.added),va='center',fontsize=7.5)
        if row.removed:ax.text(-row.removed-1,idx,str(row.removed),va='center',ha='right',fontsize=7.5)
    table('S02c_changes',frame.reset_index())
    save(fig,'S2','Maximum near-EU proximity on 150 high-emission archived routes, and ports added or removed in the archived priority scenario. Panel c counts changed ports by region, using the archive’s geographic grouping. This proximity proxy is not a legal boundary.',2,0)
def figS3():
    fig=new(4.07);blocks=[m.GOV[m.GOV.is_bri.astype(bool)],m.GOV[~m.GOV.is_bri.astype(bool)]]
    labels=['BRI linked','Outside BRI']
    for idx,(block,col) in enumerate(zip(blocks,[m.BRI,m.OTHER])):
        x=.032+idx*.488;title(fig,x,.928,chr(97+idx),labels[idx]);ax=m.map_ax(fig,[x,.51,.451,.36])
        paths(ax,block,70,'E_e',col)
        fig.text(x+.444,.928,f'{len(block):,} corridors',ha='right',va='center',fontsize=7.7,color=m.MUTED)
        ep=set(block.origin_port)|set(block.dest_port);m.points(ax,m.PORTS[m.PORTS.port_code.isin(ep)],col,2.2,.28,edge='none',zorder=3)
    title(fig,.035,.422,'c','Annual emissions per corridor')
    ax=chart(fig,[.225,.16,.675,.174]);samples=[block.E_e.to_numpy() for block in blocks];assert all((vals>0).all() for vals in samples)
    table('S03c_distributions',quantiles(ax,samples,[m.BRI,m.OTHER],labels))
    ax.set_xlabel('Tonnes CO₂e / yr (log scale)',fontsize=8.4)
    fig.text(.225,.033,'Line: 10th–90th percentile     Band: interquartile range     Dot: median',fontsize=7.9,color=m.MUTED)
    save(fig,'S3','Routes inside and outside the archive’s nonexclusive BRI footprint. Maps show up to 70 high-emission archived routes per group. Panel c summarizes annual corridor emissions across the full groups: 10th–90th percentile, interquartile range and median.',2,0)
def figS4():
    fig=new(3.65);title(fig,.03,.94,'a','Global investment plan');title(fig,.62,.94,'b','Port selection rates')
    world=m.map_ax(fig,[.032,.23,.535,.54]);positive=pd.read_csv(m.OUT/'closure_Global.csv').query('W > 0')
    paths(world,positive,80,'W','#477D81');selected(world,'Global')
    cand=m.CAND[m.CAND.is_feasible].merge(m.PORTS[['port_code','camp']],on='port_code',validate='one_to_one',how='left')
    cand['selected']=cand.port_code.isin(m.PLANS['Global'])
    assert cand.camp.notna().all() and cand.selected.sum()==len(m.PLANS['Global'])
    frame=cand.groupby('camp').selected.agg(['sum','size']).reindex(CATS);frame.columns=['selected','feasible'];frame['rate_pct']=100*frame.selected/frame.feasible
    ax=chart(fig,[.712,.30,.228,.44])
    for idx,(_,row) in enumerate(frame.iterrows()):
        ax.barh(idx,row.rate_pct,color=list(m.CAMP.values())[idx],height=.53,alpha=.84)
        ax.text(row.rate_pct+.7,idx,f'{int(row.selected)}/{int(row.feasible)}',fontsize=7.9,va='center')
    ax.set_yticks(range(3),NAMES);ax.set_ylim(2.5,-.5);ax.set_xlim(0,frame.rate_pct.max()*1.48);fit_outer(ax);ax.set_xlabel('Selected / feasible (%)',fontsize=8.3)
    key(fig,list(m.CAMP.values()),NAMES,(.035,.13))
    selection_legend(fig,(.035,.068))
    table('S04b_selection_rates',frame.reset_index())
    save(fig,'S4','The reference global plan, with up to 80 supported archived routes. Bars show the percentage of feasible ports selected within each geographic group; text gives the selected and feasible counts.',1,0)
def figS6():
    fig=new(2.83)
    for idx,name in enumerate(PLANS):
        x=.03+idx*.326
        title(fig,x,.906,chr(97+idx),PLAN_LABEL[idx]+' plan')
        ax=m.map_ax(fig,[x,.20,.287,.59],'Europe');selected(ax,name)
        for lon,lat,label in [(1.5,58.5,'North Sea'),(7.2,38.2,'Mediterranean')]:
            art=ax.text(lon,lat,label,ha='center',fontsize=6.8,color='#507983',fontstyle='italic',zorder=8)
            art.set_path_effects([patheffects.withStroke(linewidth=1.7,foreground='white')])
        distance_bar(ax,500)
    key(fig,list(m.CAMP.values()),NAMES,(.31,.096))
    save(fig,'S6','European port selections under the three investment plans at a common extent. Faint points are feasible candidates. The three matched views retain regional detail that is no longer repeated in main Figure 6.',3,3)
def figS7():
    fig=new(3.95);title(fig,.03,.929,'a','Inventory and path coverage');title(fig,.603,.929,'b','High-emission cells off path')
    world=m.map_ax(fig,[.03,.255,.51,.52])
    grid=m.GRID[(m.GRID.emission_tco2e>0)&(m.GRID.lat.abs()<72)].copy()
    threshold=grid.emission_tco2e.quantile(.75);high=grid[grid.emission_tco2e>=threshold].copy()
    cover=set(pd.read_parquet(m.DATA/'h3_grid_v5/h3_corridor_coverage_container.parquet').h3_id)
    high['off_path']=~high.h3_id.isin(cover);off=high[high.off_path]
    for frame,col,size,alpha in [(grid,'#87AEA9',.10,.62),(off,'#B2614C',.14,.70)]:
        world.scatter(frame.lon,frame.lat,s=size,c=col,lw=0,alpha=alpha,rasterized=True,transform=m.PC,zorder=3)
    bins=[-72,-45,-15,15,45,72];high['band']=pd.cut(high.lat,bins,right=False)
    frame=high.groupby('band',observed=False).off_path.agg(['sum','size']);frame.columns=['off_path','all_high'];frame['pct']=100*frame.off_path/frame.all_high
    ax=chart(fig,[.709,.288,.243,.45]);ax.barh(range(5),frame.pct,color='#B2614C',alpha=.78,height=.55)
    ax.set_yticks(range(5),['72–45°S','45–15°S','15°S–15°N','15–45°N','45–72°N']);ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_xlabel('% of high-emission cells',fontsize=8.1)
    key(fig,['#87AEA9','#B2614C'],['Inventory','High emissions, off path'],(.038,.143))
    for idx,val in enumerate(frame.pct):
        ax.text(val-2,idx,f'{val:.1f}%',va='center',ha='right',fontsize=8,color='white',weight='bold')
    table('S07b_path_coverage',frame.reset_index());assert frame.off_path.sum()==len(off)
    save(fig,'S7','Inventory and high-emission cells outside the archived representative paths. Bars measure the proportion of high-emission cells outside these paths within each latitude band. High means at or above the 75th percentile among all positive cells between 72°S and 72°N. These cell counts are a representation diagnostic, not additional corridor emissions.',1,0)
FUNCTIONS={'9':fig9,'2':fig2,'3':fig3,'4':fig4,'6':fig6,'S1':figS1,'S2':figS2,'S3':figS3,'S4':figS4,'S6':figS6,'S7':figS7}
ORDER=['2','3','4','6','9','S1','S2','S3','S4','S6','S7']
def export():
    report=ROOT/'qa/panels.json'
    saved=json.loads(report.read_text(encoding='utf8')) if report.exists() else {}
    saved.update(META);report.write_text(json.dumps(saved,indent=2,ensure_ascii=False),encoding='utf8')
    if not set(ORDER)<=set(saved):return
    book=fitz.open();toc=[];fonts={};captions=[]
    for idx,id in enumerate(ORDER):
        item=saved[id];src=fitz.open(ROOT/item['file']);page=src[0]
        fonts[id]=[name for xref,ext,kind,name,*rest in page.get_fonts() if 'TimesNewRoman' in name and len(src.extract_font(xref)[3])>0]
        assert fonts[id] and len(fonts[id])==len(page.get_fonts())
        assert len(page.get_text().strip())>10
        book.insert_pdf(src);toc.append([1,'Figure '+id,idx+1])
        captions.append('Figure '+id+'. '+item['caption'])
        page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(ROOT/'qa'/('preview_'+id+'.png'))
    book.set_toc(toc);book.save(ROOT/'Maps_v5_preview.pdf',garbage=4,deflate=True)
    (ROOT/'Captions_v5.txt').write_text('\n\n'.join(captions)+'\n',encoding='utf8')
    baseline=json.loads((ROOT/'qa/r1_hashes_before.json').read_text())
    changed=[key for key,value in baseline.items() if hashlib.sha256((m.HERE.parent/key).read_bytes()).hexdigest()!=value]
    assert not changed,changed
    (ROOT/'qa/verification.json').write_text(json.dumps({'R1_files_verified_unchanged':len(baseline),'embedded_fonts':fonts,'preview_pages':len(book)},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('ids',nargs='*');args=parser.parse_args()
    m.load_data()
    for figure_id in args.ids or FUNCTIONS:FUNCTIONS[figure_id]()
    export()



