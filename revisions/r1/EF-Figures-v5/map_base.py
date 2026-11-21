"""Multi-panel geographic figures from unchanged archived data and R1 results.

No optimization is rerun. Routes use archived, ordered H3 paths, split at the
antimeridian; missing paths are omitted, never replaced with straight lines.
"""
from pathlib import Path
import sys, json, hashlib, argparse
HERE=Path(__file__).resolve().parent.parent/'EF-R1/analysis'
sys.path.insert(0,str(HERE/'_deps'))
import numpy as np
import pandas as pd
import h3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm,LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from functools import lru_cache
import shapefile
from shapely.geometry import shape as as_shape,box as geo_box
from shapely import make_valid,union_all
from cartopy.io.shapereader import Reader

D=Path(__file__).resolve().parent; INPUT=HERE/'inputs'; DATA=INPUT/'data/derived'; OUT=HERE/'results'
MAIN=D/'figures_main'; SI=D/'figures_supporting'; QA=D/'qa'
QA.mkdir(exist_ok=True)
for p in Path('C:/Windows/Fonts').glob('times*.ttf'):font_manager.fontManager.addfont(str(p))
font_manager.findfont(font_manager.FontProperties(family='Times New Roman'),fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9,
 'axes.titlesize':9.5,'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
 'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
 'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
 'mathtext.fallback':None,'savefig.facecolor':'white','text.color':'#25343C',
 'axes.labelcolor':'#25343C','xtick.color':'#52616B','ytick.color':'#52616B'})
INK='#203847'; MUTED='#536978'; LAND='#F0ECE6'; OCEAN='#F3F8FA'; COAST='#A1B0B6'
EU='#424F91'; BRI='#107B75'; SHARED='#8D497B'; NEAR='#987346'; OTHER='#B66349'; RED='#AC4B50'
CAMP={'EU':EU,'BRI':BRI,'Other':OTHER}
PC=ccrs.PlateCarree(); WORLD=ccrs.Robinson(central_longitude=0)
REGIONS={'Europe':(-16,39,25,69),'Maritime Asia':(86,156,-12,44),
         'Malacca Strait':(92,117,-6,14),'East Asia':(111,143,18,43.6)}
LAND110=cfeature.ShapelyFeature(tuple(Reader(INPUT/'basemap/ne_110m_land.shp').geometries()),PC)
LAND50=cfeature.ShapelyFeature(tuple(Reader(INPUT/'basemap/ne_50m_land.shp').geometries()),PC)
WORLD_LAND=cfeature.ShapelyFeature(tuple(geom.simplify(.085,preserve_topology=True) for geom in LAND50.geometries()),PC)
BORDERS=cfeature.ShapelyFeature(tuple(Reader(INPUT/'basemap/ne_110m_admin_0_boundary_lines_land.shp').geometries()),PC)
MANIFEST={}; CHECKS={}; _PROJECTED={}

def load_data():
    global GOV, PORTS, CAND, GRID, PATHS, PLANS, STATS, HUB
    GOV=pd.read_parquet(DATA/'policy_analysis_v5/governance_corridors_container.parquet')
    PORTS=pd.read_parquet(DATA/'policy_analysis_v5/governance_ports_container.parquet').drop_duplicates('port_code')
    CAND=pd.read_parquet(DATA/'processed_v5/candidate_ports_container.parquet').drop_duplicates('port_code')
    GRID=pd.read_parquet(DATA/'h3_grid_v5/h3_emission_grid_container.parquet')
    paths=pd.read_parquet(DATA/'h3_grid_v5/corridor_h3_paths_container.parquet').sort_values(['corridor_id','seq_order'])
    unique=paths.h3_id.unique();ll={c:h3.cell_to_latlng(c) for c in unique}
    paths['lon']=[ll[c][1] for c in paths.h3_id];paths['lat']=[ll[c][0] for c in paths.h3_id]
    PATHS={}
    for cid,g in paths.groupby('corridor_id',sort=False):
        pts=g[['lon','lat']].to_numpy()
        splits=np.flatnonzero(np.abs(np.diff(pts[:,0]))>180)+1
        PATHS[cid]=[a for a in np.split(pts,splits) if len(a)>1]
    PLANS={name:set(pd.read_csv(DATA/folder/file).port_code) for name,folder,file in [
      ('Global','milp_results_v5','selected_ports_container.csv'),
      ('EU','policy_analysis_v5','meU_selected_ports_container.csv'),
      ('BRI','policy_analysis_v5','meB_selected_ports_container.csv')]}
    STATS=pd.read_csv(OUT/'archived_plan_closure.csv').set_index('plan')
    HUB=pd.read_csv(OUT/'R1_endpoint_reliance.csv').merge(PORTS[['port_code','port_lon','port_lat']],on='port_code',validate='one_to_one')
    CHECKS.update({'corridors':len(GOV),'h3_cells':len(GRID),
       'h3_total_Mt':float(GRID.emission_tco2e.sum()/1e6),'archived_path_corridors':len(PATHS),
       'selected_ports':{k:len(v) for k,v in PLANS.items()},
       'top20_share_pct':float(HUB.nlargest(20,'HCS_pct').HCS_pct.sum())})
    print('Loaded original paths, emissions and port plans.',flush=True)

@lru_cache(maxsize=4)
def shelf_edges(region):
    west,east,south,north=REGIONS[region]
    bounds=geo_box(west-.1,south-.1,east+.1,north+.1)
    source=D.parent/'EF-Figures-v3-S1/basemap/ne_10m_bathymetry_K_200.zip'
    pieces=[]
    padded=bounds.buffer(1.,cap_style='square')
    with shapefile.Reader(str(source)) as dataset:
        for record in dataset.iterShapes(bbox=padded.bounds):
            part=make_valid(as_shape(record.__geo_interface__)).intersection(padded)
            if not part.is_empty:pieces.append(part)
    return [union_all(pieces,grid_size=.00001).boundary.intersection(bounds).simplify(.025,preserve_topology=True)]

def map_ax(fig,box,region=None):
    ax=fig.add_axes(box,projection=WORLD if region is None else PC)
    if region is None:
        ax.set_global();ax.set_ylim(WORLD.transform_point(0,-58,PC)[1],WORLD.transform_point(0,80,PC)[1])
    else:ax.set_extent(REGIONS.get(region,region),crs=PC)
    ax.set_facecolor(OCEAN)
    if region in REGIONS:
        ax.add_geometries(shelf_edges(region),PC,facecolor='none',edgecolor='#A4BFBE',linewidth=.30,alpha=.78,zorder=.75)
    ax.add_feature(WORLD_LAND if region is None else LAND50,facecolor=LAND,edgecolor=COAST,linewidth=.30,zorder=1)
    ax.add_feature(BORDERS,facecolor='none',edgecolor='#C4CCCD',linewidth=.22,zorder=2)
    ax.gridlines(xlocs=np.arange(-180,181,60) if region is None else np.arange(-180,181,10),
                 ylocs=np.arange(-60,81,30) if region is None else np.arange(-60,81,10),
                 linewidth=.25,color='#C5CED2',alpha=.65,linestyle=(0,(2,3)),zorder=0)
    ax.spines['geo'].set_edgecolor('#C5CED2');ax.spines['geo'].set_linewidth(.55)
    return ax

def heading(ax,letter,title,subtitle=None):
    ax.text(0,1.055,f'({letter})',transform=ax.transAxes,weight='bold',fontsize=9.5,va='bottom')
    ax.annotate(title,xy=(0,1.055),xycoords='axes fraction',xytext=(20,0),
                textcoords='offset points',fontsize=9.5,va='bottom')
    if subtitle:ax.text(0,-.065,subtitle,transform=ax.transAxes,fontsize=8.2,color=MUTED,va='top')

def legend(fig,colors,labels,y=.035,fontsize=8.5,markers=None):
    markers=markers or ['o']*len(colors)
    handles=[Line2D([0],[0],marker=m,ls='',ms=4.8,color=c,
        markeredgecolor=c if m=='x' else 'white',markeredgewidth=.65 if m=='x' else .35)
        for c,m in zip(colors,markers)]
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,y),ncol=len(labels),frameon=False,
               fontsize=fontsize,handletextpad=.35,columnspacing=1.5)

def route_ids(table,n=60,weight='E_e'):
    return table[table.corridor_id.isin(PATHS)].nlargest(n,weight).corridor_id.tolist()

def routes(ax,ids,color,width=.5,alpha=.6):
    proj=ax.projection;key='world' if isinstance(proj,ccrs.Robinson) else 'local'
    cache=_PROJECTED.setdefault(key,{})
    segments=[]
    for cid in ids:
        if cid not in cache:
            cache[cid]=[proj.transform_points(PC,a[:,0],a[:,1])[:,:2] for a in PATHS.get(cid,[])]
        segments.extend(cache[cid])
    if segments:ax.add_collection(LineCollection(segments,colors=color,linewidths=width,alpha=alpha,zorder=4))

def points(ax,frame,color,size=10,alpha=.9,marker='o',zorder=7,edge='white'):
    kwargs={'linewidths':.65} if marker=='x' else {'edgecolors':edge,'linewidths':.25}
    if len(frame):ax.scatter(frame.port_lon,frame.port_lat,s=size,c=color,alpha=alpha,marker=marker,
        transform=PC,zorder=zorder,**kwargs)

def selected(ax,name,small=False):
    feasible=CAND[CAND.is_feasible]
    points(ax,feasible,'#C1C8CB',size=1.8 if small else 2.3,alpha=.60,edge='none',zorder=3)
    sel=PORTS[PORTS.port_code.isin(PLANS[name])]
    for camp,col in CAMP.items():points(ax,sel[sel.camp==camp],col,6.7 if small else 12)

def region_box(ax,name,letter):
    x0,x1,y0,y1=REGIONS[name]
    ax.plot([x0,x1,x1,x0,x0],[y0,y0,y1,y1,y0],transform=PC,color=INK,lw=.6,ls=(0,(3,2)),zorder=9)
    ax.text(x0,y1+2,letter,transform=PC,fontsize=8,weight='bold',color=INK,
            bbox={'facecolor':'white','edgecolor':'none','pad':.6},zorder=10)

def save(fig,stem,kind,panels,notes):
    folder=MAIN if kind=='main' else SI
    path=folder/stem
    fig.savefig(path.with_suffix('.pdf'),dpi=450,bbox_inches='tight',pad_inches=.055)
    fig.savefig(path.with_suffix('.png'),dpi=350,bbox_inches='tight',pad_inches=.055)
    MANIFEST[f'{folder.name}/{stem}.pdf']={'map_panels':panels,'description':notes,
        'font':'Times New Roman','sha256':hashlib.sha256(path.with_suffix('.pdf').read_bytes()).hexdigest()}
    plt.close(fig);print('Saved',stem,flush=True)

def heat_raster():
    lon=np.arange(-180,180.251,.25);lat=np.arange(-90,90.251,.25)
    a,_,_=np.histogram2d(GRID.lon,GRID.lat,bins=[lon,lat],weights=GRID.emission_tco2e)
    CHECKS['display_grid_conserves_emissions']=bool(np.isclose(a.sum(),GRID.emission_tco2e.sum(),rtol=1e-12))
    return lon,lat,np.ma.masked_less_equal(a.T,0)

HEAT=LinearSegmentedColormap.from_list('shipping_emissions',['#E4ECED','#B6CED0','#62A7AB','#D8B46C','#B56F3D','#743D39'])
NORM=LogNorm(10,100000)

def heat(ax,raster):
    lon,lat,a=raster
    if not isinstance(ax.projection,ccrs.Robinson):
        x0,x1,y0,y1=ax.get_extent(PC)
        ix=np.where((lon[:-1]>=x0-.25)&(lon[:-1]<=x1))[0];iy=np.where((lat[:-1]>=y0-.25)&(lat[:-1]<=y1))[0]
        lon=lon[ix[0]:ix[-1]+2];lat=lat[iy[0]:iy[-1]+2];a=a[iy[0]:iy[-1]+1,ix[0]:ix[-1]+1]
    return ax.pcolormesh(lon,lat,a,cmap=HEAT,norm=NORM,transform=PC,rasterized=True,zorder=3,shading='flat')

def fig2():
    r=heat_raster();fig=plt.figure(figsize=(7.4,5.25))
    a=map_ax(fig,[.035,.405,.93,.55]);m=heat(a,r);heading(a,'a','Container-shipping emissions, 2024')
    names=['Europe','Malacca Strait','East Asia']
    for i,name in enumerate(names):
        letter=chr(98+i);region_box(a,name,letter)
        ax=map_ax(fig,[.035+i*.323,.105,.285,.255],name);heat(ax,r);heading(ax,letter,name)
    cax=fig.add_axes([.24,.025,.52,.018]);cb=fig.colorbar(m,cax=cax,orientation='horizontal',extend='both')
    cb.set_ticks([10,100,1000,10000,100000],labels=['10','100','1,000','10,000','100,000']);cb.ax.tick_params(labelsize=8,length=2)
    cb.set_label('Annual emissions per 0.25° display cell (t CO₂e)',fontsize=8.7,labelpad=3)
    save(fig,'Figure_02','main',4,'Global emissions with Europe, Malacca and East Asia enlargements; one shared logarithmic scale.')

def fig3():
    groups=[('EU_ETS','EU connected',EU),('BRI_only','BRI linked only',BRI),
            ('FuelEU_exposed','Near-EU proximity',NEAR),('Vacuum','Outside selected footprints',OTHER)]
    fig=plt.figure(figsize=(7.4,4.65));shares={}
    for i,(key,title,col) in enumerate(groups):
        ax=map_ax(fig,[.035+(i%2)*.495,.555-(i//2)*.475,.435,.335])
        sub=GOV[GOV.gov_type==key];ids=route_ids(sub,60);routes(ax,ids,col,width=.55,alpha=.73)
        ep=set(sub.origin_port)|set(sub.dest_port);points(ax,PORTS[PORTS.port_code.isin(ep)],col,1.9,alpha=.4,zorder=3,edge='none')
        em=100*sub.E_e.sum()/GOV.E_e.sum();count=100*len(sub)/len(GOV)
        heading(ax,chr(97+i),title)
        ax.text(0,-.09,f'{em:.1f}% of emissions',transform=ax.transAxes,color=col,fontsize=10.5,weight='bold')
        ax.text(0,-.20,f'{len(sub):,} corridors  |  {count:.1f}% of routes',transform=ax.transAxes,fontsize=8.2,color=MUTED)
        shares[key]={'count':len(sub),'emission_share_pct':em,'corridor_share_pct':count,'mapped_ids':ids}
    CHECKS['regional_classification']=shares
    save(fig,'Figure_03','main',4,'Four mutually exclusive geographic classes, with full-inventory emission and route shares.')

def overlap_groups():
    eu=GOV.eu_ets_endpoint.astype(bool);bri=GOV.is_bri.astype(bool)
    return [('EU only',GOV[eu&~bri],EU),('Both footprints',GOV[eu&bri],SHARED),
            ('BRI only',GOV[bri&~eu],BRI),('Outside both',GOV[~eu&~bri],OTHER)]

def fig4():
    fig=plt.figure(figsize=(7.4,5.0));groups=overlap_groups()
    axes=[map_ax(fig,[.035,.49,.65,.43]),map_ax(fig,[.04,.11,.30,.29],'Europe'),map_ax(fig,[.37,.11,.30,.29],'Maritime Asia')]
    for i,ax in enumerate(axes):
        for j in [3,0,2,1]:
            name,sub,col=groups[j];routes(ax,route_ids(sub,75),col,.60 if j==1 else .40,.85 if j==1 else .42)
        heading(ax,chr(97+i),['Overlapping investment footprints','Europe','Maritime Asia'][i])
    region_box(axes[0],'Europe','b');region_box(axes[0],'Maritime Asia','c')
    ax=fig.add_axes([.735,.17,.245,.69]);ax.axis('off')
    ax.text(0,1,'(d)  Global inventory',fontsize=9.5,weight='bold',va='bottom')
    summaries=[]
    for i,(name,sub,col) in enumerate(groups):
        y=.84-i*.23;share=sub.E_e.sum()/GOV.E_e.sum()*100
        ax.text(0,y,name,fontsize=9.2,color=col,weight='bold')
        ax.text(0,y-.063,f'{len(sub):,} corridors  ·  {share:.1f}%',fontsize=8.5)
        ax.add_patch(Rectangle((0,y-.105),.92,.015,color='#E7ECEE',lw=0))
        ax.add_patch(Rectangle((0,y-.105),.92*share/100,.015,color=col,lw=0))
        summaries.append({'group':name,'count':len(sub),'emission_share_pct':share})
    ax.text(0,-.055,'Percentages: emission shares',fontsize=7.7,color=MUTED)
    CHECKS['overlap_partition']=summaries
    legend(fig,[EU,SHARED,BRI,OTHER],[g[0] for g in groups],y=.015)
    save(fig,'Figure_04','main',3,'Global overlap, two regional enlargements, and a compact inventory summary.')

def fig6():
    fig=plt.figure(figsize=(7.4,6.9));model_names=['Global plan','EU priority plan','Closed BRI network']
    selections={}
    for i,name in enumerate(['Global','EU','BRI']):
        bottom=.71-i*.305
        ax=map_ax(fig,[.035,bottom,.60,.245]);z=map_ax(fig,[.69,bottom,.27,.245],'Maritime Asia')
        closure=pd.read_csv(OUT/f'closure_{name}.csv');positive=closure[closure.W>0]
        ids=route_ids(positive,60,'W')
        for target in [ax,z]:routes(target,ids,'#659798',.42,.38);selected(target,name,small=target==ax)
        row=STATS.loc[name]
        heading(ax,chr(97+2*i),model_names[i]);heading(z,chr(98+2*i),'Maritime Asia')
        ax.text(0,-.08,f'{len(PLANS[name])} ports   |   {row.global_Mt:.2f} Mt worldwide   |   {len(positive):,} routes',
                transform=ax.transAxes,fontsize=8.3,color=MUTED)
        region_box(ax,'Maritime Asia',chr(98+2*i))
        selections[name]={'port_count':len(PLANS[name]),'worldwide_Mt':float(row.global_Mt),
          'positive_supported_routes':len(positive),'mapped_ids':ids}
        assert set(ids)<=set(positive.corridor_id)
    legend(fig,[EU,BRI,OTHER],['EU ports','BRI linked ports','Other ports'],y=.002)
    CHECKS['model_comparison']=selections
    save(fig,'Figure_06','main',6,'Three plans in aligned rows; global and Maritime Asia views share point encoding.')

PORT_NAMES={'SGSGP':'Singapore','KRBUS':'Busan','MAZAN':'Tanger Med','MATAN':'Tangier','MYTPP':'Tanjung Pelepas',
 'CNNBO':'Ningbo','NLROT':'Rotterdam','MYPKE':'Port Klang','CNYTN':'Yantian','LKCOL':'Colombo','CNQIN':'Qingdao',
 'ESALG':'Algeciras','HKHKG':'Hong Kong','CNWGG':'Waigaoqiao','CNWGQ':'Waigaoqiao','CNYSN':'Yangshan',
 'TWKAO':'Kaohsiung','USNWK':'Newark','USNYK':'New York','MXMAN':'Manzanillo','ESVAL':'Valencia','BRSTS':'Santos','CNNSA':'Nansha'}

def hub_map(ax):
    for camp,col in CAMP.items():
        sub=HUB[HUB.camp==camp].sort_values('HCS_pct',ascending=False)
        sizes=np.maximum(sub.HCS_pct.to_numpy()*24,1.1)
        points(ax,sub,col,sizes,alpha=.80)
        sub=sub[sub.port_code.isin(PLANS['Global'])]
        ax.scatter(sub.port_lon,sub.port_lat,s=np.maximum(sub.HCS_pct*24,1.1),facecolors='none',edgecolors=INK,
                    linewidths=.4,transform=PC,zorder=8)

def fig9():
    fig=plt.figure(figsize=(7.4,5.8))
    a=map_ax(fig,[.035,.48,.565,.43]);b=map_ax(fig,[.035,.10,.255,.31],'Europe');c=map_ax(fig,[.335,.10,.255,.31],'Maritime Asia')
    for ax in [a,b,c]:hub_map(ax)
    heading(a,'a','Reliance on shared ports');heading(b,'b','Europe');heading(c,'c','Maritime Asia')
    region_box(a,'Europe','b');region_box(a,'Maritime Asia','c')
    top=HUB.nlargest(20,'HCS_pct').iloc[::-1]
    ax=fig.add_axes([.785,.11,.20,.80]);y=np.arange(20)
    ax.hlines(y,0,top.HCS_pct,color=top.camp.map(CAMP),linewidth=1.4,alpha=.8)
    ax.scatter(top.HCS_pct,y,c=top.camp.map(CAMP),s=15,zorder=3)
    ax.set_yticks(y,[PORT_NAMES.get(x,x) for x in top.port_code],fontsize=7.4)
    ax.tick_params(axis='y',length=0,pad=4);ax.set_xlim(0,13.3);ax.set_xticks([0,5,10]);ax.set_xlabel('Reliance share (%)',fontsize=8.3)
    ax.set_ylim(-.8,20.5);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#C5CED2')
    ax.grid(axis='x',color='#E6EBED',lw=.5);ax.set_axisbelow(True)
    ax.text(-.52,1.04,'(d)  Twenty largest shares',transform=ax.transAxes,fontsize=9.5,weight='bold')
    for i,v in enumerate(top.HCS_pct):ax.text(v+.22,i,f'{v:.2f}',fontsize=7,va='center')
    fig.text(.04,.98,f'Top 20 ports account for {top.HCS_pct.sum():.1f}% of endpoint reliance',fontsize=10.5,color=INK)
    legend(fig,[EU,BRI,OTHER],['EU','BRI linked','Other'],y=.005,fontsize=8.2)
    handles=[plt.scatter([],[],s=24*s,facecolors='none',edgecolors=MUTED,lw=.5) for s in [1,5,10]]
    a.legend(handles,['1%','5%','10%'],loc='lower left',bbox_to_anchor=(-.01,-.13),ncol=3,frameon=False,
             fontsize=7.5,handletextpad=.7,columnspacing=1.2,labelspacing=1,borderpad=0)
    save(fig,'Figure_09','main',3,'Global proportional-symbol map, two port enlargements and a compact top-20 ranking.')

def supp1():
    grid=pd.read_parquet(DATA/'h3_grid_v5/h3_eu_ets_coverage_container.parquet')
    fig=plt.figure(figsize=(7.4,3.3))
    axes=[map_ax(fig,[.035,.20,.63,.68]),map_ax(fig,[.715,.20,.25,.68],'Europe')]
    for i,ax in enumerate(axes):
        for weight,col in [(.5,'#D9E4EC'),(1.,'#9EB9CC')]:
            sub=grid[np.isclose(grid.weight,weight)]
            ax.scatter(sub.lon,sub.lat,s=.35 if i==0 else 1.2,c=col,marker='s',lw=0,transform=PC,zorder=3,rasterized=True)
        ax.add_feature(LAND110 if i==0 else LAND50,facecolor='none',edgecolor='#647F91',linewidth=.30,zorder=6)
        ax.add_feature(BORDERS,facecolor='none',edgecolor='#7D94A3',linewidth=.22,zorder=7)
        heading(ax,chr(97+i),['Archived endpoint proxy','Europe'][i])
    region_box(axes[0],'Europe','b')
    legend(fig,['#D9E4EC','#9EB9CC'],['Half-weight cells','Full-weight cells'],y=.01,markers=['s','s'])
    save(fig,'Figure_S01','supp',2,'Archived EU proxy grid, shown at global and regional scales.')

def supp2():
    fig=plt.figure(figsize=(7.4,4.8))
    a=map_ax(fig,[.035,.56,.43,.33]);b=map_ax(fig,[.53,.56,.43,.33])
    c=map_ax(fig,[.065,.10,.34,.31],'Europe');d=map_ax(fig,[.58,.10,.34,.31],'Maritime Asia')
    cmap=LinearSegmentedColormap.from_list('proximity',['#CED9DE','#9BA9BC','#5B567C'])
    sample=GOV[GOV.corridor_id.isin(PATHS)].nlargest(150,'E_e')
    for row in sample.itertuples():routes(a,[row.corridor_id],cmap(float(row.fueleu_max)),.48,.8)
    heading(a,'a','Near-EU corridor proximity')
    changes=pd.read_csv(DATA/'policy_analysis_v5/fueleu_port_changes_container.csv')
    for i,ax in enumerate([b,c,d]):
        for state,color,marker in [('removed',RED,'x'),('added',SHARED,'o')]:
            sub=changes[changes.change==state]
            points(ax,sub,color,9 if i==0 else 15,marker=marker,edge='none')
        heading(ax,chr(98+i),['Port priority scenario','Europe','Maritime Asia'][i])
    region_box(b,'Europe','c');region_box(b,'Maritime Asia','d')
    sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap)
    cax=fig.add_axes([.11,.535,.28,.014]);cb=fig.colorbar(sm,cax=cax,orientation='horizontal')
    cb.set_label('Maximum proximity along route',fontsize=8,labelpad=1);cb.ax.tick_params(labelsize=7,length=2)
    legend(fig,[SHARED,RED],['Ports added','Ports removed'],y=.005,markers=['o','x'])
    save(fig,'Figure_S02','supp',4,'Archived corridor proximity and archived changes in the EU port-priority experiment.')

def supp3():
    fig=plt.figure(figsize=(7.4,4.4));bri=GOV[GOV.is_bri.astype(bool)];other=GOV[~GOV.is_bri.astype(bool)]
    a=map_ax(fig,[.035,.55,.43,.34]);b=map_ax(fig,[.53,.55,.43,.34])
    c=map_ax(fig,[.065,.09,.34,.32],'Europe');d=map_ax(fig,[.58,.09,.34,.32],'Maritime Asia')
    for ax,sub,col in [(a,bri,BRI),(b,other,OTHER),(c,bri,BRI),(d,bri,BRI)]:
        routes(ax,route_ids(sub,70),col,.55,.72)
        ep=set(sub.origin_port)|set(sub.dest_port);points(ax,PORTS[PORTS.port_code.isin(ep)],col,1.8,alpha=.35,edge='none')
    for ax,l,t in [(a,'a','BRI linked footprint'),(b,'b','Outside the BRI footprint'),(c,'c','Europe: BRI linked routes'),(d,'d','Maritime Asia: BRI linked routes')]:heading(ax,l,t)
    region_box(a,'Europe','c');region_box(a,'Maritime Asia','d')
    save(fig,'Figure_S03','supp',4,'Nonexclusive BRI route footprint and complementary regional details.')

def supp4():
    fig=plt.figure(figsize=(7.4,4.4))
    axes=[map_ax(fig,[.035,.45,.93,.46]),map_ax(fig,[.08,.09,.35,.30],'Europe'),map_ax(fig,[.57,.09,.35,.30],'Maritime Asia')]
    ids=route_ids(pd.read_csv(OUT/'closure_Global.csv'),80,'W')
    for i,ax in enumerate(axes):
        routes(ax,ids,'#659798',.45,.40);selected(ax,'Global',small=i==0)
        heading(ax,chr(97+i),['Reference global investment plan','Europe','Maritime Asia'][i])
    region_box(axes[0],'Europe','b');region_box(axes[0],'Maritime Asia','c')
    legend(fig,[EU,BRI,OTHER],['EU ports','BRI linked ports','Other ports'],y=-.015)
    save(fig,'Figure_S04','supp',3,'Global plan atlas with detailed selected-port maps.')

def supp6():
    fig=plt.figure(figsize=(7.4,2.6))
    for i,name in enumerate(['Global','EU','BRI']):
        ax=map_ax(fig,[.035+i*.325,.17,.285,.65],'Europe')
        selected(ax,name);heading(ax,chr(97+i),['Global plan','EU priority','Closed BRI'][i])
    legend(fig,[EU,BRI,OTHER],['EU ports','BRI linked ports','Other ports'],y=.005)
    save(fig,'Figure_S06','supp',3,'Matched Europe enlargements complement the global/Asia main-text comparison.')

def supp7():
    covered=set(pd.read_parquet(DATA/'h3_grid_v5/h3_corridor_coverage_container.parquet').h3_id)
    grid=GRID[(GRID.emission_tco2e>0)&(GRID.lat.abs()<72)].copy()
    threshold=grid.emission_tco2e.quantile(.75)
    highlight=grid[(grid.emission_tco2e>=threshold)&~grid.h3_id.isin(covered)]
    fig=plt.figure(figsize=(7.4,4.4))
    axes=[map_ax(fig,[.035,.45,.93,.46]),map_ax(fig,[.08,.09,.35,.30],'Europe'),map_ax(fig,[.57,.09,.35,.30],'Maritime Asia')]
    for i,ax in enumerate(axes):
        ax.scatter(grid.lon,grid.lat,s=.10 if i==0 else .35,color='#A7C3C6',lw=0,transform=PC,zorder=3,rasterized=True)
        ax.scatter(highlight.lon,highlight.lat,s=.15 if i==0 else .65,color=RED,lw=0,transform=PC,zorder=4,rasterized=True)
        heading(ax,chr(97+i),['Inventory and representative-path coverage','Europe','Maritime Asia'][i])
    region_box(axes[0],'Europe','b');region_box(axes[0],'Maritime Asia','c')
    legend(fig,['#A7C3C6',RED],['Inventory cells','High-emission cells outside representative paths'],y=-.015,fontsize=8)
    CHECKS['coverage_diagnostic']={'threshold_t':float(threshold),'highlighted_cells':len(highlight)}
    save(fig,'Figure_S07','supp',3,'Original path-coverage diagnostic with paired regional enlargements.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fig',default='all');args=parser.parse_args()
    load_data()
    for n,func in [('2',fig2),('3',fig3),('4',fig4),('6',fig6),('9',fig9),
                  ('S1',supp1),('S2',supp2),('S3',supp3),('S4',supp4),('S6',supp6),('S7',supp7)]:
        if args.fig in ['all',n]:func()
    report=QA/'map_manifest.json'
    if args.fig!='all' and report.exists():
        previous=json.loads(report.read_text(encoding='utf8'));previous['figures'].update(MANIFEST);previous['checks'].update(CHECKS)
    else:previous={'figures':MANIFEST,'checks':CHECKS}
    report.write_text(json.dumps(previous,indent=2),encoding='utf8')

if __name__=='__main__':main()
