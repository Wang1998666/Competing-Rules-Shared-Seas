from pathlib import Path
import sys, json, hashlib, zipfile
ROOT=Path(__file__).resolve().parent
AN=ROOT.parent/'EF-R1/analysis'
sys.path.insert(0,str(AN/'_deps'))
import numpy as np
import pandas as pd
import shapefile
from shapely.geometry import box,shape
from shapely import make_valid, orient_polygons
from shapely.ops import transform
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,patheffects
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import cartopy.crs as crs
from cartopy.io.shapereader import Reader
import pymupdf as fitz

for file in Path('C:/Windows/Fonts').glob('times*.ttf'):font_manager.fontManager.addfont(str(file))
font_manager.findfont('Times New Roman',fallback_to_default=False)
plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'pdf.fonttype':42,'axes.linewidth':.45})
BASE=AN/'inputs/basemap'
DATA=AN/'inputs/data/derived'
PC=crs.PlateCarree()
INK='#153B45'; MUTED='#4B626A'; LAND='#F2EFE7'; COAST='#809697'; GOLD='#C07C40'
FULL='#397A77'; HALF='#BB7741'
DEPTH=['#EDF6F2','#DAECE9','#C9E1E0','#B5D3D6']
FRAME=box(-62,10,78,83)
REG=crs.LambertConformal(central_longitude=10,central_latitude=50,standard_parallels=(38,62))
TO_REG=Transformer.from_crs('EPSG:4326',REG,always_xy=True)
def title(fig,x,y,letter,label):
    fig.text(x,y,letter,color=INK,weight='bold',fontsize=11,va='center')
    return fig.text(x+.024,y,label,color=INK,weight='bold',fontsize=10.9,va='center')
def geometry(ax,geoms,**style):
    if ax.projection==REG:
        ax.add_geometries([orient_polygons(transform(TO_REG.transform,geom)) for geom in geoms],REG,**style)
    else:
        ax.add_geometries(geoms,PC,**style)
def regional_geoms(geometries):
    parts=[]
    for geom in geometries:
        if not geom.intersects(FRAME):continue
        clipped=make_valid(geom).intersection(FRAME).simplify(.018,preserve_topology=True)
        if not clipped.is_empty:parts.append(clipped)
    return parts
def bathy(ax,name,tone,order):
    filename=ROOT.parent/'EF-Figures-v3-S1/basemap'/f'ne_10m_bathymetry_{name}.zip'
    with shapefile.Reader(str(filename)) as source:
        geoms=regional_geoms(shape(item.__geo_interface__) for item in source.iterShapes(bbox=FRAME.bounds))
    geometry(ax,geoms,facecolor=tone,edgecolor='none',zorder=order)
def sea(ax,x,y,name,angle=0):
    art=ax.text(x,y,name,transform=PC,ha='center',va='center',fontsize=9.1,fontstyle='italic',
                color='#356973',rotation=angle,zorder=8,clip_on=True)
    art.set_path_effects([patheffects.withStroke(linewidth=1.7,foreground=to_rgba('#EFF7F5',.75))])
def pill(ax,xy,text,offset):
    lbl=ax.annotate(text,xy,xytext=offset,textcoords='offset points',fontsize=8.1,color=INK,
        ha='left' if offset[0]>=0 else 'right',va='center',zorder=11,
        arrowprops={'arrowstyle':'-','lw':.55,'color':'#5E7575','shrinkB':3,'shrinkA':1.5})
    lbl.set_path_effects([patheffects.withStroke(linewidth=2.1,foreground='#FBFBF8')])
def draw():
    gov=pd.read_parquet(DATA/'policy_analysis_v5/governance_corridors_container.parquet')
    info=pd.read_parquet(DATA/'policy_analysis_v5/governance_ports_container.parquet')
    grid=pd.read_parquet(DATA/'h3_grid_v5/h3_eu_ets_coverage_container.parquet')
    assert grid.h3_id.is_unique and set(grid.weight)=={.5,1}
    assert (gov.eu_ets_endpoint==(gov.origin_eu|gov.dest_eu)).all() and gov.corridor_id.is_unique
    ends=pd.concat([gov.loc[gov.origin_eu,['corridor_id','origin_port']].rename(columns={'origin_port':'port_code'}),
                    gov.loc[gov.dest_eu,['corridor_id','dest_port']].rename(columns={'dest_port':'port_code'})]).drop_duplicates()
    ports=ends.groupby('port_code',as_index=False).size().rename(columns={'size':'corridors'}).merge(
        info[['port_code','port_lon','port_lat']].drop_duplicates(),on='port_code',validate='one_to_one',how='left')
    assert ports.notna().all().all()
    counts=grid.groupby('weight').size()
    assert counts[1]==103805 and counts[.5]==7386
    fig=plt.figure(figsize=(9.25,5.88),facecolor='white')
    title(fig,.025,.951,'a','European ports in the endpoint proxy')
    panel_title=title(fig,.628,.951,'b','Proxy cells on shipping paths')
    ax=fig.add_axes([.025,.172,.574,.735],projection=REG)
    ax.set_xlim(-1870000,1980000);ax.set_ylim(-1760000,1600000)
    ax.set_aspect('equal')
    ax.set_facecolor(DEPTH[0]);ax.spines['geo'].set_color('#96ABAB')
    for order,(name,tone) in enumerate(zip(['K_200','J_1000','H_3000'],DEPTH[1:]),1):bathy(ax,name,tone,order)
    geometry(ax,regional_geoms(Reader(BASE/'ne_50m_land.shp').geometries()),facecolor=LAND,edgecolor=COAST,lw=.36,zorder=4)
    geometry(ax,regional_geoms(Reader(BASE/'ne_110m_admin_0_boundary_lines_land.shp').geometries()),facecolor='none',edgecolor='#BAC0B9',lw=.30,zorder=5)
    ax.gridlines(xlocs=np.arange(-50,81,10),ylocs=np.arange(20,81,10),lw=.23,color='#98AFB0',alpha=.65,linestyle=(0,(2,4)),zorder=5)
    for lon,lat,name,angle in [(-11.5,48.7,'North Atlantic',12),(2.1,57.5,'North Sea',0),(19.2,59.25,'Baltic Sea',22),(-5.6,45.25,'Bay of Biscay',0),(7.2,38.45,'Mediterranean Sea',-5),(32.2,43.7,'Black Sea',0)]:
        sea(ax,lon,lat,name,angle)
    areas=lambda values:3+.61*np.asarray(values)
    ports=ports.sort_values('corridors',ascending=False)
    ax.scatter(ports.port_lon,ports.port_lat,s=areas(ports.corridors),c='#226861',edgecolors='white',alpha=.82,lw=.37,transform=PC,zorder=6)
    names={'NLROT':('Rotterdam',(-9,10)),'DEHAM':('Hamburg',(10,7)),'ESVAL':('Valencia',(7,7)),
           'ESALG':('Algeciras',(5,-11)),'GRPIR':('Piraeus',(7,-3)),'ROCON':('Constanța',(-10,8)),
           'FRLEH':('Le Havre',(-9,-5)),'PLGDA':('Gdańsk',(6,9))}
    for code,(name,offset) in names.items():
        row=ports.set_index('port_code').loc[code]
        point=REG.transform_point(row.port_lon,row.port_lat,PC)
        assert ax.get_xlim()[0]<point[0]<ax.get_xlim()[1] and ax.get_ylim()[0]<point[1]<ax.get_ylim()[1]
        pill(ax,point,name,offset)
    fig.text(.026,.124,'Corridors per port',fontsize=8.8,color=INK)
    marks=[Line2D([],[],ls='',marker='o',mfc='#226861',mec='white',mew=.35,ms=np.sqrt(areas(val))) for val in [5,30,100]]
    fig.legend(marks,['5','30','100'],loc='center left',bbox_to_anchor=(.149,.132),ncol=3,columnspacing=.65,handletextpad=.1,frameon=False,fontsize=8)
    fig.text(.026,.072,'Seafloor depth (m)',fontsize=8.8,color=INK)
    for idx,(tone,lbl) in enumerate(zip(DEPTH,['0–200','200–1,000','1,000–3,000','>3,000'])):
        left=.166+idx*.102
        fig.add_artist(plt.Rectangle((left,.069),.016,.019,transform=fig.transFigure,facecolor=tone,edgecolor='#98AEAF',lw=.35))
        fig.text(left+.022,.072,lbl,fontsize=7.8,color=MUTED)
    right=fig.add_axes([.625,.54,.346,.346],projection=crs.Robinson(central_longitude=0));right.set_global()
    right.set_facecolor('#F5F8F6');right.spines['geo'].set_edgecolor('#9AABAD')
    from marine_grid import build
    paths=pd.read_parquet(DATA/'h3_grid_v5/corridor_h3_paths_container.parquet',columns=['h3_id'])
    marine,extent=build(grid,paths)
    map_image=right.imshow(marine,extent=extent,origin='upper',interpolation='lanczos',zorder=2)
    title(fig,.628,.462,'c','Grid cells by assigned weight')
    fig.text(.628,.407,f'{len(grid):,} cells in the full archive',fontsize=9,color=MUTED)
    left=.63; width=.335; full=counts[1]/len(grid)
    for xpos,part,tone in [(left,full,FULL),(left+width*full,1-full,HALF)]:
        fig.add_artist(plt.Rectangle((xpos,.326),width*part,.036,transform=fig.transFigure,facecolor=tone,edgecolor='white',lw=.45))
    for idx,(weight,label,col) in enumerate([(1.,'Full weight',FULL),(.5,'Half weight',HALF)]):
        ypos=.254-.067*idx
        fig.add_artist(plt.Rectangle((left,ypos),.012,.019,transform=fig.transFigure,facecolor=col,edgecolor='none'))
        fig.text(left+.021,ypos,label,fontsize=9.3,color=INK)
        fig.text(.862,ypos,f'{counts[weight]:,}',fontsize=9.4,color=INK,ha='right')
        fig.text(.965,ypos,f'{100*counts[weight]/len(grid):.1f}%',fontsize=9.4,color=col,ha='right')
    map_note=fig.text(.628,.093,'Map: marine portions on archived paths.',color=MUTED,fontsize=8.2,fontstyle='italic')
    fig.text(.628,.062,'Counts: all proxy cells, including port buffers.',color=MUTED,fontsize=8.2,fontstyle='italic')
    pdf=ROOT/'Figure_S01.pdf'
    fig.savefig(pdf,dpi=430);fig.savefig(ROOT/'Figure_S01.png',dpi=350)
    from PIL import Image
    map_image.set_data(np.asarray(Image.open(ROOT/'Full_grid_at_sea.png')))
    panel_title.set_text('Full proxy grid at sea')
    map_note.set_text('Map: marine portions of all archived proxy cells.')
    fig.savefig(ROOT/'Figure_S01_full_grid.pdf',dpi=430);plt.close(fig)
    caption='Figure S1. EU endpoint proxy. (a) Ports flagged as EU/EEA endpoints in the 2024 study network; symbol size increases with the number of distinct study corridors incident to each port. Selected ports and seas are labeled for orientation. Water depth (Natural Earth) provides geographic context only. (b) Global distribution of archived H3 grid-cell centers, colored by assigned proxy weight. (c) Counts and shares of all distinct proxy cells by weight; the denominator is the global proxy layer, not the regional port sample. The figure does not delineate EU ETS jurisdiction.'
    caption=caption.replace('Global distribution of archived H3 grid-cell centers, colored by assigned proxy weight.', 'Marine portions of native H3 resolution-5 proxy cells occurring on archived representative corridor paths, clipped to the Natural Earth 1:10 million ocean layer. Cells belonging solely to port buffers are omitted from this panel. Colors retain the assigned proxy weights.')
    (ROOT/'Caption_S1.txt').write_text(caption+'\n',encoding='utf8')
    ports.to_csv(ROOT/'S1_ports.csv',index=False)
    page=fitz.open(pdf)
    names=[font[3] for font in page[0].get_fonts() if 'TimesNewRoman' in font[3] and page.extract_font(font[0])[3]]
    assert len(names)==len(page[0].get_fonts()) and names
    page[0].get_pixmap(matrix=fitz.Matrix(1.85,1.85)).save(ROOT/'Preview_S1.png')
    (ROOT/'checks.json').write_text(json.dumps({'fonts':names,'global_grid_counts':{str(key):int(val) for key,val in counts.items()},'flagged_ports_all_extents':len(ports),'flagged_corridors':ends.corridor_id.nunique(),'source_geo_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'basemap').glob('*.zip')}},indent=2))
if __name__=='__main__':draw()

