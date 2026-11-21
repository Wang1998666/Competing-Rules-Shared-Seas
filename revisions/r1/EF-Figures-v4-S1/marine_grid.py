from pathlib import Path
import json
import numpy as np
import pandas as pd
import shapefile
import h3
from PIL import Image, ImageDraw
from matplotlib.colors import to_rgb
import cartopy.crs as crs
from shapely.geometry import shape

ROOT=Path(__file__).resolve().parent
WIDTH=14400
HEIGHT=7200

def pixel_coords(ring):
    coords=np.asarray(ring)
    return list(map(tuple,np.rint(np.c_[(coords[:,0]+180)*WIDTH/360-.5,
                                        (90-coords[:,1])*HEIGHT/180-.5]).astype('i4')))

def ocean_mask():
    mask=Image.new('L',(WIDTH,HEIGHT))
    draw=ImageDraw.Draw(mask)
    with shapefile.Reader(str(ROOT/'basemap/ne_10m_ocean.shp')) as source:
        for record in source.iterShapes():
            geom=shape(record.__geo_interface__)
            for polygon in sorted(geom.geoms,key=lambda p:p.area,reverse=True):
                draw.polygon(pixel_coords(polygon.exterior.coords),fill=1)
                for hole in polygon.interiors:draw.polygon(pixel_coords(hole.coords),fill=0)
    return mask

def rgb(hexcol):
    return (np.asarray(to_rgb(hexcol))*255).round().astype('uint8')

def build(grid,paths):
    onpath=grid.h3_id.isin(paths.h3_id)
    flags=grid.assign(onpath=onpath)
    masks=[Image.new('L',(WIDTH,HEIGHT)) for _ in range(2)]
    pens=[ImageDraw.Draw(image) for image in masks]
    for cell,weight,matched in flags[['h3_id','weight','onpath']].itertuples(index=False,name=None):
        assert h3.get_resolution(cell)==5
        boundary=np.asarray(h3.cell_to_boundary(cell))[:,::-1].copy()
        boundary[:,0]=np.rad2deg(np.unwrap(np.deg2rad(boundary[:,0])))
        shifts=[0]
        if boundary[:,0].min()<-180:shifts.append(360)
        if boundary[:,0].max()>180:shifts.append(-360)
        value=2 if weight==1. else 1
        for shift in shifts:
            vertices=pixel_coords(boundary+[shift,0])
            pens[0].polygon(vertices,fill=value)
            if matched:pens[1].polygon(vertices,fill=value)
    del pens
    water=np.asarray(ocean_mask());masks=[np.asarray(mask) for mask in masks]
    assert all(water[int((90-lat)*HEIGHT/180),int((lon+180)*WIDTH/360)]==expected
               for lon,lat,expected in [(2.35,48.86,0),(13.4,52.5,0),(9.1,45.5,0),(-30,36,1),(2.7,55.4,1)])
    globe=crs.Robinson();angular=crs.PlateCarree()
    x0,x1=globe.x_limits;y0,y1=globe.y_limits
    width=3300;height=round(width*(y1-y0)/(x1-x0))
    xs=np.linspace(x0,x1,width,endpoint=False)+(x1-x0)/width/2
    ys=np.linspace(y1,y0,height,endpoint=False)-(y1-y0)/height/2
    views=[np.zeros((height,width,4),dtype='uint8') for _ in masks]
    land=rgb('#E6E4DB');sea=rgb('#F4F8F7')
    for start in range(0,height,32):
        xx,yy=np.meshgrid(xs,ys[start:start+32])
        values=angular.transform_points(globe,xx,yy)
        lon,lat=values[:,:,0],values[:,:,1]
        valid=np.isfinite(lon)&np.isfinite(lat)
        ix=np.clip(((np.nan_to_num(lon)+180)*WIDTH/360).astype(int),0,WIDTH-1)
        iy=np.clip(((90-np.nan_to_num(lat))*HEIGHT/180).astype(int),0,HEIGHT-1)
        wet=(water[iy,ix]>0)&valid
        for array,mask in zip(views,masks):
            stripe=array[start:start+len(yy)]
            stripe[:,:,:3]=land;stripe[wet,:3]=sea;stripe[:,:,3]=255*valid
            for level,color in [(1,'#AC642D'),(2,'#246B68')]:
                stripe[wet&(mask[iy,ix]==level),:3]=rgb(color)
            assert np.all(stripe[valid&~wet,:3]==land)
    for arr,name in zip(views,['Full_grid_at_sea','Archived_routes_at_sea']):
        Image.fromarray(arr).save(ROOT/(name+'.png'))
    report=flags.groupby(['weight','onpath']).size().rename('archived_cells_before_land_clip')
    report.to_csv(ROOT/'grid_display_scope.csv')
    assert flags.onpath.sum()==15264 and len(grid)==111191
    return views[1],(x0,x1,y0,y1)

