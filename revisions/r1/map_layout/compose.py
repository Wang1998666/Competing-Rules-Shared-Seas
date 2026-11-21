from pathlib import Path
import sys,re
from PIL import Image,ImageChops
here=Path(__file__).resolve().parent
deps=here.parents[1]/'EF-R1/analysis/_deps'
if deps.is_dir():sys.path.insert(0,str(deps))
import pymupdf as fitz
src=here/'map_sources'
out=here.parent/'figures_main';out.mkdir(exist_ok=True)
def asset(name):
 return fitz.open(src/(name+'.pdf'))
font=Path('C:/Windows/Fonts/timesbd.ttf')
ink=(0.14,0.21,0.25)
def label(pg,x,y,s,size=10.5):
 pg.insert_text((x,y),s,fontname='TimesBold',fontfile=str(font),fontsize=size,color=ink)
def trimmed(pg):
 pix=pg.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
 im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples).convert('L').point(lambda v:255 if v<241 else 0)
 a,b,c,d=im.getbbox()
 return fitz.Rect(0,max(b-6,0),pg.rect.width,min(d+6,pg.rect.height))
def change_letters(doc,old,new):
 pg=doc[0];items=[]
 for line in [line for b in pg.get_text('dict')['blocks'] for line in b.get('lines',[])]:
  for sp in line['spans']:
   if sp['text'] in old and sp['size']>=10:
    items.append(sp)
 assert {s['text'] for s in items}==set(old) and len(items)==len(old),(old,items)
 for sp in items:
  r=fitz.Rect(sp['bbox']);pg.add_redact_annot(r,fill=False)
 pg.apply_redactions(images=0,graphics=0,text=0)
 for sp in items:label(pg,*sp['origin'],new[old.index(sp['text'])],sp['size'])
def merge(n,above,below,oldletters,newletters,top_width=584):
 top=asset(above);bottom=asset(below);change_letters(bottom,oldletters,newletters)
 parts=[(top,trimmed(top[0])),(bottom,trimmed(bottom[0]))]
 w=584;ws=[top_width,584];hs=[ws[i]*r.height/r.width for i,(d,r) in enumerate(parts)]
 doc=fitz.open();pg=doc.new_page(width=w,height=sum(hs)+9)
 y=0
 for (d,r),ww,h in zip(parts,ws,hs):
  pg.show_pdf_page(fitz.Rect((w-ww)/2,y,(w+ww)/2,y+h),d,0,clip=r)
  y+=h+9
 doc.save(out/f'Figure_{n:02}.pdf',garbage=4,deflate=True);doc.close()
def view(pg,d,c,r):
 pg.show_pdf_page(fitz.Rect(*r),d,0,clip=fitz.Rect(*c))
def comparison():
 # All three columns use the same archived selections as the source maps.
 m=asset('M6');s=asset('S4');e=asset('S6')
 doc=fitz.open();p=doc.new_page(width=584,height=420)
 names=['Global plan','EU priority plan','Closed BRI plan']
 for row,(plan,offset) in enumerate(zip(names,[0,114.22,228.44])):
  yy=8+row*129
  label(p,7,yy+10,'abc'[row]+'   '+plan)
  label(p,224,yy+10,'def'[row]+'   European ports')
  if row==0:view(p,s,(12,27,336,216),(2,yy+17,217,yy+115))
  else:view(p,m,(12,28+offset,305,126+offset),(2,yy+18,217,yy+114))
  xx=row*190.1
  view(p,e,(12+xx,26,188+xx,170),(221,yy+17,372,yy+114))
  label(p,7,yy+123,['211 ports, 2,758 routes','182 ports, 1,762 routes','228 ports, 2,352 routes'][row],8.7)
 for row,(docin,c,name) in enumerate([(m,(359,45,569,179),'Worldwide reduction'),(m,(359,229,569,352),'Selected ports by group'),(s,(371,48,557,210),'Global selection rates')]):
  yy=8+row*129
  label(p,381,yy+10,'ghi'[row]+'   '+name)
  view(p,docin,c,(377,yy+20,577,yy+122))
 # Two original keys correspond to the same ordered geographic groups.
 view(p,m,(12,351,205,375),(9,396,203,419))
 view(p,m,(323,351,499,375),(225,396,399,419))
 doc.save(out/'Figure_06.pdf',garbage=4,deflate=True)
if __name__=='__main__':
 merge(2,'M2','S7','ab','ef')
 merge(4,'M4','S3','abc','def')
 comparison()
 merge(7,'S1','S2','abc','def')

