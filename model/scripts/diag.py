import sqlite3, collections, numpy as np, random
W=r"E:\hackathon-work"; LAT0=28.6; KX=111320.0*np.cos(np.radians(LAT0)); KY=110540.0
shapes=collections.defaultdict(list)
for line in list(open(W+r"\route_shapes.csv"))[1:]:
    r,k,la,lo,d=line.split(","); shapes[int(r)].append((float(la),float(lo)))
G={}
for r,p in shapes.items():
    a=np.array(p); G[r]=((a[:,1]-77)*KX,(a[:,0]-LAT0)*KY)
def frac(r,px,py):
    x,y=G[r]
    # point-to-vertex distance approx (shape points dense?) use segments
    ax,ay,dx,dy=x[:-1],y[:-1],np.diff(x),np.diff(y); L2=np.maximum(dx*dx+dy*dy,1e-9)
    X=px[:,None];Y=py[:,None]; t=np.clip(((X-ax)*dx+(Y-ay)*dy)/L2,0,1)
    d=np.hypot(ax+t*dx-X,ay+t*dy-Y).min(1); return (d<150).mean()
con=sqlite3.connect(r"file:E:\hackathon-data\bus-gps\bus_movements_2026_10_15.db?mode=ro",uri=True)
T=collections.defaultdict(list)
for vid,tid,rid,la,lo in con.execute("select vehicle_id,trip_id,route_id,lat,lng from vehicle_feed limit 4000000"):
    if rid is not None: T[(vid,tid,int(rid))].append((float(la),float(lo)))
random.seed(1); keys=[k for k in T if len(T[k])>60 and k[2] in G]; random.shuffle(keys)
res=collections.Counter(); shown=0
for k in keys[:400]:
    a=np.array(T[k][::max(1,len(T[k])//30)]); px=(a[:,1]-77)*KX; py=(a[:,0]-LAT0)*KY
    f=frac(k[2],px,py)
    if f>0.7: res["fits_own"]+=1; continue
    best=max(G, key=lambda r: frac(r,px[::3],py[::3]))
    fb=frac(best,px,py)
    res["own_bad_other_fits" if fb>0.7 else "nothing_fits"]+=1
    if shown<15: print(k, round(f,2), "best", best, round(fb,2)); shown+=1
print(res)
