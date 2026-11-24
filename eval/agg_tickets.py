import csv, collections, datetime, json, os
src=r"E:\hackathon-data\ticketing\October-Delhi.csv"
out=r"E:\hackathon-work\eval"
os.makedirs(out,exist_ok=True)
od=collections.Counter(); odh=collections.defaultdict(lambda:[0]*24); stops=collections.Counter()
hours=collections.Counter(); dow=collections.Counter(); n=0
with open(src,encoding="utf-8",errors="replace",newline="") as f:
    r=csv.reader(f); next(r)
    for row in r:
        n+=1
        try:
            s=row[6].strip(); e=row[4].strip(); rt=row[5].strip(); ts=row[7]
            h=int(ts[11:13]); d=datetime.date(int(ts[0:4]),int(ts[5:7]),int(ts[8:10])).weekday()
        except Exception: continue
        k=(s,e,rt); od[k]+=1; odh[k][h]+=1; hours[h]+=1; dow[d]+=1; stops[s]+=1; stops[e]+=1
with open(out+r"\od_route.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["start","end","route","tickets","hours"])
    for k,c in od.most_common():
        if c<10: break
        w.writerow([*k,c," ".join(map(str,odh[k]))])
with open(out+r"\stop_names.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["name","count"])
    for k,c in stops.most_common(): w.writerow([k,c])
json.dump({"rows":n,"hours":dict(sorted(hours.items())),"dow":dict(sorted(dow.items())),"n_od_route":len(od)},open(out+r"\summary.json","w"))
print(n,len(od),len(stops))
