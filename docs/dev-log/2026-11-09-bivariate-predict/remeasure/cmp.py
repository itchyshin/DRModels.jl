import csv,os,sys,math
def read(p):
    if not os.path.exists(p): return None
    return [[float(x) if x not in('NA','NaN','') else float('nan') for x in r] for r in csv.reader(open(p))]
ri={l.split('\t')[0]:l.rstrip('\n').split('\t') for l in open('r_index.txt')}
ji={l.split('\t')[0]:l.rstrip('\n').split('\t') for l in open('jl_index.txt')}
out=[]
for k in sorted(ri):
    id=ri[k][1]
    for kind in ('fitted','predict'):
        r=read(f'r_out/{k}_{kind}.csv'); j=read(f'jl_out/{k}_{kind}.csv')
        if r is None or j is None: res='missing(R=%s,J=%s)'%(r is not None, j is not None)
        else:
            nr,nj=len(r),len(j); cr,cj=len(r[0]),len(j[0])
            if nr!=nj: res=f'rows R={nr} J={nj}'
            elif cr!=cj:
                # compare first col
                d=max(abs(r[i][0]-j[i][0]) for i in range(nr)); res=f'cols R={cr} J={cj}; first col maxdiff={d:.3g}'
            else:
                d=max(abs(r[i][c]-j[i][c]) for i in range(nr) for c in range(cr) if not(math.isnan(r[i][c]) or math.isnan(j[i][c]))); res=f'maxdiff={d:.3g}'
        out.append((k,id,kind,res))
for k,id,kind,res in out:
    if kind=='fitted': print(k,id[:55].ljust(55),res)
