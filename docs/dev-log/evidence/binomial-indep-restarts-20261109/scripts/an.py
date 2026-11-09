import sys,statistics as st,collections
rows={}
for l in open(sys.argv[1]):
    p=l.rstrip('\n').split('\t')
    if p[3]=='ERR': print('ERR',p); continue
    rows[(int(p[0]),int(p[1]),p[2])]=(float(p[3]),p[4],float(p[5]),float(p[6]),float(p[7]),float(p[8]))
print(len(rows),'rows')
for q in (2,3):
    seeds=sorted({k[0] for k in rows if k[1]==q})
    print('q',q,'seeds',len(seeds))
    for mode in ('off','trigger','force'):
        low5=low2=0;conv=0
        for s in seeds:
            best=max(rows[(s,q,m)][0] for m in ('off','trigger','force') if (s,q,m) in rows)
            ll=rows[(s,q,mode)][0]
            low5+= ll<best-1e-5; low2+= ll<best-1e-2; conv+=rows[(s,q,mode)][1]=='true'
        print(' ',mode,'lower than best-of: >1e-5:',low5,' >1e-2:',low2,' converged:',conv)
    for mode in ('trigger','force'):
        r=[rows[(s,q,mode)][2]/rows[(s,q,'off')][2] for s in seeds]
        ran=sum(rows[(s,q,mode)][3] for s in seeds)
        print(' ',mode,'cost ratio median %.2f max %.2f; restarts ran in %d/%d'%(st.median(r),max(r),ran,len(seeds)))
    print('  off time median %.3fs'%st.median(rows[(s,q,'off')][2] for s in seeds))
    for s in seeds:
        o=rows[(s,q,'off')][0];f=rows[(s,q,'force')][0];t=rows[(s,q,'trigger')][0]
        if f-o>1e-5 or t-o>1e-5: print('   seed',s,'off %.4f trig %.4f force %.4f'%(o,t,f), 'winner',rows[(s,q,'force')][4])
