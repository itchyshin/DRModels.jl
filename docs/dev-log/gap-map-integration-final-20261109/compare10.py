#!/usr/bin/env python3
"""compare10.py <w8_gap_map.tsv> <w10_gap_map.tsv> <outdir>. Same rules as wave-8 PREREGISTRATION.md, w8 as base."""
import sys, csv, collections, os
a_f, b_f, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
L = lambda f: {r["cell_id"]: r for r in csv.DictReader(open(f), delimiter="\t")}
A, B = L(a_f), L(b_f)
def cls(r):
    c = r["class"]
    if c == "both-fit": return "both-fit-agree" if r["ll_flag"] == "ok" else "both-fit-disagree"
    return c
CLS = ["both-fit-agree","both-fit-disagree","R-only","Julia-only","both-refuse","crash-Julia","crash-R","nonconverged","inconclusive"]
ca = collections.Counter(cls(A[i]) for i in A); cb = collections.Counter(cls(B[i]) for i in B)
with open(out+"/outcome_table.tsv","w") as f:
    f.write("class\twave8\twave10\n")
    for c in CLS: f.write("%s\t%d\t%d\n"%(c,ca.get(c,0),cb.get(c,0)))
    f.write("Julia fits\t%d\t%d\n"%(sum(r["j_outcome"]=="FITS" for r in A.values()),sum(r["j_outcome"]=="FITS" for r in B.values())))
    f.write("R fits\t%d\t%d\n"%(sum(r["r_outcome"]=="FITS" for r in A.values()),sum(r["r_outcome"]=="FITS" for r in B.values())))
    f.write("total\t%d\t%d\n"%(len(A),len(B)))
trans = collections.Counter(); flip=[]; worse=[]; rows=[]
for i in A:
    a,b = A[i],B[i]
    trans[(cls(a),cls(b))]+=1
    row=dict(cell_id=i,w8_class=cls(a),w10_class=cls(b),w8_j=a["j_outcome"],w10_j=b["j_outcome"],r_outcome=a["r_outcome"],r_ll=a["r_ll"],w8_j_ll=a["j_ll"],w10_j_ll=b["j_ll"],w8_ll_diff=a["ll_diff"],w10_ll_diff=b["ll_diff"],w10_j_msg=b["j_msg"][:200])
    rows.append(row)
    if a["r_outcome"]!=b["r_outcome"] or a["r_ll"]!=b["r_ll"]: row["note"]="R-side differs"
    if a["class"] in ("R-only","nonconverged","inconclusive","crash-Julia") and a["r_outcome"]=="FITS" and b["j_outcome"]=="FITS": flip.append(row)
    w=[]
    if a["j_outcome"]=="FITS" and b["j_outcome"]!="FITS": w.append("fit-to-notfit")
    if cls(a)=="both-fit-agree" and cls(b)=="both-fit-disagree": w.append("agree-to-disagree")
    if a["j_outcome"]=="REFUSES" and b["j_outcome"]=="ERRORS": w.append("refuse-to-crash")
    if b["class"].startswith("crash") and not a["class"].startswith("crash"): w.append("new-crash")
    if a["j_outcome"]=="FITS" and b["j_outcome"]=="FITS":
        try:
            if abs(float(a["j_ll"])-float(b["j_ll"]))>1e-6: w.append("ll-moved>1e-6(%.3g)"%(float(b["j_ll"])-float(a["j_ll"])))
        except ValueError: pass
    if w: row=dict(row,rule=";".join(w)); worse.append(row)
def dump(name, rs):
    cols=list(rows[0].keys())+(["rule"] if name=="got_worse" else [])
    with open(out+"/"+name+".tsv","w") as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="ignore"); w.writeheader()
        for r in rs: w.writerow(r)
dump("flipped_to_fit",flip); dump("got_worse",worse)
with open(out+"/transitions.tsv","w") as f:
    f.write("w8_class\tw10_class\tcells\n")
    for (x,y),n in sorted(trans.items(), key=lambda t:-t[1]): f.write("%s\t%s\t%d\n"%(x,y,n))
print("flipped",len(flip),"worse",len(worse))
for r in worse: print(r["cell_id"],r["rule"])
print({k:v for k,v in trans.items() if k[0]!=k[1]})
