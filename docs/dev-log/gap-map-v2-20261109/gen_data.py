#!/usr/bin/env python3
"""Deterministic data generator for the gap-map v2 grid (stdlib only).

RNG: splitmix64 (integer arithmetic, bit-identical on every platform and Python
version). Uniforms are the top 53 bits / 2^53. Normals use Box-Muller; the output
CSV is committed, so any last-bit libm difference in log/cos cannot reach the grid.
Usage: python3 gen_data.py <outdir>
Writes data_v2.csv (one row per observation, many response columns).
"""
import math, sys, os

M64 = (1 << 64) - 1

class SM64:
    def __init__(self, seed):
        self.s = seed & M64
    def u64(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)
    def unif(self):
        return ((self.u64() >> 11) + 0.5) / float(1 << 53)   # in (0, 1), never 0 or 1
    def norm(self):
        u1, u2 = self.unif(), self.unif()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    def exp1(self):
        return -math.log(self.unif())
    def gamma(self, a):               # Marsaglia-Tsang, shape a, scale 1
        if a < 1.0:
            return self.gamma(a + 1.0) * self.unif() ** (1.0 / a)
        d = a - 1.0 / 3.0; c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = self.norm(); v = 1.0 + c * x
            if v <= 0: continue
            v = v ** 3; u = self.unif()
            if u < 1 - 0.0331 * x ** 4 or math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
                return d * v
    def pois(self, lam):
        if lam > 30:                  # not reached by the generators below, kept for safety
            return max(0, int(round(lam + math.sqrt(lam) * self.norm())))
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.unif()
            if p <= L: return k
            k += 1

inv_logit = lambda z: 1.0 / (1.0 + math.exp(-z))

def main(outdir):
    r = SM64(20261109)
    n, G, G2 = 160, 16, 8
    eg = [0.5 * r.norm() for _ in range(G)]
    eg2 = [0.4 * r.norm() for _ in range(G2)]
    ehp = [0.4 * r.norm() for _ in range(2 * G)]
    rows = []
    cut = (-1.0, 0.0, 1.2)
    for i in range(n):
        gi = i % G                      # g cycles: every group has 10 rows, well mixed
        g2i = (i * 3 + i // G) % G2
        hi = 2 * gi + (i // G) % 2      # h nested in g (2 sub-groups per g)
        f = "abc"[(i * 7 + i // 5) % 3]
        f2 = "uv"[(i // 3) % 2]
        x1, x2, x3 = r.norm(), r.norm(), r.norm()
        off = 0.3 * r.norm()
        w = 0.5 + 1.5 * r.unif()
        wi = 1 + int(4 * r.unif())
        tr = 5 + int(11 * r.unif())
        fe = {"a": 0.0, "b": 0.4, "c": -0.3}[f] + (0.25 if f2 == "v" else 0.0)
        a1, a2 = eg[gi], eg2[g2i]
        d = dict(id=i + 1, g="g%02d" % (gi + 1), g2="k%d" % (g2i + 1), h="g%02d%s" % (gi + 1, "ab"[hi % 2]),
                 f=f, f2=f2, x1=x1, x2=x2, x3=x3, off=off, w=w, wi=wi, tr=tr)
        # gaussian, with sigma depending on x1
        d["y_g"] = 1.0 + 0.5 * x1 - 0.3 * x2 + fe + a1 + a2 + math.exp(-0.2 + 0.3 * x1) * r.norm()
        d["y_ln"] = math.exp(0.5 + 0.3 * x1 + 0.5 * fe + a1 + math.exp(-0.7 + 0.2 * x2) * r.norm())
        mu_g = math.exp(1.0 + 0.3 * x1 + 0.5 * fe + a1)
        d["y_gam"] = mu_g * r.gamma(4.0) / 4.0
        lam = math.exp(0.5 + 0.4 * x1 + 0.3 * fe + a1 + off)
        d["y_pois"] = r.pois(lam)
        lam_nb = math.exp(0.7 + 0.3 * x1 + 0.3 * fe + a1 + off) * r.gamma(4.0) / 4.0
        d["y_nb"] = r.pois(min(lam_nb, 28.0))
        p = inv_logit(-0.2 + 0.8 * x1 - 0.5 * x2 + 0.4 * fe + a1)
        d["y_bin"] = 1 if r.unif() < p else 0
        s = sum(1 for _ in range(tr) if r.unif() < p)
        d["s"], d["fl"] = s, tr - s
        mub = inv_logit(0.2 + 0.6 * x1 + 0.3 * fe + a1); phi = 8.0
        ga, gb = r.gamma(mub * phi), r.gamma((1 - mub) * phi)
        yb = ga / (ga + gb); yb = min(max(yb, 1e-4), 1 - 1e-4)
        d["y_beta"] = yb
        ub = r.unif()
        d["y_zoib"] = 0.0 if ub < 0.08 else (1.0 if ub > 0.92 else yb)
        # zero-inflated Poisson / NB: structural zero prob depends on x2
        pz = inv_logit(-1.0 + 0.6 * x2)
        d["y_zip"] = 0 if r.unif() < pz else r.pois(lam)
        d["y_zinb"] = 0 if r.unif() < pz else r.pois(min(lam_nb, 28.0))
        # hurdle: zero with prob depending on x1, otherwise zero-truncated NB (rejection)
        if r.unif() < inv_logit(-0.5 + 0.5 * x1):
            d["y_hu"] = 0
        else:
            while True:
                k = r.pois(min(math.exp(0.9 + 0.3 * x1 + a1) * r.gamma(4.0) / 4.0, 28.0))
                if k > 0: break
            d["y_hu"] = k
        while True:
            k = r.pois(min(math.exp(0.9 + 0.3 * x1 + a1) * r.gamma(4.0) / 4.0, 28.0))
            if k > 0: break
        d["y_tnb"] = k
        while True:
            k = r.pois(math.exp(0.6 + 0.4 * x1 + a1))
            if k > 0: break
        d["y_tp"] = k
        lat = 0.8 * x1 + 0.5 * (f == "b") + a1 + (r.unif() - 0.5 + 0.0)  # latent with logistic-ish noise below
        ul = r.unif(); lat = 0.8 * x1 + 0.5 * (f == "b") + a1 + math.log(ul / (1 - ul))
        d["y_ord"] = 1 + sum(1 for c in cut if lat > c)
        e1 = [0.0, 0.7 * x1 - 0.2, -0.5 * x1 + 0.3]
        pr = [math.exp(v) for v in e1]; tot = sum(pr); uu = r.unif() * tot
        d["y_cat"] = "c1" if uu < pr[0] else ("c2" if uu < pr[0] + pr[1] else "c3")
        # student nu = 5, scale depends on x2
        chi2 = 2.0 * r.gamma(2.5)
        d["y_st"] = 1.0 + 0.5 * x1 + fe + math.exp(-0.3 + 0.2 * x2) * r.norm() / math.sqrt(chi2 / 5.0)
        # skew-normal: delta = 0.8
        dl = 0.8
        d["y_sk"] = 0.5 + 0.4 * x1 + fe + (dl * abs(r.norm()) + math.sqrt(1 - dl * dl) * r.norm())
        # tweedie p = 1.5 (compound Poisson-gamma, alpha = 1), phi = 1.2
        mut = math.exp(0.5 + 0.3 * x1 + 0.3 * fe + 0.5 * a1); ph = 1.2
        lamt = mut ** 0.5 / (ph * 0.5); sc = ph * 0.5 * mut ** 0.5
        nn = r.pois(min(lamt, 28.0)); d["y_tw"] = sum(r.exp1() * sc for _ in range(nn))
        # bivariate Gaussian with rho depending on x1
        rho = math.tanh(0.5 + 0.4 * x1)
        z1, z2 = r.norm(), r.norm()
        e1b, e2b = z1, rho * z1 + math.sqrt(1 - rho * rho) * z2
        d["y1"] = 0.5 + 0.6 * x1 + a1 + math.exp(-0.2 + 0.2 * x2) * e1b
        d["y2"] = -0.3 + 0.4 * x2 + 0.3 * a1 + math.exp(-0.4) * e2b
        d["y1m"] = "NA" if i % 11 == 3 else d["y1"]
        d["y2m"] = "NA" if i % 13 == 5 else d["y2"]
        d["y1p"] = math.exp(0.3 + 0.3 * x1 + 0.3 * e1b)
        d["y2p"] = math.exp(0.1 + 0.2 * x2 + 0.3 * e2b)
        # bivariate Student-t pair (shared nu = 6)
        c2 = 2.0 * r.gamma(3.0); sq = math.sqrt(c2 / 6.0)
        d["ys1"] = 0.4 + 0.5 * x1 + 0.7 * e1b / sq
        d["ys2"] = -0.2 + 0.3 * x2 + 0.7 * e2b / sq
        # meta-analysis: known sampling variances vi
        vi = 0.02 + 0.18 * r.unif()
        d["vi"] = vi
        d["yi"] = 0.3 + 0.4 * x1 + 0.3 * a1 + 0.2 * r.norm() + math.sqrt(vi) * r.norm()
        # count-binomial overdispersed (beta-binomial) successes
        pbb = min(max(r.gamma(mub * 6.0) / (r.gamma(mub * 6.0) + r.gamma((1 - mub) * 6.0)), 1e-3), 1 - 1e-3)
        sb = sum(1 for _ in range(tr) if r.unif() < pbb)
        d["sb"], d["flb"] = sb, tr - sb
        # ---- derived columns appended last so the draws above keep the same stream ----
        d["gn"] = gi + 1                                   # numeric grouping code
        d["gna"] = "NA" if i % 17 == 4 else d["g"]         # grouping factor with missing labels
        d["gu"] = ("g%02d" % (gi + 1)) if gi < 5 else ("s%03d" % i if i % 4 == 0 else "g%02d" % (gi + 1))  # unbalanced, with singletons
        d["x1m"] = "NA" if i % 9 == 2 else d["x1"]          # predictor with missing values
        d["y_gm"] = "NA" if i % 10 == 7 else d["y_g"]       # gaussian response with missing values
        d["y_pm"] = "NA" if i % 10 == 4 else d["y_pois"]    # Poisson response with missing values
        d["y_ordf"] = "o%d" % d["y_ord"]                    # ordinal response as a label
        rows.append(d)
    cols = list(rows[0].keys())
    with open(os.path.join(outdir, "data_v2.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for d in rows:
            fh.write(",".join(v if isinstance(v, str) else (repr(v) if isinstance(v, float) else str(v)) for v in (d[c] for c in cols)) + "\n")
    print("wrote", len(rows), "rows,", len(cols), "columns")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
