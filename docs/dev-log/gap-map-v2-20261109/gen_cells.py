#!/usr/bin/env python3
"""Cell generator for the gap-map v2 grid. Writes cells.tsv.

One cell = one model spelled twice, once as an R drmTMB call and once as a Julia
DRModels call, both evaluated by the runners with the data frame `d` in scope
(Julia also has F("y ~ x") for @formula). An empty `jl` means Julia has no spelling
for the request (for example a non-canonical link on a family whose Julia type has no
link field); the classifier reports it as "no Julia syntax", never as a crash.
Columns: id, area, chunk, probes (std|ext), r, jl, note.
"""
import sys

cells, seen = [], set()

def add(area, name, r, jl, chunk=None, probes="std", note=""):
    cid = "%s|%s" % (area, name)
    assert cid not in seen, cid
    seen.add(cid)
    for s in (cid, r, jl, note):
        assert "\t" not in s and "\n" not in s, cid
    cells.append(dict(id=cid, area=area, chunk=chunk or area, probes=probes, r=r, jl=jl, note=note))

# family key -> (R constructor, Julia constructor)
FAM = {
    "gaussian": ("gaussian()", "DRModels.Gaussian()"),
    "student": ("student()", "DRModels.Student()"),
    "skew": ("skew_normal()", "DRModels.SkewNormal()"),
    "lognormal": ("lognormal()", "DRModels.LogNormal()"),
    "gamma": ('stats::Gamma(link = "log")', "DRModels.Gamma()"),
    "beta": ("beta_family()", "DRModels.Beta()"),
    "poisson": ('stats::poisson(link = "log")', "DRModels.Poisson()"),
    "nb2": ("nbinom2()", "DRModels.NegBinomial2()"),
    "binomial": ('stats::binomial(link = "logit")', "DRModels.Binomial()"),
    "bb": ("beta_binomial()", "DRModels.BetaBinomial()"),
    "zob": ("zero_one_beta()", "DRModels.ZeroOneBeta()"),
    "tweedie": ("tweedie()", "DRModels.Tweedie()"),
    "tnb": ("truncated_nbinom2()", "DRModels.TruncatedNegBinomial2()"),
    "tpois": ("truncated_poisson()", "DRModels.TruncatedPoisson()"),   # R has no such constructor
    "cumul": ("cumulative_logit()", "DRModels.CumulativeLogit()"),
    "categorical": ("categorical()", ""),                               # Julia has no categorical family
}
RESP = {"gaussian": "y_g", "student": "y_st", "skew": "y_sk", "lognormal": "y_ln", "gamma": "y_gam", "beta": "y_beta",
        "poisson": "y_pois", "nb2": "y_nb", "binomial": "y_bin", "bb": "cbind(sb, flb)", "zob": "y_zoib",
        "tweedie": "y_tw", "tnb": "y_tnb", "tpois": "y_tp", "cumul": "y_ord", "categorical": "y_cat"}
HAS_SIGMA = {"gaussian", "student", "skew", "lognormal", "gamma", "beta", "nb2", "bb", "zob", "tweedie", "tnb"}

def uni(fam, mu, dp=None, rx="", jx="", rfam=None, jfam=None, resp=None):
    """Univariate call. mu: right-hand side of the mean formula. dp: ordered dict dpar -> rhs."""
    y = resp or RESP[fam]
    parts_r = ["%s ~ %s" % (y, mu)] + ["%s ~ %s" % (k, v) for k, v in (dp or {}).items()]
    parts_j = ['F("%s ~ %s")' % (y, mu)] + ['F("%s ~ %s")' % (k, v) for k, v in (dp or {}).items()]
    rf = rfam or FAM[fam][0]
    jf = jfam if jfam is not None else FAM[fam][1]
    r = "drmTMB(bf(%s), family = %s, data = d%s)" % (", ".join(parts_r), rf, rx)
    j = "" if jf == "" else "drm(bf(%s), %s; data = d%s)" % (", ".join(parts_j), jf, jx)
    return r, j

def biv(mu1, mu2, s1="1", s2="1", rho="1", nu=None, fam="biv_gaussian()", jfam="DRModels.Gaussian()", rx="", jx=""):
    pr = ["mu1 = %s" % mu1, "mu2 = %s" % mu2, "sigma1 = ~ %s" % s1, "sigma2 = ~ %s" % s2]
    y1, r1 = mu1.split(" ~ ", 1); y2, r2 = mu2.split(" ~ ", 1)
    pj = ['mu1 = F("%s")' % mu1, 'mu2 = F("%s")' % mu2, 'sigma1 = F("sigma1 ~ %s")' % s1, 'sigma2 = F("sigma2 ~ %s")' % s2]
    if nu is not None:
        pr.append("nu = ~ %s" % nu); pj.append('nu = F("nu ~ %s")' % nu)
    if rho is not None:
        pr.append("rho12 = ~ %s" % rho); pj.append('rho12 = F("rho12 ~ %s")' % rho)
    r = "drmTMB(bf(%s), family = %s, data = d%s)" % (", ".join(pr), fam, rx)
    j = "drm(bf(%s), %s; data = d%s)" % (", ".join(pj), jfam, jx)
    return r, j

# =====================================================================================
# A. Distributional-parameter formulas: sigma, nu, zoi, coi on every family that has them
# =====================================================================================
for fam in ["gaussian", "student", "skew", "lognormal", "gamma", "beta", "poisson", "nb2", "binomial", "bb", "zob",
            "tweedie", "tnb", "tpois", "cumul"]:
    for mu in ("x1", "x1 + x2 + f"):
        dp = {"sigma": "1"} if fam in HAS_SIGMA else None
        r, j = uni(fam, mu, dp)
        add("A", "%s|mu=%s|sigma=1" % (fam, mu.replace(" ", "")), r, j, chunk="A-" + fam)
for fam in sorted(HAS_SIGMA):
    for sg in ("x1", "x1 + x2", "f", "x1 + f"):
        r, j = uni(fam, "x1", {"sigma": sg})
        add("A", "%s|mu=x1|sigma=%s" % (fam, sg.replace(" ", "")), r, j, chunk="A-" + fam)
for fam in ("student", "skew", "tweedie"):
    for nu in ("1", "x1", "f"):
        r, j = uni(fam, "x1", {"sigma": "1", "nu": nu})
        add("A", "%s|mu=x1|sigma=1|nu=%s" % (fam, nu), r, j, chunk="A-" + fam)
    r, j = uni(fam, "x1", {"sigma": "x2", "nu": "x1"})
    add("A", "%s|mu=x1|sigma=x2|nu=x1" % fam, r, j, chunk="A-" + fam)
for dp, nm in (({"sigma": "1", "zoi": "x1", "coi": "1"}, "zoi=x1"), ({"sigma": "1", "zoi": "1", "coi": "x1"}, "coi=x1"),
               ({"sigma": "x2", "zoi": "x1", "coi": "x1"}, "sigma=x2|zoi=x1|coi=x1"), ({"sigma": "1", "zoi": "f", "coi": "1"}, "zoi=f")):
    r, j = uni("zob", "x1", dp); add("A", "zob|mu=x1|" + nm, r, j, chunk="A-zob")

# =====================================================================================
# B. Fixed-effect design terms: factors, interactions, transforms, no intercept
# =====================================================================================
TERMS = ["f * x1", "f * f2", "f + f2 + x1:f2", "x1 * x2 * f", "x1 + x2 + x1:x2", "0 + f + x1", "I(x1^2) + x2",
         "poly(x1, 2)", "log(abs(x1) + 1) + x2"]
for fam in ["gaussian", "poisson", "binomial", "gamma", "beta", "nb2", "lognormal", "student"]:
    for t in TERMS:
        dp = {"sigma": "1"} if fam in HAS_SIGMA else None
        r, j = uni(fam, t, dp)
        add("B", "%s|mu=%s" % (fam, t.replace(" ", "")), r, j, chunk="B-" + fam)
for fam in ("gaussian", "gamma", "nb2"):
    for sg in ("f * x1", "I(x1^2)", "0 + f"):
        r, j = uni(fam, "x1", {"sigma": sg})
        add("B", "%s|mu=x1|sigma=%s" % (fam, sg.replace(" ", "")), r, j, chunk="B-" + fam)

# =====================================================================================
# C. Offsets and weights
# =====================================================================================
for fam in ["poisson", "nb2", "gaussian", "gamma", "binomial", "lognormal", "beta", "student", "tweedie", "bb", "tnb"]:
    dp = {"sigma": "1"} if fam in HAS_SIGMA else None
    r, j = uni(fam, "x1 + offset(off)", dp)
    add("C", "%s|offset(off)" % fam, r, j, chunk="C-offset")
for rr, jj, nm in (uni("poisson", "x1 + offset(log(w))", None) + ("offset(log(w))",),
                   uni("poisson", "x1 + offset(off) + offset(log(w))", None) + ("two offsets",),
                   uni("nb2", "x1", {"sigma": "x2 + offset(off)"}) + ("nb2 sigma offset",),
                   uni("poisson", "x1", {"zi": "x2 + offset(off)"}, resp="y_zip") + ("zip zi offset",),
                   uni("poisson", "x1 + offset(off)", {"zi": "x2"}, resp="y_zip") + ("zip mu offset",),
                   uni("nb2", "x1 + offset(off)", {"sigma": "1", "zi": "1"}, resp="y_zinb") + ("zinb mu offset",)):
    add("C", "poisson-family|" + nm, rr, jj, chunk="C-offset")
for fam in ["gaussian", "student", "lognormal", "gamma", "beta", "poisson", "nb2", "binomial", "bb", "tweedie", "skew", "tnb", "zob", "cumul"]:
    dp = {"sigma": "1"} if fam in HAS_SIGMA else None
    r, j = uni(fam, "x1", dp, rx=", weights = w", jx=", weights = d.w")
    add("C", "%s|weights=w" % fam, r, j, chunk="C-weights")
r, j = uni("gaussian", "x1", {"sigma": "x1"}, rx=", weights = w", jx=", weights = d.w"); add("C", "gaussian|sigma=x1|weights=w", r, j, chunk="C-weights")
for fam in ("gaussian", "poisson", "nb2", "binomial"):
    dp = {"sigma": "1"} if fam in HAS_SIGMA else None
    r, j = uni(fam, "x1", dp, rx=", weights = wi", jx=", weights = Float64.(d.wi)"); add("C", "%s|weights=wi(integer)" % fam, r, j, chunk="C-weights")
for fam in ("gaussian", "poisson"):
    dp = {"sigma": "1"} if fam in HAS_SIGMA else None
    r, j = uni(fam, "x1", dp, rx=", weights = w * (id %% 7 != 0)", jx=", weights = d.w .* (d.id .% 7 .!= 0)")
    add("C", "%s|weights-with-zeros" % fam, r, j, chunk="C-weights")
r, j = uni("poisson", "x1 + offset(off)", None, rx=", weights = w", jx=", weights = d.w"); add("C", "poisson|offset+weights", r, j, chunk="C-weights")
r, j = uni("poisson", "x1 + (1 | g)", None, rx=", weights = w", jx=", weights = d.w"); add("C", "poisson|(1|g)+weights", r, j, chunk="C-weights")
r, j = uni("gaussian", "x1 + (1 | g)", {"sigma": "1"}, rx=", weights = w", jx=", weights = d.w"); add("C", "gaussian|(1|g)+weights", r, j, chunk="C-weights")
r, j = uni("poisson", "x1 + offset(off) + (1 | g)", None); add("C", "poisson|(1|g)+offset", r, j, chunk="C-offset")
r, j = uni("nb2", "x1 + offset(off) + (1 | g)", {"sigma": "1"}); add("C", "nb2|(1|g)+offset", r, j, chunk="C-offset")
r, j = uni("gaussian", "x1 + (1 | g)", {"sigma": "1"}, rx=", weights = w, REML = TRUE", jx=", weights = d.w, method = :REML"); add("C", "gaussian|(1|g)+weights+REML", r, j, chunk="C-weights")

# =====================================================================================
# D. Links other than the canonical one (Julia has a link field only on Binomial)
# =====================================================================================
for lk in ("logit", "probit", "cloglog", "cauchit", "log", "identity"):
    r, j = uni("binomial", "x1 + x2", None, rfam='stats::binomial(link = "%s")' % lk, jfam="DRModels.Binomial(:%s)" % lk)
    add("D", "binomial01|link=%s" % lk, r, j, chunk="D")
    r, j = uni("binomial", "x1 + x2", None, resp="cbind(s, fl)", rfam='stats::binomial(link = "%s")' % lk, jfam="DRModels.Binomial(:%s)" % lk)
    if lk in ("logit", "probit", "cloglog", "cauchit"):
        add("D", "binomial-cbind|link=%s" % lk, r, j, chunk="D")
for lk in ("probit", "cloglog"):
    r, j = uni("binomial", "x1 + (1 | g)", None, rfam='stats::binomial(link = "%s")' % lk, jfam="DRModels.Binomial(:%s)" % lk)
    add("D", "binomial01|(1|g)|link=%s" % lk, r, j, chunk="D")
    r, j = uni("binomial", "x1 + (1 + x1 | g)", None, resp="cbind(s, fl)", rfam='stats::binomial(link = "%s")' % lk, jfam="DRModels.Binomial(:%s)" % lk)
    add("D", "binomial-cbind|(1+x1|g)|link=%s" % lk, r, j, chunk="D")
for fam, lks in (("gaussian", ["log", "inverse"]), ("poisson", ["identity", "sqrt"]), ("gamma", ["inverse", "identity"]),
                 ("lognormal", ["log"]), ("nb2", ["identity"])):
    for lk in lks:
        base = {"gaussian": "gaussian", "poisson": "stats::poisson", "gamma": "stats::Gamma", "lognormal": "lognormal", "nb2": "nbinom2"}[fam]
        r, j = uni(fam, "x1", {"sigma": "1"} if fam in HAS_SIGMA else None, rfam='%s(link = "%s")' % (base, lk), jfam="")
        add("D", "%s|link=%s" % (fam, lk), r, j, chunk="D", note="Julia family type has no link field")
for lk in ("probit", "cloglog"):
    r, j = uni("beta", "x1", {"sigma": "1"}, rfam='beta_family(link = "%s")' % lk, jfam="")
    add("D", "beta|link=%s" % lk, r, j, chunk="D", note="Julia family type has no link field")

# =====================================================================================
# E. Zero-inflated and hurdle parts
# =====================================================================================
for zi in ("1", "x1", "x2", "f", "x1 + x2"):
    for mu in ("x1", "x1 + f"):
        r, j = uni("poisson", mu, {"zi": zi}, resp="y_zip"); add("E", "zip|mu=%s|zi=%s" % (mu.replace(" ", ""), zi.replace(" ", "")), r, j, chunk="E")
for sg in ("1", "x1"):
    for zi in ("1", "x2", "f"):
        r, j = uni("nb2", "x1", {"sigma": sg, "zi": zi}, resp="y_zinb"); add("E", "zinb|sigma=%s|zi=%s" % (sg, zi), r, j, chunk="E")
for mu, zi, nm in (("x1 + (1 | g)", "1", "zip|mu=(1|g)|zi=1"), ("x1", "1 + (1 | g)", "zip|zi=(1|g)")):
    r, j = uni("poisson", mu, {"zi": zi}, resp="y_zip"); add("E", nm, r, j, chunk="E")
r, j = uni("nb2", "x1 + (1 | g)", {"sigma": "1", "zi": "1"}, resp="y_zinb"); add("E", "zinb|mu=(1|g)|zi=1", r, j, chunk="E")
r, j = uni("nb2", "x1", {"sigma": "x2 + (1 | g)", "zi": "1"}, resp="y_zinb"); add("E", "zinb|sigma=x2+(1|g)|zi=1", r, j, chunk="E")
for sg in ("1", "x1"):
    for hu in ("1", "x1", "f", "x1 + x2"):
        r, j = uni("tnb", "x1", {"sigma": sg, "hu": hu}, resp="y_hu"); add("E", "hurdle-nb2|sigma=%s|hu=%s" % (sg, hu.replace(" ", "")), r, j, chunk="E")
r, j = uni("tnb", "x1 + (1 | g)", {"sigma": "1", "hu": "1"}, resp="y_hu"); add("E", "hurdle-nb2|mu=(1|g)|hu=1", r, j, chunk="E")
r, j = uni("tnb", "x1 + (1 | g)", {"sigma": "1"}, resp="y_tnb"); add("E", "tnb|mu=(1|g)", r, j, chunk="E")
r, j = uni("tnb", "x1 + offset(off)", {"sigma": "1", "hu": "1"}, resp="y_hu"); add("E", "hurdle-nb2|offset", r, j, chunk="E")
r, j = uni("poisson", "x1", {"hu": "x1"}, resp="y_hu", rfam='stats::poisson(link = "log")'); add("E", "hurdle-poisson|hu=x1", r, j, chunk="E")
r, j = uni("poisson", "x1", {"zi": "x2"}, resp="y_zip", rx=", weights = w", jx=", weights = d.w"); add("E", "zip|weights", r, j, chunk="E")
r, j = uni("binomial", "x1", {"zi": "1"}, resp="y_bin"); add("E", "zero-inflated-binomial|zi=1", r, j, chunk="E")
r, j = uni("gamma", "x1", {"sigma": "1", "zi": "1"}, resp="y_gam"); add("E", "zero-inflated-gamma|zi=1", r, j, chunk="E")
r, j = uni("poisson", "x1", {"zi": "x2"}, resp="y_zinb"); add("E", "zip|response=y_zinb(overdispersed)", r, j, chunk="E")

# =====================================================================================
# F. Student and skew extras
# =====================================================================================
for fam in ("student", "skew"):
    r, j = uni(fam, "x1 + (1 | g)", {"sigma": "1", "nu": "1"}); add("F", "%s|mu=(1|g)" % fam, r, j, chunk="F")
    r, j = uni(fam, "x1 + offset(off)", {"sigma": "1", "nu": "1"}); add("F", "%s|offset" % fam, r, j, chunk="F")
    r, j = uni(fam, "f * x1", {"sigma": "f", "nu": "1"}); add("F", "%s|f*x1|sigma=f" % fam, r, j, chunk="F")
    r, j = uni(fam, "x1", {"sigma": "1", "nu": "1"}, rx=", REML = TRUE", jx=", method = :REML"); add("F", "%s|REML" % fam, r, j, chunk="F")
r, j = uni("student", "x1", {"sigma": "x2", "nu": "x1 + x2"}); add("F", "student|nu=x1+x2", r, j, chunk="F")
r, j = uni("student", "x1", {"sigma": "x2", "nu": "x1"}, resp="y_g"); add("F", "student|response=y_g(light tails)", r, j, chunk="F")
r, j = uni("skew", "x1", {"sigma": "x2", "nu": "x1"}, resp="y_g"); add("F", "skew|response=y_g(symmetric)", r, j, chunk="F")
r, j = uni("gaussian", "x1", {"sigma": "x1"}, resp="y_st", rx=", REML = TRUE", jx=", method = :REML"); add("F", "gaussian|REML|sigma=x1|response=y_st", r, j, chunk="F")
r, j = uni("gaussian", "f * x1", {"sigma": "x1 + f"}, rx=", REML = TRUE", jx=", method = :REML"); add("F", "gaussian|REML|f*x1|sigma=x1+f", r, j, chunk="F")
r, j = uni("lognormal", "x1", {"sigma": "x1"}, rx=", REML = TRUE", jx=", method = :REML"); add("F", "lognormal|REML|sigma=x1", r, j, chunk="F")

# =====================================================================================
# G. Bivariate models
# =====================================================================================
MU = [("y1 ~ x1", "y2 ~ x2"), ("y1 ~ x1 + x2", "y2 ~ x2"), ("y1 ~ x1 + f", "y2 ~ x2 + f2"), ("y1 ~ x1 * x2", "y2 ~ x1"), ("y1 ~ 1", "y2 ~ 1")]
for (m1, m2) in MU:
    for rho in ("1", "x1"):
        r, j = biv(m1, m2, "1", "1", rho)
        add("G", "gauss|%s|%s|rho=%s" % (m1.replace(" ", ""), m2.replace(" ", ""), rho), r, j, chunk="G-gauss")
for s1, s2 in (("x1", "1"), ("x2", "x1"), ("f", "1"), ("x1 + x2", "x1")):
    for rho in ("1", "x1 + x2"):
        r, j = biv("y1 ~ x1", "y2 ~ x2", s1, s2, rho)
        add("G", "gauss|sigma1=%s|sigma2=%s|rho=%s" % (s1.replace(" ", ""), s2, rho.replace(" ", "")), r, j, chunk="G-gauss")
for rho in ("f", "I(x1^2)", "x1 * x2"):
    r, j = biv("y1 ~ x1", "y2 ~ x2", "1", "1", rho); add("G", "gauss|rho=%s" % rho.replace(" ", ""), r, j, chunk="G-gauss")
for rho in ("1", "x1"):
    r, j = biv("y1 ~ x1", "y2 ~ x2", "1", "1", rho, rx=", REML = TRUE", jx=", method = :REML"); add("G", "gauss|REML|rho=%s" % rho, r, j, chunk="G-gauss")
r, j = biv("y1 ~ x1", "y2 ~ x2", "1", "1", "1", rx=", weights = w", jx=", weights = d.w"); add("G", "gauss|weights", r, j, chunk="G-gauss")
r, j = biv("y1 ~ x1 + offset(off)", "y2 ~ x2", "1", "1", "1"); add("G", "gauss|offset", r, j, chunk="G-gauss")
for rho in ("1", "x1"):
    for nm, a, b in (("both-partial", "y1m", "y2m"), ("y1-partial", "y1m", "y2"), ("y2-partial", "y1", "y2m")):
        r, j = biv("%s ~ x1" % a, "%s ~ x2" % b, "1", "1", rho)
        add("G", "gauss|missing=%s|rho=%s|R-default(drop)" % (nm, rho), r, j, chunk="G-miss")
        r, j = biv("%s ~ x1" % a, "%s ~ x2" % b, "1", "1", rho, rx=', missing = miss_control(response = "include")')
        add("G", "gauss|missing=%s|rho=%s|R-include" % (nm, rho), r, j, chunk="G-miss")
r, j = biv("y1m ~ x1", "y2m ~ x2", "1", "1", "1", rx=', REML = TRUE, missing = miss_control(response = "include")', jx=", method = :REML")
add("G", "gauss|missing=both-partial|REML|R-include", r, j, chunk="G-miss")
# random effects in bivariate models
r, j = biv("y1 ~ x1 + (1 | p | g)", "y2 ~ x2 + (1 | p | g)", "1", "1", "1"); add("G", "gauss|RE=(1|p|g) in mu1,mu2|rho=1", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | p | g)", "y2 ~ x2 + (1 | p | g)", "1", "1", "x1"); add("G", "gauss|RE=(1|p|g) in mu1,mu2|rho=x1", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | p | g)", "y2 ~ x2 + (1 | p | g)", "x1", "1", "1"); add("G", "gauss|RE=(1|p|g) in mu1,mu2|sigma1=x1", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | g)", "y2 ~ x2 + (1 | g)", "1", "1", "1"); add("G", "gauss|RE=(1|g) unlabelled in mu1,mu2", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | g)", "y2 ~ x2", "1", "1", "1"); add("G", "gauss|RE=(1|g) in mu1 only", r, j, chunk="G-re")
r, j = biv("y1 ~ x1", "y2 ~ x2", "1 + (1 | p | g)", "1 + (1 | p | g)", "1"); add("G", "gauss|RE=(1|p|g) in sigma1,sigma2", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | p | g) + (1 | q | g2)", "y2 ~ x2 + (1 | p | g) + (1 | q | g2)", "1", "1", "1"); add("G", "gauss|RE=two labelled blocks g,g2", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 + x1 | p | g)", "y2 ~ x2 + (1 + x1 | p | g)", "1", "1", "1"); add("G", "gauss|RE=(1+x1|p|g) in mu1,mu2", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | p | g)", "y2 ~ x2 + (1 | p | g)", "1", "1", "1", rx=", REML = TRUE", jx=", method = :REML"); add("G", "gauss|RE=(1|p|g)|REML", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | p | g)", "y2 ~ x2 + (1 | p | g)", "1", "1", "1", rx=", weights = w", jx=", weights = d.w"); add("G", "gauss|RE=(1|p|g)|weights", r, j, chunk="G-re")
r, j = biv("y1m ~ x1 + (1 | p | g)", "y2m ~ x2 + (1 | p | g)", "1", "1", "1", rx=', missing = miss_control(response = "include")'); add("G", "gauss|RE=(1|p|g)|missing|R-include", r, j, chunk="G-re")
r, j = biv("y1 ~ x1 + (1 | g)", "y2 ~ x2 + (1 | g)", "1", "1", "1", rx=", REML = TRUE", jx=", method = :REML"); add("G", "gauss|RE=(1|g) unlabelled|REML", r, j, chunk="G-re")
# bivariate lognormal / Student
for nm, mu, s, rho in (("base", ("y1p ~ x1", "y2p ~ x2"), ("1", "1"), "1"), ("mu=x1+x2", ("y1p ~ x1 + x2", "y2p ~ x2"), ("1", "1"), "1"),
                       ("sigma1=x1", ("y1p ~ x1", "y2p ~ x2"), ("x1", "1"), "1"), ("rho=x1", ("y1p ~ x1", "y2p ~ x2"), ("1", "1"), "x1"),
                       ("mu=f", ("y1p ~ f", "y2p ~ f2"), ("1", "1"), "1")):
    r, j = biv(mu[0], mu[1], s[0], s[1], rho, fam="biv_lognormal()", jfam="DRModels.LogNormal()"); add("G", "biv_lognormal|" + nm, r, j, chunk="G-lognormal")
r, j = biv("y1p ~ x1 + (1 | p | g)", "y2p ~ x2 + (1 | p | g)", "1", "1", "1", fam="biv_lognormal()", jfam="DRModels.LogNormal()"); add("G", "biv_lognormal|RE=(1|p|g)", r, j, chunk="G-lognormal")
r, j = biv("y1p ~ x1", "y2p ~ x2", "1", "1", "1", fam="biv_lognormal()", jfam="DRModels.LogNormal()", rx=", weights = w", jx=", weights = d.w"); add("G", "biv_lognormal|weights", r, j, chunk="G-lognormal")
for nm, mu, s, nu, rho in (("base", ("ys1 ~ x1", "ys2 ~ x2"), ("1", "1"), "1", "1"), ("sigma1=x1", ("ys1 ~ x1", "ys2 ~ x2"), ("x1", "1"), "1", "1"),
                           ("rho=x1", ("ys1 ~ x1", "ys2 ~ x2"), ("1", "1"), "1", "x1"), ("nu=x1", ("ys1 ~ x1", "ys2 ~ x2"), ("1", "1"), "x1", "1"),
                           ("mu=x1+x2", ("ys1 ~ x1 + x2", "ys2 ~ x2 + f"), ("1", "1"), "1", "1")):
    r, j = biv(mu[0], mu[1], s[0], s[1], rho, nu=nu, fam="biv_student()", jfam="DRModels.Student()"); add("G", "biv_student|" + nm, r, j, chunk="G-student")
r, j = biv("ys1 ~ x1 + (1 | p | g)", "ys2 ~ x2 + (1 | p | g)", "1", "1", "1", nu="1", fam="biv_student()", jfam="DRModels.Student()"); add("G", "biv_student|RE=(1|p|g)", r, j, chunk="G-student")
# mixed-family (cross-family) bivariate
def mixed(f1, f2, m1, m2, s1=None, s2=None, rfam=None, jfam=None):
    pr = ["mu1 = %s" % m1, "mu2 = %s" % m2] + (["sigma1 = ~ %s" % s1] if s1 else []) + (["sigma2 = ~ %s" % s2] if s2 else [])
    pj = ['mu1 = F("%s")' % m1, 'mu2 = F("%s")' % m2] + (['sigma1 = F("sigma1 ~ %s")' % s1] if s1 else []) + (['sigma2 = F("sigma2 ~ %s")' % s2] if s2 else [])
    return ("drmTMB(bf(%s), family = c(%s, %s), data = d)" % (", ".join(pr), FAM[f1][0], FAM[f2][0]),
            "drm(bf(%s), (%s, %s); data = d)" % (", ".join(pj), FAM[f1][1], FAM[f2][1]))
for f1, f2, m1, m2, s1, s2 in (("gaussian", "poisson", "y_g ~ x1", "y_pois ~ x1", "1", None), ("gaussian", "binomial", "y_g ~ x1", "y_bin ~ x1", "1", None),
                               ("gaussian", "nb2", "y_g ~ x1", "y_nb ~ x1", "1", "1"), ("lognormal", "gamma", "y_ln ~ x1", "y_gam ~ x1", "1", "1"),
                               ("poisson", "binomial", "y_pois ~ x1", "y_bin ~ x1", None, None), ("gaussian", "beta", "y_g ~ x1", "y_beta ~ x1", "1", "1")):
    r, j = mixed(f1, f2, m1, m2, s1, s2); add("G", "mixed|%s+%s" % (f1, f2), r, j, chunk="G-mixed")
r, j = mixed("gaussian", "poisson", "y_g ~ x1 + x2", "y_pois ~ x1 + f", "x1", None); add("G", "mixed|gaussian+poisson|sigma1=x1|mu2=x1+f", r, j, chunk="G-mixed")

# =====================================================================================
# H. Meta-analysis and known sampling variance
# =====================================================================================
def meta(mu, sg="1", rx="", jx="", fam="gaussian", mv="meta_V(V = vi)", jmv="meta_V(vi)", resp="yi"):
    dp = {"sigma": sg} if sg is not None else None
    r, _ = uni(fam, "%s + %s" % (mu, mv), dp, rx=rx, resp=resp)
    _, j = uni(fam, "%s + %s" % (mu, jmv), dp, jx=jx, resp=resp)
    return r, j
for mu in ("1", "x1", "x1 + f", "0 + f", "x1 * x2"):
    r, j = meta(mu); add("H", "meta|mu=%s" % mu.replace(" ", ""), r, j, chunk="H")
for sg in ("x1", "f"):
    r, j = meta("x1", sg); add("H", "meta|sigma=%s" % sg, r, j, chunk="H")
r, j = meta("x1 + (1 | g)"); add("H", "meta|(1|g)", r, j, chunk="H")
r, j = meta("x1 + (1 | g) + (1 | g2)"); add("H", "meta|(1|g)+(1|g2)", r, j, chunk="H")
r, j = meta("x1", rx=", REML = TRUE", jx=", method = :REML"); add("H", "meta|REML", r, j, chunk="H")
r, j = meta("x1 + (1 | g)", rx=", REML = TRUE", jx=", method = :REML"); add("H", "meta|(1|g)|REML", r, j, chunk="H")
r, j = meta("x1", mv="meta_known_V(V = vi)"); add("H", "meta|meta_known_V alias", r, j, chunk="H")
r, j = meta("x1", rx=", weights = w", jx=", weights = d.w"); add("H", "meta|weights", r, j, chunk="H")
r, j = meta("x1 + offset(off)"); add("H", "meta|offset", r, j, chunk="H")
r, j = meta("x1", sg=None); add("H", "meta|no sigma formula", r, j, chunk="H")
for fam, resp in (("lognormal", "y_ln"), ("gamma", "y_gam"), ("student", "y_st")):
    dp = {"sigma": "1"}
    r, _ = uni(fam, "x1 + meta_V(V = vi)", dp, resp=resp); _, j = uni(fam, "x1 + meta_V(vi)", dp, resp=resp)
    add("H", "meta|family=%s" % fam, r, j, chunk="H")
r, j = meta("x1", resp="y1m"); add("H", "meta|response with NA", r, j, chunk="H")
r, j = meta("x1", mv="meta_V(V = vi, scale = \"proportional\")", jmv="meta_V(vi)"); add("H", "meta|proportional scale (R planned)", r, "", chunk="H", note="Julia has no scale= argument for meta_V")

# =====================================================================================
# I. Ordinal and categorical responses
# =====================================================================================
for mu in ("x1", "x1 + f", "x1 * f", "f * f2", "x1 + x2 + x3"):
    r, j = uni("cumul", mu); add("I", "cumul|mu=%s" % mu.replace(" ", ""), r, j, chunk="I")
for mu in ("x1 + (1 | g)", "x1 + (1 + x1 | g)", "x1 + (1 | g) + (0 + x1 | g)", "x1 + (1 | g) + (1 | g2)", "x1 + (1 | g/h)"):
    r, j = uni("cumul", mu); add("I", "cumul|mu=%s" % mu.replace(" ", ""), r, j, chunk="I")
r, j = uni("cumul", "x1", rx=", weights = w", jx=", weights = d.w"); add("I", "cumul|weights", r, j, chunk="I")
r, j = uni("cumul", "x1 + offset(off)"); add("I", "cumul|offset", r, j, chunk="I")
r, j = uni("cumul", "x1", {"sigma": "1"}); add("I", "cumul|sigma=1", r, j, chunk="I")
r, j = uni("cumul", "x1", resp="y_ordf"); add("I", "cumul|response=label o1..o4", r, j, chunk="I", note="R reads the label as an ordered factor, Julia sees strings")
r, j = uni("cumul", "x1", resp="y_bin"); add("I", "cumul|two categories", r, j, chunk="I")
r, j = uni("categorical", "x1"); add("I", "categorical|mu=x1", r, j, chunk="I")
r, j = uni("categorical", "x1 + f"); add("I", "categorical|mu=x1+f", r, j, chunk="I")

# =====================================================================================
# J. Random-effect structures the wave-10 grid does not reach (ordinary groups)
# =====================================================================================
for fam in ("gaussian", "poisson", "binomial"):
    dp = {"sigma": "1"} if fam in HAS_SIGMA else None
    for re in ("(1 | g/h)", "(1 | g:f2)", "(f | g)", "(1 + f | g)", "(0 + f | g)", "(1 | g) + (1 | g2) + (1 | h)"):
        r, j = uni(fam, "x1 + " + re, dp); add("J", "%s|%s" % (fam, re.replace(" ", "")), r, j, chunk="J-" + fam)
for fam in ("lognormal", "gamma", "beta", "nb2", "student", "tweedie", "bb", "skew"):
    dp = {"sigma": "1"}
    r, j = uni(fam, "x1 + (1 | g/h)", dp); add("J", "%s|(1|g/h)" % fam, r, j, chunk="J-" + fam)
for fam in ("gaussian", "student", "lognormal", "gamma", "beta", "nb2", "tweedie", "bb", "zob", "skew"):
    r, j = uni(fam, "x1 + (1 | g)", {"sigma": "x1"}); add("J", "%s|mu=(1|g)|sigma=x1" % fam, r, j, chunk="J-sig-" + fam)
for fam in ("gaussian", "lognormal", "gamma", "nb2", "beta", "student", "tweedie", "bb"):
    r, j = uni(fam, "x1", {"sigma": "x1 + (1 | g)"}); add("J", "%s|sigma=x1+(1|g)" % fam, r, j, chunk="J-sigre-" + fam)
r, j = uni("gaussian", "x1", {"sigma": "x1 + (0 + x1 | g)"}); add("J", "gaussian|sigma=x1+(0+x1|g)", r, j, chunk="J-sigre-gaussian")
r, j = uni("gaussian", "x1", {"sigma": "x1 + (1 + x1 | g)"}); add("J", "gaussian|sigma=x1+(1+x1|g)", r, j, chunk="J-sigre-gaussian")
r, j = uni("gaussian", "x1", {"sigma": "x1 + (1 | g) + (1 | g2)"}); add("J", "gaussian|sigma=x1+(1|g)+(1|g2)", r, j, chunk="J-sigre-gaussian")
r, j = uni("gaussian", "x1 + (1 | g)", {"sigma": "x1 + (1 | g2)"}); add("J", "gaussian|mu=(1|g)|sigma=(1|g2)", r, j, chunk="J-sigre-gaussian")
for fam in ("gaussian", "poisson", "binomial"):
    dp = {"sigma": "1"} if fam in HAS_SIGMA else None
    r, j = uni(fam, "x1 + (1 | gn)", dp); add("J", "%s|group=numeric code gn" % fam, r, j, chunk="J-grp")
    r, j = uni(fam, "x1 + (1 | gu)", dp); add("J", "%s|group=unbalanced with singletons gu" % fam, r, j, chunk="J-grp")
r, j = uni("gaussian", "x1 + (1 | gna)", {"sigma": "1"}); add("J", "gaussian|group has NA labels", r, j, chunk="J-grp")
r, j = uni("poisson", "x1 + (1 | gna)", None); add("J", "poisson|group has NA labels", r, j, chunk="J-grp")
r, j = uni("gaussian", "x1 + (1 | id)", {"sigma": "1"}); add("J", "gaussian|group=observation id (n groups = n)", r, j, chunk="J-grp")
r, j = uni("poisson", "x1 + (1 | id)", None); add("J", "poisson|observation-level (1|id)", r, j, chunk="J-grp")
r, j = uni("gaussian", "x1 + (1 | f2)", {"sigma": "1"}); add("J", "gaussian|group has 2 levels (1|f2)", r, j, chunk="J-grp")
r, j = uni("gaussian", "x1 + (1 | g) + (0 + x1 | g) + (0 + x2 | g2)", {"sigma": "1"}); add("J", "gaussian|slopes on two different groups", r, j, chunk="J-grp")
r, j = uni("gaussian", "x1 + (1 | g) + (1 | g)", {"sigma": "1"}); add("J", "gaussian|duplicated term (1|g)+(1|g)", r, j, chunk="J-grp")
r, j = uni("gaussian", "x1 + (1 + x1 || g)", {"sigma": "1"}); add("J", "gaussian|(1+x1||g) double bar", r, j, chunk="J-grp")
r, j = uni("poisson", "x1 + (1 + x1 || g)", None); add("J", "poisson|(1+x1||g) double bar", r, j, chunk="J-grp")

# =====================================================================================
# K. Missing values in the data (non-bivariate)
# =====================================================================================
r, j = uni("gaussian", "x1", {"sigma": "1"}, resp="y_gm"); add("K", "gaussian|NA in response", r, j, chunk="K")
r, j = uni("gaussian", "x1m", {"sigma": "1"}, resp="y_g"); add("K", "gaussian|NA in mu predictor", r, j, chunk="K")
r, j = uni("gaussian", "x2", {"sigma": "x1m"}, resp="y_g"); add("K", "gaussian|NA in sigma predictor", r, j, chunk="K")
r, j = uni("poisson", "x1", None, resp="y_pm"); add("K", "poisson|NA in response", r, j, chunk="K")
r, j = uni("poisson", "x1m", None, resp="y_pois"); add("K", "poisson|NA in predictor", r, j, chunk="K")
r, j = uni("gaussian", "x1m + (1 | g)", {"sigma": "1"}, resp="y_gm"); add("K", "gaussian|NA in both + (1|g)", r, j, chunk="K")
r, j = uni("poisson", "x1m + (1 | g)", None, resp="y_pm"); add("K", "poisson|NA in both + (1|g)", r, j, chunk="K")
r, j = uni("gaussian", "x1", {"sigma": "1"}, resp="y_gm", rx=", weights = w", jx=", weights = d.w"); add("K", "gaussian|NA in response + weights", r, j, chunk="K")
r, j = uni("gaussian", "x1", {"sigma": "1"}, resp="y_gm", rx=', missing = miss_control(response = "include")'); add("K", "gaussian|NA in response|R-include", r, j, chunk="K")
r, j = uni("nb2", "x1m", {"sigma": "1"}, resp="y_nb"); add("K", "nb2|NA in predictor", r, j, chunk="K")
r, j = uni("gaussian", "x1", {"sigma": "1"}, resp="y_gm", rx=", REML = TRUE", jx=", method = :REML"); add("K", "gaussian|NA in response|REML", r, j, chunk="K")

# =====================================================================================
# M. Entry points exercised in depth (probes=ext): prediction, simulation, residuals, intervals, summary
# =====================================================================================
EXT = [("gaussian", "x1 + f", {"sigma": "x1"}), ("gaussian", "x1 + (1 | g)", {"sigma": "1"}), ("poisson", "x1 + f", None),
       ("poisson", "x1 + (1 | g)", None), ("nb2", "x1 + f", {"sigma": "x2"}), ("binomial", "x1 + f", None),
       ("binomial", "x1 + (1 | g)", None), ("gamma", "x1 + f", {"sigma": "x1"}), ("lognormal", "x1 + f", {"sigma": "x1"}),
       ("beta", "x1 + f", {"sigma": "x1"}), ("student", "x1 + f", {"sigma": "x1", "nu": "1"}), ("skew", "x1", {"sigma": "1", "nu": "1"}),
       ("tweedie", "x1 + f", {"sigma": "1", "nu": "1"}), ("zob", "x1", {"sigma": "1", "zoi": "x1", "coi": "1"}), ("bb", "x1 + f", {"sigma": "x1"}),
       ("tnb", "x1 + f", {"sigma": "x1"}), ("cumul", "x1 + f", None), ("cumul", "x1 + (1 | g)", None)]
for fam, mu, dp in EXT:
    r, j = uni(fam, mu, dp); add("M", "%s|mu=%s|entry-points" % (fam, mu.replace(" ", "")), r, j, chunk="M", probes="ext")
r, j = uni("poisson", "x1", {"zi": "x2"}, resp="y_zip"); add("M", "zip|entry-points", r, j, chunk="M", probes="ext")
r, j = uni("tnb", "x1", {"sigma": "1", "hu": "x1"}, resp="y_hu"); add("M", "hurdle-nb2|entry-points", r, j, chunk="M", probes="ext")
r, j = biv("y1 ~ x1 + f", "y2 ~ x2", "x1", "1", "x1"); add("M", "biv_gaussian|entry-points", r, j, chunk="M", probes="ext")
r, j = meta("x1 + f"); add("M", "meta|entry-points", r, j, chunk="M", probes="ext")

with open("cells.tsv", "w") as fh:
    fh.write("id\tarea\tchunk\tprobes\tr\tjl\tnote\n")
    for c in cells:
        fh.write("\t".join(c[k] for k in ("id", "area", "chunk", "probes", "r", "jl", "note")) + "\n")
import collections
print(len(cells), "cells"); print(sorted(collections.Counter(c["area"] for c in cells).items()))
