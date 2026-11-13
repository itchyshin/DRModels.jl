"""Original dense Gaussian reference, independent of both target packages."""
import math
import numpy as np

SHAPE, RATE = 2.0, 3.0

def prior(eta, shape=SHAPE, rate=RATE):
    return rate*math.exp(eta)-(shape-1)*eta-shape*math.log(rate)+math.lgamma(shape)

def prior_terms(eta, shape=SHAPE, rate=RATE):
    """Return prior negative log density, eta derivative and eta curvature."""
    return prior(eta,shape,rate), rate*math.exp(eta)-(shape-1), rate*math.exp(eta)

def arrays(case):
    x, y, group = (np.asarray(case[k]) for k in ('x', 'y', 'group'))
    X = np.column_stack((np.ones(len(x)), x))
    Z = (group[:, None] == np.unique(group)[None, :]).astype(float)
    return x, y, X, Z

def evaluate(case, point, shape=SHAPE, rate=RATE):
    b0, b1, alpha, gamma, eta = np.asarray(point, dtype=float)
    x, y, X, Z = arrays(case)
    d = np.exp(2*(alpha+gamma*x)); K = math.exp(2*eta)*(Z@Z.T)
    V = np.diag(d)+K; A = np.linalg.inv(V); r = y-X@np.array([b0,b1]); u=A@r
    sign, logdet = np.linalg.slogdet(V)
    assert sign > 0
    B=A-np.outer(u,u); zero=np.zeros_like(V)
    Vi=[zero,zero,np.diag(2*d),np.diag(2*x*d),2*K]
    m=[X[:,0],X[:,1],np.zeros(len(x)),np.zeros(len(x)),np.zeros(len(x))]
    Vij=[[zero for j in range(5)] for i in range(5)]
    Vij[2][2]=np.diag(4*d); Vij[2][3]=Vij[3][2]=np.diag(4*x*d)
    Vij[3][3]=np.diag(4*x*x*d); Vij[4][4]=4*K
    P, pg, ph=prior_terms(eta, shape, rate)
    objective=(len(x)*math.log(2*math.pi)+logdet+r@u)/2+P
    g=np.empty(5); h=np.empty((5,5))
    for i in range(5):
        g[i]=-m[i]@u+0.5*np.trace(B@Vi[i])
        for j in range(5):
            du=-A@Vi[j]@u-A@m[j]
            dB=-A@Vi[j]@A-np.outer(du,u)-np.outer(u,du)
            h[i,j]=-m[i]@du+0.5*np.trace(dB@Vi[i]+B@Vij[i][j])
    g[4]+=pg; h[4,4]+=ph
    return float(objective), g, h, P

def scalar(case, point, tau_override=None, shape=SHAPE, rate=RATE):
    """Independent groupwise objective: no dense inverse or analytic derivatives."""
    b0,b1,alpha,gamma,eta=map(float,point)
    x,y,group=(np.asarray(case[k]) for k in ('x','y','group'))
    tau = math.exp(eta) if tau_override is None else tau_override
    value=len(x)*math.log(2*math.pi)/2
    for label in np.unique(group):
        ix=group==label; xx=x[ix]; rr=y[ix]-b0-b1*xx
        d=np.exp(2*(alpha+gamma*xx)); w=1/d
        denom=1+tau*tau*sum(w)
        value += (sum(np.log(d))+math.log(denom)+sum(rr*rr*w)-tau*tau*sum(rr*w)**2/denom)/2
    return float(value+(prior(eta, shape, rate) if tau_override is None else 0))

def boundary(case, point):
    return scalar(case, [*point, 0.0], tau_override=0.0)

def finite_difference(case, point, shape=SHAPE, rate=RATE):
    p=np.asarray(point,dtype=float); g=np.empty(5); h=np.empty((5,5)); hg=1e-5; hh=2e-4
    eye=np.eye(5); f=lambda x: scalar(case,x,shape=shape,rate=rate)
    base=f(p)
    for i in range(5):
        g[i]=(f(p+hg*eye[i])-f(p-hg*eye[i]))/(2*hg)
        h[i,i]=(f(p+hh*eye[i])-2*base+f(p-hh*eye[i]))/(hh*hh)
        for j in range(i):
            h[i,j]=h[j,i]=(f(p+hh*eye[i]+hh*eye[j])-f(p+hh*eye[i]-hh*eye[j])-f(p-hh*eye[i]+hh*eye[j])+f(p-hh*eye[i]-hh*eye[j]))/(4*hh*hh)
    return g,h

def mutation_values(case, point, shape=SHAPE, rate=RATE):
    obj,_,_,p=evaluate(case,point,shape,rate); eta=point[4]
    mutations={'added_jacobian':obj-eta, 'swapped_shape_rate':obj-p+prior(eta,rate,shape),
            'omitted_normalizer':obj+shape*math.log(rate)-math.lgamma(shape),
            'omitted_penalty':obj-p, 'doubled_penalty':obj+p,
            'per_group_penalty':obj+(len(set(case['group']))-1)*p,
            'wrong_target_gamma':obj-p+prior(point[3],shape,rate)}
    if abs(math.lgamma(shape)) > 1e-14:
        mutations['missing_lgamma_only']=obj-math.lgamma(shape)
    return mutations
