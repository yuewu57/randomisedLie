#!/usr/bin/env python3
"""Reproducible v4 experiments: exact-reference rough Lie ODEs and smooth checks.

Tested with Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Numba 0.65.1. No network or external data are required.
Run with --help. All randomisation is in the integrator, not the ODE.
The manufactured rough reference is explicit, NOT obtained by time-stepping.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from numba import njit
import numba

TAG = 'v4_YW_01102026'
METHODS = ['RLCF2', 'RRKMK2', 'paired_RLCF2', 'paired_RRKMK2', 'RLE']
DET = ['Lie_Euler', 'CF_midpoint', 'RKMK_midpoint', 'RKMK4']
D = np.array([[.45, .12, -.08], [.12, .85, .10], [-.08, .10, 1.55]])
Y0 = np.array([.7, .4, .5]); Y0 /= np.linalg.norm(Y0)

@njit(cache=True)
def cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

@njit(cache=True)
def qmul(a, b):
    return np.array([a[0]*b[0]-np.dot(a[1:], b[1:]),
                    a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
                    a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
                    a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]])

@njit(cache=True)
def qexp(v):
    s2 = np.dot(v, v)
    if s2 < 1.e-12:
        f = .5-s2/48+s2*s2/3840
        a = 1-s2/8+s2*s2/384
    else:
        s=math.sqrt(s2); f=math.sin(s/2)/s; a=math.cos(s/2)
    return np.array([a, f*v[0], f*v[1], f*v[2]])

@njit(cache=True)
def qapply(q, y):
    # Unit-quaternion action; no numerical projection is used in the integrator.
    v=q[1:]; c=cross(v,y)
    return y+2*(q[0]*c+cross(v,c))

@njit(cache=True)
def rotate(v, y):
    a2=np.dot(v,v)
    if a2 < 1.e-10:
        s=1-a2/6+a2*a2/120
        c=.5-a2/24+a2*a2/720
    else:
        a=math.sqrt(a2); s=math.sin(a)/a; c=2*(math.sin(a/2)/a)**2
    u=cross(v,y)
    return y+s*u+c*cross(v,u)

@njit(cache=True)
def dinv(u, b):
    a2=np.dot(u,u)
    if a2 < 1.e-6:
        c=1/12+a2/720+a2*a2/30240+a2*a2*a2/1209600
    else:
        a=math.sqrt(a2); c=(1-.5*a/math.tan(.5*a))/a2
    z=cross(u,b)
    return b-.5*z+c*cross(u,z)

@njit(cache=True)
def qdistance(q, r):
    rc=np.array([r[0],-r[1],-r[2],-r[3]])
    z=qmul(q,rc)
    return 2*math.atan2(math.sqrt(np.dot(z[1:],z[1:])),abs(z[0]))

@njit(cache=True)
def sdist(y,z):
    return math.atan2(np.linalg.norm(cross(y,z)),np.dot(y,z))

@njit(cache=True)
def pp(z):
    # Periodic primitive of tent(z)-1/4 on [0,1], with pp(0)=pp(1)=0.
    if z <= .5:
        return .5*z*z-.25*z
    return .75*z-.5*z*z-.25

@njit(cache=True)
def rough(t, alpha, phase):
    """Exact infinite tent-series value at a binary floating-point argument.

    g=(1-r) sum r^k [tent({2^k t+phase})-1/4], r=2^-alpha.
    j=int_0^t g(s) ds. Doubling a binary fraction eventually gives zero;
    then the entire remaining geometric tail is added analytically.
    Phase is added AFTER reduction modulo one, avoiding large-argument loss.
    """
    r=2.**(-alpha); w=1-r; invfreq=1.
    frac=t-math.floor(t); g=0.; j=0.; p0=pp(phase)
    const=min(phase,1-phase)-.25
    for k in range(1100):
        if frac == 0.:
            g += w/(1-r)*const
            return g,j
        z=frac+phase
        if z >= 1.: z-=1.
        g += w*(min(z,1-z)-.25)
        j += w*invfreq*(pp(z)-p0)
        frac*=2
        if frac >= 1.: frac-=1.
        w*=r; invfreq*=.5
    raise ValueError('Binary series evaluator failed to terminate')

@njit(cache=True)
def data(t,alpha,gamma,smooth):
    if smooth:
        pi=math.pi
        w=np.array([.8+.35*math.sin(2*pi*t)+.1*math.cos(6*pi*t),
                    gamma*(.65+.4*math.cos(2*pi*t+.3)),
                    .25*gamma*math.sin(4*pi*t+.2)])
        return w,np.array([1.,0.,0.,0.])
    g1,j1=rough(t,alpha,0.)
    g2,j2=rough(t,alpha,3/16)
    th1=.9*t+2*j1; th2=gamma*(.7*t+2*j2)
    b1=.9+2*g1; b2=gamma*(.7+2*g2)
    w=np.array([b1,b2*math.cos(th1),b2*math.sin(th1)])
    c1=math.cos(th1/2); s1=math.sin(th1/2)
    c2=math.cos(th2/2); s2=math.sin(th2/2)
    return w,np.array([c1*c2,s1*c2,c1*s2,s1*s2])

@njit(cache=True)
def sphere_a(w,yr,y,kappa):
    return w+kappa*(D@(y-yr))

@njit(cache=True)
def run_random(N,taus,alpha,gamma,kappa,sphere,smooth,refs):
    P=taus.shape[0]; h=1./N
    terminal=np.zeros((P,5)); maximum=np.zeros((P,5)); gap=np.zeros((P,2))
    defects=np.zeros(5); matrices=np.zeros((P,3,3))
    for p in range(P):
        if sphere:
            ys=np.empty((5,3))
            for m in range(5): ys[m]=Y0
        else:
            qs=np.zeros((5,4)); qs[:,0]=1.
        for n in range(N):
            t=n*h; r=taus[p,n]
            w0,q0=data(t,alpha,gamma,smooth)
            wr,qr=data((n+r)*h,alpha,gamma,smooth)
            ws,qsamp=data((n+1-r)*h,alpha,gamma,smooth)
            if sphere:
                yr0=qapply(q0,Y0); yrr=qapply(qr,Y0); yrs=qapply(qsamp,Y0)
                for m in range(5):
                    y=ys[m].copy()
                    if m == 4:
                        v=h*sphere_a(wr,yrr,y,kappa)
                    else:
                        aa=sphere_a(w0,yr0,y,kappa); u=r*h*aa
                        z=rotate(u,y); b=sphere_a(wr,yrr,z,kappa)
                        if m == 1 or m == 3: b=dinv(u,b)
                        if m >= 2:
                            u2=(1-r)*h*aa; z2=rotate(u2,y)
                            b2=sphere_a(ws,yrs,z2,kappa)
                            if m == 3: b2=dinv(u2,b2)
                            b=.5*(b+b2)
                        v=h*b
                    ys[m]=rotate(v,y)
                    e=sdist(ys[m],refs[n+1])
                    maximum[p,m]=max(maximum[p,m],e)
                    defects[m]=max(defects[m],abs(np.linalg.norm(ys[m])-1))
                gap[p,0]=max(gap[p,0],sdist(ys[0],ys[1]))
                gap[p,1]=max(gap[p,1],sdist(ys[2],ys[3]))
            else:
                vs=np.empty((5,3))
                vs[0]=h*wr; vs[1]=h*dinv(r*h*w0,wr)
                vs[2]=.5*h*(wr+ws)
                vs[3]=.5*h*(dinv(r*h*w0,wr)+dinv((1-r)*h*w0,ws))
                vs[4]=vs[0]
                for m in range(5):
                    qs[m]=qmul(qexp(vs[m]),qs[m])
                    e=qdistance(qs[m],refs[n+1])
                    maximum[p,m]=max(maximum[p,m],e)
                    defects[m]=max(defects[m],abs(np.linalg.norm(qs[m])-1))
                gap[p,0]=max(gap[p,0],qdistance(qs[0],qs[1]))
                gap[p,1]=max(gap[p,1],qdistance(qs[2],qs[3]))
        for m in range(5):
            terminal[p,m]=sdist(ys[m],refs[-1]) if sphere else qdistance(qs[m],refs[-1])
        if not sphere:
            for j in range(3):
                ej=np.zeros(3); ej[j]=1.; matrices[p,:,j]=qapply(qs[2],ej)
    return terminal,maximum,gap,defects,matrices

@njit(cache=True)
def run_det(N,alpha,gamma,kappa,sphere,smooth,refs):
    h=1./N; terminal=np.zeros(4); maximum=np.zeros(4)
    if sphere:
        ys=np.empty((4,3))
        for m in range(4): ys[m]=Y0
    else:
        qs=np.zeros((4,4)); qs[:,0]=1.
    for n in range(N):
        t=n*h
        w0,q0=data(t,alpha,gamma,smooth)
        wm,qm=data((n+.5)*h,alpha,gamma,smooth)
        w1,q1=data((n+1)*h,alpha,gamma,smooth)
        for m in range(4):
            if sphere:
                y=ys[m].copy(); a=sphere_a(w0,qapply(q0,Y0),y,kappa)
                if m==0: v=h*a
                elif m==1 or m==2:
                    u=.5*h*a; b=sphere_a(wm,qapply(qm,Y0),rotate(u,y),kappa)
                    v=h*b if m==1 else h*dinv(u,b)
                else:
                    u2=.5*h*a
                    k2=dinv(u2,sphere_a(wm,qapply(qm,Y0),rotate(u2,y),kappa))
                    u3=.5*h*k2
                    k3=dinv(u3,sphere_a(wm,qapply(qm,Y0),rotate(u3,y),kappa))
                    u4=h*k3
                    k4=dinv(u4,sphere_a(w1,qapply(q1,Y0),rotate(u4,y),kappa))
                    v=h*(a+2*k2+2*k3+k4)/6
                ys[m]=rotate(v,y); e=sdist(ys[m],refs[n+1])
            else:
                if m==0: v=h*w0
                elif m==1: v=h*wm
                elif m==2: v=h*dinv(.5*h*w0,wm)
                else:
                    k2=dinv(.5*h*w0,wm); k3=dinv(.5*h*k2,wm); k4=dinv(h*k3,w1)
                    v=h*(w0+2*k2+2*k3+k4)/6
                qs[m]=qmul(qexp(v),qs[m]); e=qdistance(qs[m],refs[n+1])
            maximum[m]=max(maximum[m],e)
    for m in range(4):
        terminal[m]=sdist(ys[m],refs[-1]) if sphere else qdistance(qs[m],refs[-1])
    return terminal,maximum


def hat(w):
    return np.array([[0.,-w[2],w[1]],[w[2],0.,-w[0]],[-w[1],w[0],0.]])

def smooth_reference(gamma,kappa=0.,sphere=False):
    def rhs(t,q):
        w,_=data(t,.5,gamma,True)
        if sphere:
            return np.cross(w+kappa*(D@(q-Y0)),q)
        return .5*qmul(np.r_[0.,w],q)
    initial=Y0.copy() if sphere else np.array([1.,0.,0.,0.])
    sol=solve_ivp(rhs,(0.,1.),initial,method='DOP853',rtol=2.3e-14,atol=2e-15,
                  max_step=1/64,dense_output=True)
    sol2=solve_ivp(rhs,(0.,1.),initial,method='DOP853',rtol=2.3e-14,atol=1e-15,
                   max_step=1/128,dense_output=True)
    if not sol.success or not sol2.success: raise RuntimeError('Reference solve failed')
    return sol2, float(sdist(sol.y[:,-1],sol2.y[:,-1]) if sphere else qdistance(sol.y[:,-1],sol2.y[:,-1]))

def refs_for(N,alpha,gamma,sphere,smooth,ref=None):
    if smooth:
        return ref.sol(np.linspace(0,1,N+1)).T.copy()
    qs=np.array([data(n/N,alpha,gamma,False)[1] for n in range(N+1)])
    return np.array([qapply(q,Y0) for q in qs]) if sphere else qs

def rms_stats(x):
    xx=x*x; mu=float(np.mean(xx)); se=float(np.std(xx,ddof=1)/math.sqrt(len(x)))
    return dict(rms=math.sqrt(mu),ci95_low=math.sqrt(max(0,mu-1.96*se)),
                ci95_high=math.sqrt(mu+1.96*se),sample_max=float(np.max(x)))

def work(method,sphere,N):
    if sphere:
        c={'RLCF2':(2,2,0),'RRKMK2':(2,2,1),'paired_RLCF2':(3,3,0),
           'paired_RRKMK2':(3,3,2),'RLE':(1,1,0),'Lie_Euler':(1,1,0),
           'CF_midpoint':(2,2,0),'RKMK_midpoint':(2,2,1),'RKMK4':(4,4,3)}[method]
    else:
        c={'RLCF2':(1,1,0),'RRKMK2':(2,1,1),'paired_RLCF2':(2,1,0),
           'paired_RRKMK2':(3,1,2),'RLE':(1,1,0),'Lie_Euler':(1,1,0),
           'CF_midpoint':(1,1,0),'RKMK_midpoint':(2,1,1),'RKMK4':(3,1,3)}[method]
    # Counts of distinct evaluations per step; no reuse across neighbouring steps.
    return dict(coefficient_evaluations=N*c[0],exponential_actions=N*c[1],dexp_inverse_actions=N*c[2])

def run_case(alpha,gamma,kappa,sphere,smooth,Ns,P,seed,outdir,label):
    ref,check=smooth_reference(gamma,kappa,sphere) if smooth else (None,0.)
    rows=[]; coupled=[]; last_matrices=None
    case_id=f'{label}_a{alpha:g}_g{gamma:g}_k{kappa:g}'
    print('CASE',case_id,'paths',P,'reference_check',check,flush=True)
    for N in Ns:
        refs=refs_for(N,alpha,gamma,sphere,smooth,ref)
        # Independent reproducible grids; methods at a given N share all nodes.
        rng=np.random.default_rng(np.random.SeedSequence([seed,N]))
        taus=rng.random((P,N))
        tm=time.perf_counter()
        te,me,ga,de,mat=run_random(N,taus,alpha,gamma,kappa,sphere,smooth,refs)
        dt=time.perf_counter()-tm
        dte,dme=run_det(N,alpha,gamma,kappa,sphere,smooth,refs)
        for i,m in enumerate(METHODS):
            rr=dict(case=case_id,geometry='S2' if sphere else 'SO3',smooth=smooth,
                    alpha=alpha,gamma=gamma,kappa=kappa,N=N,h=1/N,paths=P,method=m,
                    endpoint=rms_stats(te[:,i]),grid_maximum=rms_stats(me[:,i]),
                    norm_defect=float(de[i]),**work(m,sphere,N))
            if not sphere and m=='paired_RLCF2':
                Rref=np.column_stack([qapply(refs[-1],np.eye(3)[j]) for j in range(3)])
                emat=mat-Rref; mean=emat.mean(axis=0)
                rr['matrix_mean_Frobenius']=float(np.linalg.norm(mean))
                rr['matrix_centred_variance']=float(np.sum((emat-mean)**2)/(P-1))
            rows.append(rr)
        for i,m in enumerate(DET):
            rows.append(dict(case=case_id,geometry='S2' if sphere else 'SO3',smooth=smooth,
                    alpha=alpha,gamma=gamma,kappa=kappa,N=N,h=1/N,paths=1,method=m,
                    endpoint=dict(rms=float(dte[i]),ci95_low=float(dte[i]),ci95_high=float(dte[i]),sample_max=float(dte[i])),
                    grid_maximum=dict(rms=float(dme[i]),ci95_low=float(dme[i]),ci95_high=float(dme[i]),sample_max=float(dme[i])),
                    **work(m,sphere,N)))
        coupled.append(dict(N=N,h=1/N,single=rms_stats(ga[:,0]),paired=rms_stats(ga[:,1]),
                            single_max_scaled=float(np.max(ga[:,0])*N**(1+alpha)),
                            paired_max_scaled=float(np.max(ga[:,1])*N**(1+alpha))))
        print(N,'CF %.4e'%np.sqrt(np.mean(te[:,0]**2)),
              'paired %.4e'%np.sqrt(np.mean(te[:,2]**2)),
              'mid %.4e'%dte[1],'RK4 %.4e'%dte[3], 'sec %.2f'%dt,flush=True)
    case=dict(id=case_id,alpha=alpha,gamma=gamma,kappa=kappa,sphere=sphere,smooth=smooth,
              paths=P,seed=seed,Ns=Ns,reference_refinement_difference=check,
              rows=rows,coupled=coupled)
    (outdir/f'{case_id}_{TAG}.json').write_text(json.dumps(case,indent=2)+'\n')
    return case

def checks():
    # Basic primitive and SO(3) Jacobian diagnostics. Independent exponential
    # checks and smooth asymptotic constants are in the companion check script;
    # fourth-order convergence is tested by the smooth experiment cases.
    out={}
    for alpha in [.25,.5,.75]:
        err=[]
        for t in [.137,.283,.719]:
            eps=1e-6
            gd=rough(t,alpha,3/16)[0]
            finite=(rough(t+eps,alpha,3/16)[1]-rough(t-eps,alpha,3/16)[1])/(2*eps)
            err.append(abs(gd-finite))
        out[f'primitive_finite_difference_a{alpha}']=max(err)
    from scipy.linalg import expm
    u=np.array([.2,-.12,.17]); v=np.array([-.4,.7,.3]); eps=1e-6
    Eu=expm(hat(u))
    rhs=(expm(hat(u+eps*dinv(u,v)))-expm(hat(u-eps*dinv(u,v))))/(2*eps)@Eu.T
    out['right_dexp_inverse_directional_residual']=float(np.linalg.norm(rhs-hat(v)))
    out['exact_reference_endpoint_quaternion']=data(1.,.5,1.,False)[1].tolist()
    out['infinite_series_dyadic_value']=float(rough(.5,.5,0.)[0])
    out['endpoint_primitive']=float(rough(1.,.25,3/16)[1])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(f'results_{TAG}'))
    p.add_argument('--paths',type=int,default=1024)
    p.add_argument('--mode',choices=['main','offgrid','stress','smooth','all','checks'],default='all')
    p.add_argument('--maxN',type=int,default=1024)
    args=p.parse_args()
    if args.paths<2 or args.maxN<32: p.error('--paths >= 2 and --maxN >= 32 required')
    args.out.mkdir(parents=True,exist_ok=True)
    cases=[]; Ns=[n for n in [32,64,128,256,512,1024] if n<=args.maxN]
    if args.mode in ['main','all']:
        for alpha in [.25,.5,.75]:
            for sphere in [False,True]:
                cases.append(run_case(alpha,1.,2. if sphere else 0.,sphere,False,Ns,args.paths,
                                      1102026+int(alpha*1000)+int(sphere)*10000,args.out,'rough_'+('S2' if sphere else 'SO3')))
    if args.mode in ['offgrid','all']:
        on=[n for n in [48,96,192,384,768] if n<=args.maxN]
        for sphere in [False,True]:
            cases.append(run_case(.5,1.,2. if sphere else 0.,sphere,False,on,args.paths,23456,args.out,'offgrid_'+('S2' if sphere else 'SO3')))
        # Approximately matched budget of 1024 coefficient evaluations on S2.
        if args.maxN >= 341:
            cases.append(run_case(.5,1.,2.,True,False,[341],args.paths,23456,args.out,'cost_S2'))
    if args.mode in ['stress','all']:
        cases.append(run_case(.5,5.,4.,True,False,Ns,args.paths,34567,args.out,'stress_S2'))
    if args.mode in ['smooth','all']:
        cases.append(run_case(1.,1.,0.,False,True,Ns,args.paths,45678,args.out,'smooth_SO3'))
        cases.append(run_case(1.,1.,2.,True,True,Ns,args.paths,56789,args.out,'smooth_S2'))
    meta=dict(version=TAG,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
              numba=numba.__version__,platform=platform.platform(),mode=args.mode,checks=checks(),cases=cases)
    (args.out/f'experiment_summary_{args.mode}_{TAG}.json').write_text(json.dumps(meta,indent=2)+'\n')
    flat=[]
    for ca in cases:
        for row in ca['rows']:
            v={k:a for k,a in row.items() if k not in ['endpoint','grid_maximum']}
            for name in ['endpoint','grid_maximum']:
                v.update({name+'_'+k:a for k,a in row[name].items()})
            flat.append(v)
    if flat:
        fields=sorted(set().union(*(r.keys() for r in flat)))
        with (args.out/f'experiment_table_{args.mode}_{TAG}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(flat)
    print('COMPLETED',args.mode,flush=True)

if __name__=='__main__':
    main()
