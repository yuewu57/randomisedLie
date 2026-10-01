from pathlib import Path
import importlib.util, json, math, argparse
import numpy as np
from scipy.linalg import expm
from numpy.polynomial.legendre import leggauss
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Independent algebraic and smooth-constant checks for the v4 draft.')
parser.add_argument('--out',type=Path,default=p.parent/'results')
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('lie',p/'randomised_lie_verification_v4_YW_01102026.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out={}; dyad={}
for alpha in [.25,.5,.75]:
 r=2**(-alpha); e=[]
 for k in range(1,17):
  h=2.**(-k); val=m.rough(h,alpha,0.)[0]-m.rough(0.,alpha,0.)[0]
  exact=(1-r)/(2*r-1)*(h**alpha-h); e.append(abs(val-exact))
 dyad[str(alpha)]=max(e)
out['rough_dyadic_formula_max_abs_error']=dyad
rng=np.random.default_rng(846); qe=[];re=[]
for _ in range(20):
 v=rng.normal(size=3);y=rng.normal(size=3);y/=np.linalg.norm(y)
 E=expm(m.hat(v)); q=m.qexp(v)
 qe.append(np.linalg.norm(np.column_stack([m.qapply(q,np.eye(3)[j]) for j in range(3)])-E))
 re.append(np.linalg.norm(m.rotate(v,y)-E@y))
out['quaternion_exponential_vs_scipy_expm']=max(qe);out['Rodrigues_vs_scipy_expm']=max(re)
sol,referr=m.smooth_reference(1.);qT=sol.sol(1.);RT=np.column_stack([m.qapply(qT,np.eye(3)[j]) for j in range(3)])
results=[]
for n in [128,256]:
 x,w=leggauss(n);ts=(x+1)/2;w=w/2;B=np.zeros((3,3));V=0.;pi=np.pi
 for t,wt in zip(ts,w):
  om,_=m.data(t,1.,1.,True)
  op=np.array([.7*pi*np.cos(2*pi*t)-.6*pi*np.sin(6*pi*t),-.8*pi*np.sin(2*pi*t+.3),pi*np.cos(4*pi*t+.2)])
  opp=np.array([-1.4*pi*pi*np.sin(2*pi*t)-3.6*pi*pi*np.cos(6*pi*t),-1.6*pi*pi*np.cos(2*pi*t+.3),-4*pi*pi*np.sin(4*pi*t+.2)])
  q=sol.sol(t);Rt=np.column_stack([m.qapply(q,np.eye(3)[j]) for j in range(3)])
  A=m.hat(om);Ap=m.hat(op)
  B+=wt*(RT@Rt.T@(A@Ap-Ap@A)@Rt)/12
  V+=wt*np.linalg.norm(m.hat(opp))**2/720
 results.append({'quadrature_points':n,'B':B.tolist(),'B_Frobenius':float(np.linalg.norm(B)),'V':float(V)})
out['smooth_linear_bias_variance_constants']=results
out['B_refinement_difference']=float(np.linalg.norm(np.array(results[0]['B'])-np.array(results[1]['B'])))
out['V_refinement_difference']=abs(results[0]['V']-results[1]['V'])
# Finite-difference convergence of the primitive is intentionally not tested
# against machine precision for a Holder derivative.
fd={}
for alpha in [.25,.5,.75]:
 seq=[]
 for eps in [1e-3,1e-4,1e-5,1e-6]:
  es=[]
  for t in [.137,.283,.719]:
   deriv=(m.rough(t+eps,alpha,3/16)[1]-m.rough(t-eps,alpha,3/16)[1])/(2*eps)
   es.append(abs(deriv-m.rough(t,alpha,3/16)[0]))
  seq.append({'epsilon':eps,'max_abs_error':max(es)})
 fd[str(alpha)]=seq
out['rough_primitive_FD_diagnostic']=fd
out['notes']='Binary tail is summed algebraically, subject to floating-point arithmetic. Finite-difference errors for Holder derivatives are not reference-solver errors. Smooth integral and DOP853 refinement checks are numerical diagnostics, not rigorous enclosures.'
assert max(dyad.values())<1e-13
assert max(qe)<1e-13 and max(re)<1e-13
out['algebraic_check_assertions']='passed'
path=args.out/'independent_checks_v4_YW_01102026.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
