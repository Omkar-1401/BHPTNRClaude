"""EXPLORATION pass 15b (login-node arithmetic only): P1 collapse test of the pass-12 per-case shape amplitudes against candidate spin variables."""
import json, sys, numpy as np
sys.path[:0]=['../spin_dep_split','../spin_dep','..','.']
import surfinBH
from ot_join import kerr_isco
from physics_common import w220_exact
R=json.load(open('p12/p12_spline_v3.json')); rows=R['rows']; n=len(R['knots'])
Z=np.array([r['z'] for r in rows]); q=np.array([r['q'] for r in rows]); c=np.array([r['chi1'] for r in rows])
Zb, Za = Z[:,:n], Z[:,n:-3]
u,sv,vt=np.linalg.svd(Zb,full_matrices=False); A=Zb@vt[0]
u2,sv2,vt2=np.linalg.svd(Za,full_matrices=False); B=Za@vt2[0]
print('corr(A,B)',np.corrcoef(A,B)[0,1],' B/A slope',np.polyfit(A,B,1))
nu=q/(1+q)**2; X1=q/(1+q)
F=surfinBH.LoadFits('NRSur3dq8Remnant')
chif=np.array([F.chif(qq,[0,0,cc],[0,0,0])[0][2] for qq,cc in zip(q,c)])
iso=[kerr_isco(cc) for cc in c]; print(type(iso[0]), iso[0])
Om=np.array([i[1] for i in iso]); risco=np.array([i[0] for i in iso])
w=np.array([np.real(w220_exact(cc)) for cc in c]); wf=np.array([np.real(w220_exact(cf)) for cf in chif])
mf=np.array([F.mf(qq,[0,0,cc],[0,0,0])[0] for qq,cc in zip(q,c)])
br=w*mf/wf
V=dict(chi1=c, X1chi1=X1*c, chif=chif, Omisco=Om, ln_Omisco=np.log(Om), r_isco=risco, ln_risco=np.log(risco), w220=w, ln_br=np.log(br), chif_m_chi1=chif-c)
def fit(cols,y):
    M=np.column_stack([np.ones_like(y)]+cols); co,*_=np.linalg.lstsq(M,y,rcond=None); r=y-M@co; return np.sqrt(np.mean(r**2))
for name,y in (('A (beta shape amp)',A),('B (alpha shape amp)',B)):
    print(name,'rms',np.std(y).round(4),' current form chi,chi2,nu:',fit([c,c**2,nu],y).round(4), ' chi,chi2:',fit([c,c**2],y).round(4))
    for k,v in V.items():
        print(f'   {k:12s} 1var {fit([v],y):.4f}  +nu {fit([v,nu],y):.4f}  +v2 {fit([v,v**2],y):.4f}')
