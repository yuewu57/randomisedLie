#!/usr/bin/env python3
"""Recreate the four paper plots from recorded statistics (no simulations)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TAG='v4_YW_01102026'

def main() -> None:
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=root/'results')
    parser.add_argument('--out',type=Path,default=root/f'randomised_lie_assets_{TAG}')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=json.loads((args.results/f'experiment_summary_all_{TAG}.json').read_text())
    checks=json.loads((args.results/f'independent_checks_{TAG}.json').read_text())
    cases={c['id']:c for c in data['cases']}
    def rows(cid,m):
        return sorted([r for r in cases[cid]['rows'] if r['method']==m],key=lambda r:r['N'])
    def save(fig,name):
        fig.tight_layout()
        for ext in ('pdf','png'):
            fig.savefig(args.out/f'{name}_{TAG}.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    methods=[('CF_midpoint','Deterministic CF midpoint','o'),('RKMK4','Deterministic RKMK4','s'),
             ('RLCF2','Single-node RLCF2','^'),('paired_RLCF2','Paired-node RLCF2','D')]
    fig,ax=plt.subplots(figsize=(7.1,4.5))
    for method,label,marker in methods:
        rs=rows('rough_S2_a0.5_g1_k2',method)
        n=np.array([r['N'] for r in rs]);e=np.array([r['grid_maximum']['rms'] for r in rs])
        lo=np.array([r['grid_maximum']['ci95_low'] for r in rs]);hi=np.array([r['grid_maximum']['ci95_high'] for r in rs])
        ax.errorbar(n,e,yerr=[e-lo,hi-e],marker=marker,capsize=2,label=label)
    ax.set_xscale('log',base=2);ax.set_yscale('log')
    ax.set_xlabel('Number of time steps N');ax.set_ylabel('RMS of maximum geodesic error over the grid')
    ax.set_title(r'Rough nonlinear $S^2$ problem: $\alpha=0.5$');ax.grid(True,alpha=.25);ax.legend(fontsize=9)
    save(fig,'rough_sphere_convergence')
    fig,ax=plt.subplots(figsize=(7.1,4.5))
    for a,marker in [(.25,'o'),(.5,'s'),(.75,'^')]:
        rs=cases[f'rough_S2_a{a:g}_g1_k2']['coupled']
        ax.loglog([r['N'] for r in rs],[r['single']['rms'] for r in rs],marker=marker,label=fr'$\alpha={a:g}$')
    ax.set_xlabel('Number of time steps N');ax.set_ylabel('RMS of maximum coupled RLCF2–RRKMK2 distance')
    ax.set_title(r'Common-node trajectory comparison on $S^2$');ax.grid(True,alpha=.25);ax.legend()
    save(fig,'coupled_sphere_difference')
    fig,ax=plt.subplots(figsize=(7.1,4.5))
    for method,label,marker in methods:
        rs=rows('smooth_S2_a1_g1_k2',method)
        ax.loglog([r['N'] for r in rs],[r['endpoint']['rms'] for r in rs],marker=marker,label=label)
    ax.set_xlabel('Number of time steps N');ax.set_ylabel('Terminal geodesic error / RMS error')
    ax.set_title(r'Smooth nonlinear $S^2$ control');ax.grid(True,alpha=.25);ax.legend(fontsize=9)
    save(fig,'smooth_sphere_comparison')
    rs=rows('smooth_SO3_a1_g1_k0','paired_RLCF2');n=np.array([r['N'] for r in rs],float);h=1/n
    constants=checks['smooth_linear_bias_variance_constants'][-1]
    fig,ax=plt.subplots(figsize=(7.1,4.5))
    ax.loglog(n,[r['matrix_mean_Frobenius'] for r in rs],marker='o',label='Norm of empirical matrix mean error')
    ax.loglog(n,[np.sqrt(r['matrix_centred_variance']) for r in rs],marker='s',label='Empirical centred RMS matrix error')
    ax.loglog(n,constants['B_Frobenius']*h**2,linestyle='--',label=r'$\|\mathcal{B}_1\|_F h^2$')
    ax.loglog(n,np.sqrt(constants['V'])*h**2.5,linestyle=':',label=r'$\sqrt{\mathcal{V}_1}\,h^{5/2}$')
    ax.set_xlabel('Number of time steps N');ax.set_ylabel('Frobenius error')
    ax.set_title(r'Paired RLCF2 on smooth $SO(3)$: bias and fluctuation');ax.grid(True,alpha=.25);ax.gend()
    save(fig,'smooth_matrix_bias_fluctuation')
    print(f'Four PDF/PNG figures written to {args.out}')

if __name__=='__main__':
    main()
