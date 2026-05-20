"""
Complete Sensitivity Analysis for Airport Taxi Queue Optimizer.

6 Analyses, each answering a specific research question:
  1. Delay sensitivity       -> When do delays matter?
  2. Commit size / lookahead -> What planning horizon is needed?
  3. VoT & cost sensitivity  -> How does the passenger-driver tradeoff work?
  4. Demand scaling          -> Is the policy robust under congestion?
  5. Interval length delta   -> Is PCA discretization accurate?
  6. SS vs Transient gap     -> When does transient modeling become necessary?

Usage:
    python sensitivity_analysis.py --analysis all --n_intervals 15
    python sensitivity_analysis.py --analysis delay --n_intervals 288
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, time, copy
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from config import QueueConfig
from data import load_default_data, aggregate_passengers
from model.generator import build_Q_non_erlang_vec, build_P_from_Q, make_state_vectors
from model.simulation import uniformized_with_checkpoint_blocks

# ================================================================
# SHARED INFRASTRUCTURE
# ================================================================

def build_eff_nr_zero_pad(mu_0, mu_add, mu_remove, pad_mu0, pad_mus):
    is_torch = torch.is_tensor(mu_0)
    n = len(mu_0); mu_eff = mu_0 - mu_remove
    if is_torch:
        mu0_d = torch.zeros_like(mu_eff); mus_d = torch.zeros_like(mu_add)
    else:
        mu0_d = np.zeros_like(mu_eff); mus_d = np.zeros_like(mu_add)
    if 0 < pad_mu0 < n: mu0_d[pad_mu0:] = mu_eff[:-pad_mu0]
    elif pad_mu0 == 0:   mu0_d[:] = mu_eff
    if 0 < pad_mus < n:  mus_d[pad_mus:] = mu_add[:-pad_mus]
    elif pad_mus == 0:    mus_d[:] = mu_add
    return mu0_d + mus_d

def build_window_eff_nr(el, wo, maw, mrw, mac, mrc, m0, p0, ps, nt, dev, dt):
    maf = torch.zeros(nt, device=dev, dtype=dt); mrf = torch.zeros(nt, device=dev, dtype=dt)
    if el > 0:
        maf[:el] = torch.tensor(mac[:el], device=dev, dtype=dt)
        mrf[:el] = torch.tensor(mrc[:el], device=dev, dtype=dt)
    maf[el:el+wo] = maw; mrf[el:el+wo] = mrw
    return build_eff_nr_zero_pad(m0, maf, mrf, p0, ps)[el:el+wo]

def make_pi0(cfg, dev, dt):
    Nn = cfg.K_P + cfg.M + 1; N = (cfg.K_S+1)*Nn
    pi0 = torch.zeros(N, dtype=dt, device=dev); pi0[cfg.M] = 1.0; return pi0

def _unif(pi, Q, W, il, dev, dt):
    P, g = build_P_from_Q(Q); P = P.coalesce()
    return uniformized_with_checkpoint_blocks(pi, P.indices()[0], P.indices()[1], P.values(), g, W, il, max_K_cap=30000, tol_tail=1e-12, block_size=60)

def compute_objective_detailed(pi0, eff, lam, a1v, a2v, ma, mr, cfg, dev, dt):
    KS, KP, M = cfg.K_S, cfg.K_P, cfg.M
    sv = make_state_vectors(KS, KP, M, device=dev, dtype=dt)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'], sv['w_block_pax'], sv['w_block_taxi']], dim=0)
    obj = torch.tensor(0.0, device=dev, dtype=dt)
    tp=tt=tr_=ta=trm=tpl=ttl=0.0; pi = pi0
    for j in range(len(lam)):
        p=lam[j]; c=eff[j]; a1=a1v[j]; a2=a2v[j]; ctl=cfg.fuel_cost+cfg.time_to_city*a2; d=cfg.interval_length
        Q,_,_ = build_Q_non_erlang_vec(K_S=KS,K_P=KP,M=M,lam=c,alpha=p,tau=cfg.tau,device=dev,dtype=dt)
        Ap,Ar,At,Abp,Abt,piT = _unif(pi,Q,W,cfg.interval_length,dev,dt)
        cp_=a1*Ap; ct_=a2*(At+Ar); ca_=ma[j]*d*cfg.cost_per_vehicle_add
        cr_=mr[j]*d*ctl; cpl=cfg.cost_pax_lost*p*Abp; ctl_=ctl*c*Abt
        obj=obj+cp_+ct_+ca_+cr_+cpl+ctl_
        tp+=cp_.item(); tt+=ct_.item(); ta+=ca_.item(); trm+=cr_.item(); tpl+=cpl.item(); ttl+=ctl_.item()
        pi=piT
    return obj, {'pax_wait':tp,'taxi_idle':tt,'add_cost':ta,'remove_cost':trm,'pax_block':tpl,'taxi_block':ttl}

@torch.no_grad()
def propagate_pi(pi0, eff, lam, cfg, dev, dt):
    KS, KP, M = cfg.K_S, cfg.K_P, cfg.M
    sv = make_state_vectors(KS, KP, M, device=dev, dtype=dt)
    W = torch.stack([sv['w_pass'], sv['w_stage'], sv['w_pick'], sv['w_block_pax'], sv['w_block_taxi']], dim=0)
    pi = pi0.clone()
    for j in range(len(lam)):
        Q,_,_ = build_Q_non_erlang_vec(K_S=KS,K_P=KP,M=M,lam=float(eff[j]),alpha=float(lam[j]),tau=cfg.tau,device=dev,dtype=dt)
        _,_,_,_,_,pi = _unif(pi,Q,W,cfg.interval_length,dev,dt)
    return pi

# ================================================================
# OPTIMIZERS
# ================================================================

def run_full_day(lambdas, mus_init, alpha1, alpha2, config, max_iter=300, lr=1.0, epsilon=1e-2, seed=42, device='cpu', dtype=torch.float32):
    torch.manual_seed(seed); rng=np.random.RandomState(seed); n=len(lambdas)
    p0,ps=config.get_delay_blocks(); pi0=make_pi0(config,device,dtype)
    lt=torch.tensor(lambdas,dtype=dtype,device=device); m0=torch.tensor(mus_init,dtype=dtype,device=device)
    a1t=torch.tensor(alpha1,dtype=dtype,device=device); a2t=torch.tensor(alpha2,dtype=dtype,device=device)
    ma=torch.nn.Parameter(torch.tensor(rng.uniform(0,0.1,n),dtype=dtype,device=device))
    mr=torch.nn.Parameter(torch.tensor(rng.uniform(0,0.05,n),dtype=dtype,device=device))
    opt=torch.optim.Adam([ma,mr],lr=lr); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.1,patience=15)
    prev=None
    for step in range(max_iter):
        opt.zero_grad(); eff=build_eff_nr_zero_pad(m0,ma,mr,p0,ps)
        obj,_=compute_objective_detailed(pi0,eff,lt,a1t,a2t,ma,mr,config,device,dtype); obj.backward(); opt.step()
        with torch.no_grad():
            ma.data.clamp_(min=0.0); mr.data.clamp_(min=0.0)
            for j in range(n): mr.data[j].clamp_(max=mus_init[j])
            v=obj.item()
            if prev is not None and abs(prev-v)<epsilon: break
            prev=v
        sch.step(v)
    with torch.no_grad():
        eff=build_eff_nr_zero_pad(m0,ma,mr,p0,ps); fobj,det=compute_objective_detailed(pi0,eff,lt,a1t,a2t,ma,mr,config,device,dtype)
    return {'mu_add':ma.detach().cpu().numpy(),'mu_remove':mr.detach().cpu().numpy(),'objective':fobj.item(),'eff_nr':eff.detach().cpu().numpy(),'details':det}

def run_greedy(lambdas, mus_init, alpha1, alpha2, config, commit_size=5, buffer_size=None, max_iter=300, lr=1.0, epsilon=1e-2, seed=42, device='cpu', dtype=torch.float32):
    torch.manual_seed(seed); rng=np.random.RandomState(seed); n=len(lambdas)
    p0,ps=config.get_delay_blocks()
    if buffer_size is None: buffer_size=ps
    lt=torch.tensor(lambdas,dtype=dtype,device=device); m0=torch.tensor(mus_init,dtype=dtype,device=device)
    a1t=torch.tensor(alpha1,dtype=dtype,device=device); a2t=torch.tensor(alpha2,dtype=dtype,device=device)
    mac=np.zeros(n); mrc=np.zeros(n); pi=make_pi0(config,device,dtype); el=0
    while el<n:
        ec=min(el+commit_size,n); eo=min(ec+buffer_size,n); wc=ec-el; wo=eo-el
        maw=torch.nn.Parameter(torch.tensor(rng.uniform(0,0.1,wo),dtype=dtype,device=device))
        mrw=torch.nn.Parameter(torch.tensor(rng.uniform(0,0.05,wo),dtype=dtype,device=device))
        o=torch.optim.Adam([maw,mrw],lr=lr); s=torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='min',factor=0.1,patience=15)
        pf=pi.detach().clone(); prev=None
        for step in range(max_iter):
            o.zero_grad(); eff=build_window_eff_nr(el,wo,maw,mrw,mac,mrc,m0,p0,ps,n,device,dtype)
            obj,_=compute_objective_detailed(pf,eff,lt[el:eo],a1t[el:eo],a2t[el:eo],maw,mrw,config,device,dtype); obj.backward(); o.step()
            with torch.no_grad():
                maw.data.clamp_(min=0.0); mrw.data.clamp_(min=0.0)
                for j in range(wo): mrw.data[j].clamp_(max=mus_init[el+j])
                v=obj.item()
                if prev is not None and abs(prev-v)<epsilon: break
                prev=v
            s.step(v)
        with torch.no_grad():
            mac[el:ec]=maw.data[:wc].cpu().numpy(); mrc[el:ec]=mrw.data[:wc].cpu().numpy()
            ef=build_eff_nr_zero_pad(m0,torch.tensor(mac,dtype=dtype,device=device),torch.tensor(mrc,dtype=dtype,device=device),p0,ps)
            pi=propagate_pi(pi,ef[el:ec],lt[el:ec],config,device,dtype)
        el=ec
    with torch.no_grad():
        ef=build_eff_nr_zero_pad(m0,torch.tensor(mac,dtype=dtype,device=device),torch.tensor(mrc,dtype=dtype,device=device),p0,ps)
        fobj,det=compute_objective_detailed(make_pi0(config,device,dtype),ef,lt,a1t,a2t,torch.tensor(mac,dtype=dtype,device=device),torch.tensor(mrc,dtype=dtype,device=device),config,device,dtype)
    return {'mu_add':mac,'mu_remove':mrc,'objective':fobj.item(),'eff_nr':ef.cpu().numpy(),'details':det}

def run_do_nothing(lambdas, mus_init, alpha1, alpha2, config, device='cpu', dtype=torch.float32):
    n=len(lambdas); p0,ps=config.get_delay_blocks(); pi0=make_pi0(config,device,dtype)
    lt=torch.tensor(lambdas,dtype=dtype,device=device); m0=torch.tensor(mus_init,dtype=dtype,device=device)
    a1t=torch.tensor(alpha1,dtype=dtype,device=device); a2t=torch.tensor(alpha2,dtype=dtype,device=device)
    z=torch.zeros(n,dtype=dtype,device=device)
    with torch.no_grad():
        eff=build_eff_nr_zero_pad(m0,z,z,p0,ps); fobj,det=compute_objective_detailed(pi0,eff,lt,a1t,a2t,z,z,config,device,dtype)
    return {'objective':fobj.item(),'details':det,'mu_add':np.zeros(n),'mu_remove':np.zeros(n)}

def run_multi_seed(fn, ns, bs, **kw):
    objs=[]; adds=[]; rems=[]; dets=[]
    for s in range(ns):
        r=fn(seed=bs+s,**kw); objs.append(r['objective']); adds.append(r['mu_add']); rems.append(r['mu_remove'])
        if 'details' in r and r['details']: dets.append(r['details'])
    res={'objectives':np.array(objs),'mu_add':np.array(adds),'mu_remove':np.array(rems),'obj_mean':np.mean(objs),'obj_std':np.std(objs)}
    if dets: res['details_mean']={k:np.mean([d[k] for d in dets]) for k in dets[0]}
    return res

# ================================================================
# STEADY-STATE OPTIMIZER (Analysis 6)
# ================================================================

def compute_ss_dist(Q, cfg, dev, dt, n_iter=500):
    P,_=build_P_from_Q(Q); P=P.coalesce(); Nn=cfg.K_P+cfg.M+1; N=(cfg.K_S+1)*Nn
    pi=torch.ones(N,dtype=dt,device=dev)/N; Pr=P.indices()[0]; Pc=P.indices()[1]; Pv=P.values()
    for _ in range(n_iter):
        pn=torch.zeros_like(pi); pn.index_add_(0,Pc,Pv*pi[Pr]); pi=pn
    return pi

def run_ss_opt(lambdas, mus_init, alpha1, alpha2, config, max_iter=300, lr=1.0, epsilon=1e-2, seed=42, device='cpu', dtype=torch.float32):
    torch.manual_seed(seed); rng=np.random.RandomState(seed); n=len(lambdas)
    p0,ps=config.get_delay_blocks(); KS,KP,M=config.K_S,config.K_P,config.M
    lt=torch.tensor(lambdas,dtype=dtype,device=device); m0=torch.tensor(mus_init,dtype=dtype,device=device)
    a1t=torch.tensor(alpha1,dtype=dtype,device=device); a2t=torch.tensor(alpha2,dtype=dtype,device=device)
    sv=make_state_vectors(KS,KP,M,device=device,dtype=dtype)
    ma=torch.nn.Parameter(torch.tensor(rng.uniform(0,0.1,n),dtype=dtype,device=device))
    mr=torch.nn.Parameter(torch.tensor(rng.uniform(0,0.05,n),dtype=dtype,device=device))
    opt=torch.optim.Adam([ma,mr],lr=lr); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=0.1,patience=15)
    prev=None
    for step in range(max_iter):
        opt.zero_grad(); eff=build_eff_nr_zero_pad(m0,ma,mr,p0,ps)
        obj=torch.tensor(0.0,device=device,dtype=dtype)
        for j in range(n):
            p=lt[j]; c=eff[j]; a1=a1t[j]; a2=a2t[j]; ctl=config.fuel_cost+config.time_to_city*a2; d=config.interval_length
            Q,_,_=build_Q_non_erlang_vec(K_S=KS,K_P=KP,M=M,lam=c,alpha=p,tau=config.tau,device=device,dtype=dtype)
            pss=compute_ss_dist(Q,config,device,dtype)
            Ep=torch.dot(sv['w_pass'],pss); Et=torch.dot(sv['w_pick'],pss); Er=torch.dot(sv['w_stage'],pss)
            Ebp=torch.dot(sv['w_block_pax'],pss); Ebt=torch.dot(sv['w_block_taxi'],pss)
            obj=obj+(a1*Ep+a2*(Et+Er))*d+ma[j]*d*config.cost_per_vehicle_add+mr[j]*d*ctl+config.cost_pax_lost*p*Ebp*d+ctl*c*Ebt*d
        obj.backward(); opt.step()
        with torch.no_grad():
            ma.data.clamp_(min=0.0); mr.data.clamp_(min=0.0)
            for j in range(n): mr.data[j].clamp_(max=mus_init[j])
            v=obj.item()
            if prev is not None and abs(prev-v)<epsilon: break
            prev=v
        sch.step(v)
    return {'mu_add':ma.detach().cpu().numpy(),'mu_remove':mr.detach().cpu().numpy(),'objective':v}

# ================================================================
# ANALYSIS 1: DELAY SENSITIVITY
# ================================================================

def analysis_delay(lambdas, mus_init, alpha1, alpha2, base_config, n_seeds=3, max_iter=300, lr=0.5, epsilon=1e-3, base_seed=42, device='cpu', dtype=torch.float32, out_dir='results/sensitivity'):
    print("\n"+"="*70+"\nANALYSIS 1: DELAY SENSITIVITY\n"+"="*70)
    dnr=[0,5,10,15,20]; dex=[0,5,10,15,20]; results={}
    for dr in dnr:
        for de in dex:
            cfg=copy.deepcopy(base_config); cfg.delay_non_reserved=float(dr); cfg.delay_extra=float(de); p0,ps=cfg.get_delay_blocks()
            k=f"{dr}_{de}"; print(f"  d_d={dr}, d_e_ex={de} (pad={p0},{ps})", end=' ', flush=True)
            fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=cfg,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
            dn=run_do_nothing(lambdas,mus_init,alpha1,alpha2,cfg,device=device,dtype=dtype)
            det=fd.get('details_mean',{})
            results[k]={'fd_mean':fd['obj_mean'],'fd_std':fd['obj_std'],'do_nothing':dn['objective'],'impr_pct':(dn['objective']-fd['obj_mean'])/dn['objective']*100,'pax_wait':det.get('pax_wait',0),'taxi_idle':det.get('taxi_idle',0),'total_mu_add':float(fd['mu_add'].mean(axis=0).sum())}
            print(f"-> FD={fd['obj_mean']:.1f}, impr={results[k]['impr_pct']:.1f}%")
    _save_json(results,out_dir,'delay_sensitivity.json')
    fig,axes=plt.subplots(2,2,figsize=(14,11))
    for idx,(fld,ttl,cm) in enumerate([('fd_mean','Full-Day Cost','YlOrRd'),('impr_pct','Improvement (%)','YlGn'),('pax_wait','Passenger Wait Cost','Reds'),('taxi_idle','Taxi Idle Cost','Blues')]):
        ax=axes[idx//2][idx%2]; mat=np.zeros((len(dnr),len(dex)))
        for i,dr in enumerate(dnr):
            for j,de in enumerate(dex): mat[i,j]=results[f"{dr}_{de}"][fld]
        im=ax.imshow(mat,cmap=cm,aspect='auto'); ax.set_xticks(range(len(dex))); ax.set_xticklabels(dex)
        ax.set_yticks(range(len(dnr))); ax.set_yticklabels(dnr); ax.set_xlabel('$\\delta_e$ extra (min)'); ax.set_ylabel('$\\delta_d$ (min)'); ax.set_title(ttl)
        for i in range(len(dnr)):
            for j in range(len(dex)):
                f=f'{mat[i,j]:.0f}' if 'Cost' in ttl else f'{mat[i,j]:.1f}%'; ax.text(j,i,f,ha='center',va='center',fontsize=7)
        plt.colorbar(im,ax=ax,shrink=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'delay_sensitivity.png'),dpi=150); plt.close()
    print(f"  Saved to {out_dir}/delay_sensitivity.*"); return results

# ================================================================
# ANALYSIS 2: COMMIT SIZE
# ================================================================

def analysis_commit(lambdas, mus_init, alpha1, alpha2, config, n_seeds=3, max_iter=300, lr=0.5, epsilon=1e-3, base_seed=42, device='cpu', dtype=torch.float32, out_dir='results/sensitivity'):
    print("\n"+"="*70+"\nANALYSIS 2: COMMIT SIZE\n"+"="*70)
    n=len(lambdas); ps=config.get_delay_blocks()[1]
    commits=sorted(set([c for c in [3,6,9,12,18,24,36,48,72,n] if c<=n]))
    print(f"  Full-day...",end=' ',flush=True)
    fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=config,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
    dn=run_do_nothing(lambdas,mus_init,alpha1,alpha2,config,device=device,dtype=dtype); print(f"FD={fd['obj_mean']:.2f}")
    results={'full_day':fd['obj_mean'],'fd_std':fd['obj_std'],'do_nothing':dn['objective'],'greedy':{}}
    for c in commits:
        buf=min(ps,n-c) if c<n else 0; print(f"  commit={c}...",end=' ',flush=True)
        gr=run_multi_seed(run_greedy,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=config,commit_size=c,buffer_size=buf,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
        gap=(gr['obj_mean']-fd['obj_mean'])/fd['obj_mean']*100
        results['greedy'][c]={'mean':gr['obj_mean'],'std':gr['obj_std'],'gap_pct':gap}; print(f"obj={gr['obj_mean']:.2f} (gap={gap:+.2f}%)")
    _save_json(results,out_dir,'commit_sensitivity.json')
    fig,axes=plt.subplots(1,2,figsize=(14,5)); ch=[c*config.interval_length/60 for c in commits]
    ms=[results['greedy'][c]['mean'] for c in commits]; ss=[results['greedy'][c]['std'] for c in commits]; gs=[results['greedy'][c]['gap_pct'] for c in commits]
    axes[0].errorbar(ch,ms,yerr=ss,fmt='o-',color='#E8475F',capsize=4,label='Greedy')
    axes[0].axhline(y=fd['obj_mean'],color='#2E86AB',lw=2,label='Full-Day'); axes[0].axhline(y=dn['objective'],color='gray',ls='--',label='Do-Nothing')
    axes[0].set_xlabel('Commit (hours)'); axes[0].set_ylabel('Objective'); axes[0].set_title('Cost vs Horizon'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].bar(range(len(commits)),gs,color='#E8475F',alpha=0.7); axes[1].set_xticks(range(len(commits))); axes[1].set_xticklabels([f'{h:.1f}h' for h in ch],rotation=45)
    axes[1].set_xlabel('Commit'); axes[1].set_ylabel('Gap (%)'); axes[1].set_title('Myopia Cost'); axes[1].grid(True,alpha=0.3,axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'commit_sensitivity.png'),dpi=150); plt.close()
    print(f"  Saved to {out_dir}/commit_sensitivity.*"); return results

# ================================================================
# ANALYSIS 3: VoT & COST
# ================================================================

def analysis_vot_cost(lambdas, mus_init, alpha1, alpha2, base_config, n_seeds=3, max_iter=300, lr=0.5, epsilon=1e-3, base_seed=42, device='cpu', dtype=torch.float32, out_dir='results/sensitivity'):
    print("\n"+"="*70+"\nANALYSIS 3: VoT & COST\n"+"="*70)
    mults=[0.5,0.75,1.0,1.5,2.0]; results={'vot_2d':{},'c_add':{},'c_bp':{}}
    print("  --- VoT 2D (a1 x a2) ---")
    for m1 in mults:
        for m2 in mults:
            a1s=alpha1*m1; a2s=alpha2*m2; k=f"{m1}_{m2}"; print(f"  a1x{m1}, a2x{m2}...",end=' ',flush=True)
            fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=a1s,alpha2=a2s,config=base_config,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
            det=fd.get('details_mean',{})
            results['vot_2d'][k]={'obj':fd['obj_mean'],'mu_add':float(fd['mu_add'].mean(axis=0).sum()),'mu_rem':float(fd['mu_remove'].mean(axis=0).sum()),'pax_wait':det.get('pax_wait',0),'taxi_idle':det.get('taxi_idle',0)}
            print(f"obj={fd['obj_mean']:.1f}")
    print("  --- Dispatch cost ---")
    for m in [0.25,0.5,1.0,2.0,4.0]:
        cfg=copy.deepcopy(base_config); cfg.cost_per_vehicle_add=base_config.cost_per_vehicle_add*m
        print(f"  c_a={cfg.cost_per_vehicle_add:.0f}...",end=' ',flush=True)
        fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=cfg,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
        results['c_add'][str(m)]={'obj':fd['obj_mean'],'mu_add':float(fd['mu_add'].mean(axis=0).sum()),'val':cfg.cost_per_vehicle_add}; print(f"obj={fd['obj_mean']:.1f}")
    print("  --- Blocking penalty ---")
    for m in [0.25,0.5,1.0,2.0,4.0]:
        cfg=copy.deepcopy(base_config); cfg.cost_pax_lost=base_config.cost_pax_lost*m
        print(f"  c_bp={cfg.cost_pax_lost:.0f}...",end=' ',flush=True)
        fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=cfg,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
        results['c_bp'][str(m)]={'obj':fd['obj_mean'],'mu_add':float(fd['mu_add'].mean(axis=0).sum()),'val':cfg.cost_pax_lost}; print(f"obj={fd['obj_mean']:.1f}")
    _save_json(results,out_dir,'vot_cost_sensitivity.json')
    fig=plt.figure(figsize=(18,11)); nm=len(mults)
    for col,(fld,ttl,cm) in enumerate([('obj','Objective','YlOrRd'),('mu_add','Total $\\mu^+$','Purples'),('pax_wait','Passenger Wait Cost','Reds')]):
        ax=fig.add_subplot(2,3,col+1); mat=np.zeros((nm,nm))
        for i,m1 in enumerate(mults):
            for j,m2 in enumerate(mults): mat[i,j]=results['vot_2d'][f"{m1}_{m2}"][fld]
        im=ax.imshow(mat,cmap=cm,aspect='auto'); ax.set_xticks(range(nm)); ax.set_xticklabels([f'{m}x' for m in mults])
        ax.set_yticks(range(nm)); ax.set_yticklabels([f'{m}x' for m in mults])
        ax.set_xlabel('Driver VoT ($\\alpha_2$)'); ax.set_ylabel('Pax VoT ($\\alpha_1$)'); ax.set_title(ttl)
        for i in range(nm):
            for j in range(nm):
                f=f'{mat[i,j]:.0f}' if fld=='obj' else f'{mat[i,j]:.2f}'; ax.text(j,i,f,ha='center',va='center',fontsize=7)
        plt.colorbar(im,ax=ax,shrink=0.8)
    cam=[0.25,0.5,1.0,2.0,4.0]; bpm=cam
    ax=fig.add_subplot(2,3,4); ax.plot(cam,[results['c_add'][str(m)]['obj'] for m in cam],'o-',color='#2E86AB'); ax.set_xscale('log',base=2)
    ax.set_xlabel('$c_a$ mult'); ax.set_ylabel('Objective'); ax.set_title('Obj vs Dispatch Cost'); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(2,3,5); ax.plot(cam,[results['c_add'][str(m)]['mu_add'] for m in cam],'o-',color='#E8475F'); ax.set_xscale('log',base=2)
    ax.set_xlabel('$c_a$ mult'); ax.set_ylabel('$\\mu^+$'); ax.set_title('Dispatch vs Cost'); ax.grid(True,alpha=0.3)
    ax=fig.add_subplot(2,3,6); ax.plot(bpm,[results['c_bp'][str(m)]['mu_add'] for m in bpm],'o-',color='#F5A623'); ax.set_xscale('log',base=2)
    ax.set_xlabel('$c_{bp}$ mult'); ax.set_ylabel('$\\mu^+$'); ax.set_title('Dispatch vs Block Penalty'); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'vot_cost_sensitivity.png'),dpi=150); plt.close()
    print(f"  Saved to {out_dir}/vot_cost_sensitivity.*"); return results

# ================================================================
# ANALYSIS 4: DEMAND SCALING
# ================================================================

def analysis_demand(lambdas, mus_init, alpha1, alpha2, config, n_seeds=3, max_iter=300, lr=0.5, epsilon=1e-3, base_seed=42, device='cpu', dtype=torch.float32, out_dir='results/sensitivity'):
    print("\n"+"="*70+"\nANALYSIS 4: DEMAND SCALING\n"+"="*70)
    scales=[0.5,0.75,1.0,1.25,1.5,2.0]; results={}
    for s in scales:
        ls=lambdas*s; print(f"  lam x{s:.2f}...",end=' ',flush=True)
        fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=ls,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=config,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
        dn=run_do_nothing(ls,mus_init,alpha1,alpha2,config,device=device,dtype=dtype)
        det=fd.get('details_mean',{})
        impr=(dn['objective']-fd['obj_mean'])/dn['objective']*100
        results[str(s)]={'fd_mean':fd['obj_mean'],'fd_std':fd['obj_std'],'do_nothing':dn['objective'],'impr_pct':impr,'mu_add':float(fd['mu_add'].mean(axis=0).sum()),'pax_wait':det.get('pax_wait',0)}
        print(f"FD={fd['obj_mean']:.1f}, impr={impr:.1f}%")
    _save_json(results,out_dir,'demand_sensitivity.json')
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fds=[results[str(s)]['fd_mean'] for s in scales]; fss=[results[str(s)]['fd_std'] for s in scales]
    dns=[results[str(s)]['do_nothing'] for s in scales]; ims=[results[str(s)]['impr_pct'] for s in scales]; ads=[results[str(s)]['mu_add'] for s in scales]
    axes[0].errorbar(scales,fds,yerr=fss,fmt='o-',color='#2E86AB',capsize=4,label='Full-Day'); axes[0].plot(scales,dns,'s--',color='gray',label='Do-Nothing')
    axes[0].set_xlabel('Scale'); axes[0].set_ylabel('Objective'); axes[0].set_title('Cost vs Demand'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].bar(range(len(scales)),ims,color='#2E86AB',alpha=0.7); axes[1].set_xticks(range(len(scales))); axes[1].set_xticklabels([f'{s}' for s in scales])
    axes[1].set_xlabel('Scale'); axes[1].set_ylabel('Improvement (%)'); axes[1].set_title('Value of Optimization'); axes[1].grid(True,alpha=0.3,axis='y')
    axes[2].plot(scales,ads,'o-',color='#E8475F'); axes[2].set_xlabel('Scale'); axes[2].set_ylabel('$\\mu^+$'); axes[2].set_title('Dispatch Volume'); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'demand_sensitivity.png'),dpi=150); plt.close()
    print(f"  Saved to {out_dir}/demand_sensitivity.*"); return results

# ================================================================
# ANALYSIS 5: INTERVAL LENGTH
# ================================================================

def analysis_delta(lambdas, mus_init, alpha1, alpha2, base_config, n_seeds=3, max_iter=300, lr=0.5, epsilon=1e-3, base_seed=42, device='cpu', dtype=torch.float32, out_dir='results/sensitivity'):
    print("\n"+"="*70+"\nANALYSIS 5: INTERVAL LENGTH\n"+"="*70)
    deltas=[1.0,2.0,3.0,5.0,10.0,15.0]; max_min=len(lambdas)*base_config.interval_length; results={}
    for dl in deltas:
        cfg=copy.deepcopy(base_config); cfg.interval_length=dl; cfg.group_size=max(1,int(dl)); ni=int(max_min/dl)
        try:
            pax=pd.read_csv('Datasets/Total_Passengers_Arrival.csv'); taxi=pd.read_csv('Datasets/Total_Passengers_departures.csv')
            pax["Total_Passengers"]=pax["Total_Passengers"]/cfg.data_scale_factor; taxi["Total_Passengers"]=taxi["Total_Passengers"]/cfg.data_scale_factor
            lam=aggregate_passengers(pax,cfg.group_size)['total_rate'].values[:ni]; mus=aggregate_passengers(taxi,cfg.group_size)['total_rate'].values[:ni]
        except:
            r=base_config.interval_length/dl; lam=np.interp(np.arange(ni),np.arange(len(lambdas))*r,lambdas); mus=np.interp(np.arange(ni),np.arange(len(mus_init))*r,mus_init)
        a1=np.full(ni,base_config.alpha1_default); a2=np.repeat(base_config.alpha2_base,max(1,int(60/dl)))[:ni]
        if len(a2)<ni: a2=np.resize(a2,ni)
        p0,ps=cfg.get_delay_blocks(); print(f"  D={dl:.0f}min, n={ni}, pad=({p0},{ps})...",end=' ',flush=True)
        t0=time.time()
        fd=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lam,mus_init=mus,alpha1=a1,alpha2=a2,config=cfg,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
        rt=time.time()-t0; results[str(dl)]={'delta':dl,'n':ni,'obj':fd['obj_mean'],'std':fd['obj_std'],'runtime':rt,'rt_per_seed':rt/n_seeds}
        print(f"obj={fd['obj_mean']:.1f}, time={rt:.1f}s")
    base_obj=results[str(deltas[0])]['obj']
    for k in results: results[k]['pct_err']=(results[k]['obj']-base_obj)/base_obj*100
    _save_json(results,out_dir,'delta_sensitivity.json')
    fig,axes=plt.subplots(1,3,figsize=(16,5)); ds=deltas
    objs=[results[str(d)]['obj'] for d in ds]; stds=[results[str(d)]['std'] for d in ds]
    errs=[results[str(d)]['pct_err'] for d in ds]; rts=[results[str(d)]['rt_per_seed'] for d in ds]
    axes[0].errorbar(ds,objs,yerr=stds,fmt='o-',color='#2E86AB',capsize=4); axes[0].axhline(y=base_obj,color='gray',ls='--',label=f'D={deltas[0]} base')
    axes[0].set_xlabel('D (min)'); axes[0].set_ylabel('Objective'); axes[0].set_title('Objective vs D'); axes[0].legend(); axes[0].grid(True,alpha=0.3)
    axes[1].plot(ds,errs,'o-',color='#E8475F'); axes[1].axhline(y=0,color='black',lw=0.5); axes[1].axhline(y=1,color='gray',ls='--',alpha=0.5,label='1%')
    axes[1].set_xlabel('D (min)'); axes[1].set_ylabel('% Error'); axes[1].set_title('PCA Accuracy'); axes[1].legend(); axes[1].grid(True,alpha=0.3)
    axes[2].plot(ds,rts,'o-',color='#F5A623'); axes[2].set_xlabel('D (min)'); axes[2].set_ylabel('Runtime (s)'); axes[2].set_yscale('log'); axes[2].set_title('Computational Cost'); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'delta_sensitivity.png'),dpi=150); plt.close()
    print(f"  Saved to {out_dir}/delta_sensitivity.*"); return results

# ================================================================
# ANALYSIS 6: SS vs TRANSIENT GAP
# ================================================================

def analysis_ss_gap(lambdas, mus_init, alpha1, alpha2, base_config, n_seeds=3, max_iter=300, lr=0.5, epsilon=1e-3, base_seed=42, device='cpu', dtype=torch.float32, out_dir='results/sensitivity'):
    print("\n"+"="*70+"\nANALYSIS 6: SS vs TRANSIENT GAP\n"+"="*70)
    n=len(lambdas); tds=[0,5,10,15,20,30,40]; results={}
    lt=torch.tensor(lambdas,dtype=dtype,device=device); m0=torch.tensor(mus_init,dtype=dtype,device=device)
    a1t=torch.tensor(alpha1,dtype=dtype,device=device); a2t=torch.tensor(alpha2,dtype=dtype,device=device)
    for td in tds:
        cfg=copy.deepcopy(base_config); cfg.delay_non_reserved=td/3.0; cfg.delay_extra=2.0*td/3.0; p0,ps=cfg.get_delay_blocks()
        print(f"  delay={td}min (d_d={cfg.delay_non_reserved:.1f}, d_e_ex={cfg.delay_extra:.1f})")
        print(f"    Transient...",end=' ',flush=True)
        tr=run_multi_seed(run_full_day,n_seeds,base_seed,lambdas=lambdas,mus_init=mus_init,alpha1=alpha1,alpha2=alpha2,config=cfg,max_iter=max_iter,lr=lr,epsilon=epsilon,device=device,dtype=dtype)
        print(f"obj={tr['obj_mean']:.1f}")
        print(f"    SS opt...",end=' ',flush=True)
        ss_list=[]
        for s in range(n_seeds): ss_list.append(run_ss_opt(lambdas,mus_init,alpha1,alpha2,cfg,max_iter=max_iter,lr=lr,epsilon=epsilon,seed=base_seed+s,device=device,dtype=dtype))
        print(f"done")
        print(f"    SS-on-transient...",end=' ',flush=True)
        sotr_objs=[]; sotr_dets=[]
        for ss in ss_list:
            with torch.no_grad():
                mat=torch.tensor(ss['mu_add'],dtype=dtype,device=device); mrt=torch.tensor(ss['mu_remove'],dtype=dtype,device=device)
                eff=build_eff_nr_zero_pad(m0,mat,mrt,p0,ps); pi0=make_pi0(cfg,device,dtype)
                ov,det=compute_objective_detailed(pi0,eff,lt,a1t,a2t,mat,mrt,cfg,device,dtype)
                sotr_objs.append(ov.item()); sotr_dets.append(det)
        sotr_mean=np.mean(sotr_objs); print(f"obj={sotr_mean:.1f}")
        dn=run_do_nothing(lambdas,mus_init,alpha1,alpha2,cfg,device=device,dtype=dtype)
        gap=(sotr_mean-tr['obj_mean'])/tr['obj_mean']*100
        tr_det=tr.get('details_mean',{})
        ss_det_mean={k:np.mean([d[k] for d in sotr_dets]) for k in sotr_dets[0]} if sotr_dets else {}
        comp_gap={k:ss_det_mean.get(k,0)-tr_det.get(k,0) for k in tr_det} if tr_det else {}
        results[str(td)]={'delay':td,'transient':tr['obj_mean'],'tr_std':tr['obj_std'],'ss_on_tr':sotr_mean,'do_nothing':dn['objective'],'gap_pct':gap,'comp_gap':comp_gap,'tr_det':tr_det,'ss_det':ss_det_mean}
        print(f"    GAP={gap:+.2f}%")
    # Save last-delay control profiles for plot
    last_tr_add=tr['mu_add'].mean(axis=0) if len(tr['mu_add'].shape)>1 else tr['mu_add'][0]
    last_ss_add=ss_list[0]['mu_add']
    _save_json(results,out_dir,'ss_gap_sensitivity.json')
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    tr_o=[results[str(d)]['transient'] for d in tds]; ss_o=[results[str(d)]['ss_on_tr'] for d in tds]; dn_o=[results[str(d)]['do_nothing'] for d in tds]; gps=[results[str(d)]['gap_pct'] for d in tds]
    ax=axes[0][0]; ax.plot(tds,tr_o,'o-',color='#2E86AB',lw=2,label='Transient-optimal'); ax.plot(tds,ss_o,'s-',color='#E8475F',lw=2,label='SS on Transient')
    ax.plot(tds,dn_o,'^--',color='gray',label='Do-Nothing'); ax.fill_between(tds,tr_o,ss_o,alpha=0.15,color='#E8475F')
    ax.set_xlabel('Total Delay (min)'); ax.set_ylabel('Objective'); ax.set_title('Cost Comparison'); ax.legend(); ax.grid(True,alpha=0.3)
    ax=axes[0][1]; ax.plot(tds,gps,'o-',color='#E8475F',lw=2.5,markersize=8); ax.fill_between(tds,0,gps,alpha=0.2,color='#E8475F')
    ax.set_xlabel('Total Delay (min)'); ax.set_ylabel('Gap (%)'); ax.set_title('Price of Ignoring Transience'); ax.grid(True,alpha=0.3)
    ax=axes[1][0]; x=np.arange(len(tds)); cks=['pax_wait','taxi_idle','pax_block','taxi_block','add_cost','remove_cost']
    cls=['Pax Wait','Taxi Idle','Pax Block','Taxi Block','Add','Remove']; ccs=['#E8475F','#2E86AB','#F5A623','#7B68EE','#20B2AA','#CD853F']
    bp=np.zeros(len(tds)); bn=np.zeros(len(tds))
    for ck,cl,cc in zip(cks,cls,ccs):
        vs=[results[str(d)].get('comp_gap',{}).get(ck,0) for d in tds]; p=np.maximum(vs,0); ng=np.minimum(vs,0)
        ax.bar(x,p,bottom=bp,color=cc,alpha=0.7,label=cl); ax.bar(x,ng,bottom=bn,color=cc,alpha=0.7); bp+=p; bn+=ng
    ax.set_xticks(x); ax.set_xticklabels([f'{d}' for d in tds]); ax.set_xlabel('Delay (min)'); ax.set_ylabel('Cost Diff'); ax.set_title('Gap Breakdown'); ax.legend(fontsize=7); ax.grid(True,alpha=0.3,axis='y')
    ax=axes[1][1]; t_ax=np.arange(n)*base_config.interval_length; ax2=ax.twinx(); ax2.fill_between(t_ax,lambdas,alpha=0.1,color='gray'); ax2.set_ylabel('$\\lambda$',color='gray')
    ax.plot(t_ax,last_tr_add,color='#2E86AB',lw=1.5,label='Transient $\\mu^+$'); ax.plot(t_ax,last_ss_add,color='#E8475F',lw=1.5,ls='--',label='SS $\\mu^+$')
    ax.set_xlabel('Time (min)'); ax.set_ylabel('$\\mu^+$'); ax.set_title(f'Controls (delay={tds[-1]}min)'); ax.legend(loc='upper left'); ax.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir,'ss_gap_sensitivity.png'),dpi=150); plt.close()
    print(f"  Saved to {out_dir}/ss_gap_sensitivity.*"); return results

# ================================================================
# UTILITIES
# ================================================================

def _save_json(data, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    def conv(o):
        if isinstance(o,(np.floating,np.integer)): return float(o)
        if isinstance(o,np.ndarray): return o.tolist()
        if isinstance(o,dict): return {str(k):conv(v) for k,v in o.items()}
        if isinstance(o,(list,tuple)): return [conv(v) for v in o]
        return o
    with open(os.path.join(out_dir,filename),'w') as f: json.dump(conv(data),f,indent=2)

# ================================================================
# MAIN
# ================================================================

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Sensitivity Analysis')
    parser.add_argument('--n_intervals',type=int,default=15)
    parser.add_argument('--n_seeds',type=int,default=3)
    parser.add_argument('--max_iter',type=int,default=300)
    parser.add_argument('--lr',type=float,default=0.5)
    parser.add_argument('--epsilon',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--K_S',type=int,default=30)
    parser.add_argument('--K_P',type=int,default=10)
    parser.add_argument('--M',type=int,default=20)
    parser.add_argument('--analysis',type=str,default='all',choices=['all','delay','commit','vot','demand','delta','ss_gap'])
    parser.add_argument('--out_dir',type=str,default='results/sensitivity')
    args=parser.parse_args()

    config=QueueConfig(); config.K_S=args.K_S; config.K_P=args.K_P; config.M=args.M
    lambdas,mus_init=load_default_data(config); lambdas=lambdas[:args.n_intervals]; mus_init=mus_init[:args.n_intervals]
    alpha1,alpha2=config.get_alpha_arrays(size=args.n_intervals); p0,ps=config.get_delay_blocks()

    print("="*70+"\nSENSITIVITY ANALYSIS\n"+"="*70)
    print(f"  Intervals: {args.n_intervals}, Seeds: {args.n_seeds}, States: {(config.K_S+1)*(config.K_P+config.M+1)}")
    print(f"  Delays: pad_mu0={p0}, pad_mus={ps}, Analysis: {args.analysis}")
    print("="*70)

    os.makedirs(args.out_dir,exist_ok=True)
    kw=dict(n_seeds=args.n_seeds,max_iter=args.max_iter,lr=args.lr,epsilon=args.epsilon,base_seed=args.seed,out_dir=args.out_dir)
    t0=time.time()

    if args.analysis in ('all','delay'):  analysis_delay(lambdas,mus_init,alpha1,alpha2,config,**kw)
    if args.analysis in ('all','commit'): analysis_commit(lambdas,mus_init,alpha1,alpha2,config,**kw)
    if args.analysis in ('all','vot'):    analysis_vot_cost(lambdas,mus_init,alpha1,alpha2,config,**kw)
    if args.analysis in ('all','demand'): analysis_demand(lambdas,mus_init,alpha1,alpha2,config,**kw)
    if args.analysis in ('all','delta'):  analysis_delta(lambdas,mus_init,alpha1,alpha2,config,**kw)
    if args.analysis in ('all','ss_gap'): analysis_ss_gap(lambdas,mus_init,alpha1,alpha2,config,**kw)

    print(f"\nTotal: {time.time()-t0:.1f}s. Results in {args.out_dir}/")