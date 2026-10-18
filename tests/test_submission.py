"""Tests for solver recovery, provenance, and extreme quasi-2D virials."""
import csv
import json
import math
import pytest
import torch
from becgpp import default_cfg,make_grid,ground_state,paths
from becgpp.interactions import q2d_virial_kernel
from becgpp.thomasfermi import _invert_local,extract_tf
from becgpp.io import append_csv
from becgpp import solvers
from becgpp.grid import dV
from becgpp.modes import mode_single,mode_sweep

def cfg(**kw):
    c=dict(mode='single',Ngrid=32,L=6,beta2=0.,seed='gaussian',
           save_ckpt=False,save_figs=False,show_inline=False,
           want_lll=False,want_vortices=False,maxit=300,res_tol=1e-5)
    c.update(kw)
    return default_cfg(**c)

@pytest.mark.parametrize('width',[1e-3,1e-5,1e-8])
def test_q2d_virial_thin_limit(width):
    u=torch.tensor([1.,2.],dtype=torch.float64)
    assert torch.allclose(q2d_virial_kernel(u,width),-1/u,rtol=2e-6,atol=1e-12)

def test_q2d_virial_asymptotic_switch():
    from scipy.special import kve
    # Either side of the switch, compare against an independent Bessel library.
    for x in [999.,1001.,2000.]:
        width=.3; u=2*width*math.sqrt(x)
        expected=2*x*(kve(0,x)-kve(1,x))/(math.sqrt(2*math.pi)*width)
        actual=q2d_virial_kernel(torch.tensor([u]),width).item()
        assert abs(actual/expected-1)<2e-10

def test_early_cg_stall_is_recovered():
    c=cfg(Ngrid=64,L=8,beta2=100.,Omega=.8,seed='triangular',
          seed_noise=0.,seed_phase_noise=0.,maxit=4000,res_tol=1e-6,energy_tol=0.)
    _,_,obs=ground_state(c,verbose=False)
    assert obs['resid_rel']<1e-6,obs
    assert obs['converged']

@pytest.mark.parametrize('change',[dict(Ngrid=32.9),dict(L=0),dict(pad=1),
    dict(beta2=float('nan')),dict(Omega=float('inf')),dict(maxit=-1),dict(nseeds=0)])
def test_invalid_numeric_configuration(change):
    with pytest.raises(ValueError): cfg(**change)

@pytest.mark.parametrize('bad',[torch.zeros((32,32)),torch.ones((3,3)),
                               torch.full((32,32),float('nan'))])
def test_bad_initial_states_rejected(bad):
    with pytest.raises(ValueError): ground_state(cfg(),psi0=bad,verbose=False)

def test_grid_is_double_even_after_default_dtype_change():
    before=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        G=make_grid(cfg())
        assert G['R2'].dtype==torch.float64 and G['K2'].dtype==torch.float64
    finally: torch.set_default_dtype(before)

def test_zero_tolerances_support_fixed_iteration_benchmarks():
    _,_,obs=ground_state(cfg(beta2=30.,maxit=5,res_tol=0.,energy_tol=0.),verbose=False)
    assert obs['iters']==5 and not obs['converged']

def test_cubic_quintic_liquid_vacuum_selection():
    # For b2=-4,b3=2 coexistence is -1.5; spinodal is -2.
    n=_invert_local(-1.75,torch.tensor([0.]),-4.,2.)
    assert n.item()==0.,'metastable liquid must not replace vacuum'

def test_trapped_cq_tf_reports_mass_error():
    c=cfg(beta2=-10.,beta3=10.,trap_coeff=50.,Ngrid=64,L=3.,tf_cache=False)
    G=make_grid(c); tf=extract_tf(c,G)
    assert tf['kind']=='local_trapped_cubicquintic'
    actual=abs((tf['n'].sum()*dV(G)).item()-1)
    assert abs(actual-tf['mass_error'])<1e-14
    assert tf['converged']==(actual<1e-7)

def test_nonlocal_cq_tf_does_not_hide_phase_mass_error():
    c=cfg(beta2=-4000.,beta3=4000.,trap_coeff=50.,Ngrid=32,L=3.,
          G_C=1e-8,kernel='newton',tf_cache=False)
    G=make_grid(c); tf=extract_tf(c,G,iters=100)
    actual=abs((tf['n'].sum()*dV(G)).item()-1)
    assert abs(actual-tf['mass_error'])<1e-14
    if actual>=1e-7: assert not tf['converged']
    assert actual>1e-7,'chosen grid should expose a finite phase-occupancy jump'
    # An unconverged relaxed fixed point need not solve the TF equation; its
    # unresolved phase occupancy must remain visible, not be normalized away.
    assert not tf['converged'] and torch.isfinite(tf['n']).all()

def test_tf_cache_distinguishes_iteration_budget():
    c=cfg(beta2=30.,G_C=2.,kernel='newton',tf_cache=True)
    G=make_grid(c)
    a=extract_tf(c,G,iters=1); b=extract_tf(c,G,iters=2)
    assert a is not b

def test_single_csv_appends(tmp_path):
    p=tmp_path/'runs.csv'; append_csv(p,dict(E=1.,rid='a'))
    append_csv(p,dict(E=2.,rid='b',resid=1e-6))
    with p.open(newline='') as stream: rows=list(csv.DictReader(stream))
    assert [r['rid'] for r in rows]==['a','b']
    assert rows[0]['E']=='1.0' and rows[1]['resid']=='1e-06'

def test_sweep_writes_full_json(tmp_path,monkeypatch):
    import becgpp.modes as modes
    paths.configure(str(tmp_path))
    monkeypatch.setattr(modes,'sweep_figs',lambda *a,**k:None)
    rows=mode_sweep(cfg(mode='sweep',sweep_param='beta2',sweep_values=[0.,2.],maxit=20))
    for r in rows:
        record=json.loads((tmp_path/(r['rid']+'.json')).read_text())
        assert record['cfg']['beta2']==r['beta2']
        assert record['diagnostics']['E']==r['E']

def test_multiseed_reports_total_solver_time(monkeypatch):
    calls=[]
    def fake(c,G=None,verbose=False):
        j=len(calls); calls.append(c)
        return torch.ones((2,2)),G,dict(E=float(2-j),walltime=float(j+1))
    monkeypatch.setattr(solvers,'ground_state',fake)
    _,_,obs=solvers.ground_state_multiseed(cfg(nseeds=3),G={})
    assert obs['walltime']==6. and obs['selected_seed_walltime']==3.
