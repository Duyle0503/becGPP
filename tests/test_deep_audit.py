"""Regression tests for the independently reproduced CPC submission defects."""
import math
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.special import k0e
from becgpp import default_cfg, make_grid, ground_state, paths
from becgpp.grid import dV, resample_state
from becgpp.interactions import kernel_convolution, clear_kernel_cache
from becgpp.fields import radial_average
from becgpp.thomasfermi import extract_tf
import becgpp.modes as modes

torch.set_num_threads(2)

@pytest.fixture(autouse=True)
def isolate(tmp_path):
    paths.configure(str(tmp_path))
    yield
    clear_kernel_cache()

def cfg(**kw):
    c = dict(mode='single', Ngrid=48, L=6., beta2=0., seed='gaussian',
             save_ckpt=False, save_figs=False, show_inline=False, maxit=1000,
             res_tol=1e-7, energy_tol=0., want_lll=False, want_vortices=False)
    c.update(kw)
    return default_cfg(**c)

@pytest.mark.parametrize('dim,kernel,lz,N',[
    ('2D','newton',0.,9),('2D','log',0.,9),('quasi2D','q2d',.3,9),
    ('3D','newton',0.,5)])
def test_fft_matches_independent_direct_cell_quadrature(dim,kernel,lz,N):
    """No FFTs or package kernel builder in the reference; density reaches edges."""
    c=cfg(dimension=dim,kernel=kernel,l_z=lz,G_C=1.,Ngrid=N,L=2.)
    G=make_grid(c); d=G['ndim']; dx=G['dx']
    ids=np.indices((N,)*d).reshape(d,-1).T
    delta=(ids[:,None,:]-ids[None,:,:])*dx
    nodes,w=np.polynomial.legendre.leggauss(4)
    K=np.zeros((len(ids),len(ids)))
    for index in np.ndindex(*((4,)*d)):
        shift=np.array([nodes[i] for i in index])*dx/2
        r=np.sqrt(np.sum((delta+shift)**2,axis=-1))
        vals=1/r if kernel=='newton' else (-np.log(r) if kernel=='log' else k0e(r*r/(4*lz*lz))/(np.sqrt(2*np.pi)*lz))
        K+=np.prod([w[i]/2 for i in index])*vals
    if kernel=='newton':
        selfval=(4*np.arcsinh(1) if d==2 else 2.3800773640)/dx
    elif kernel=='log': selfval=-np.log(dx)+1.5+.5*np.log(2)-np.pi/4
    else:
        # Independent tensor integration using scipy adaptive quadrature.
        from scipy.integrate import dblquad
        selfval=4*dblquad(lambda y,x:k0e((x*x+y*y)/(4*lz*lz))/(np.sqrt(2*np.pi)*lz),0,dx/2,lambda x:0,lambda x:dx/2,epsabs=1e-10)[0]/dx**2
    np.fill_diagonal(K,selfval)
    rho=np.random.default_rng(7).random((N,)*d)
    ref=(K@rho.ravel()*dx**d).reshape(rho.shape)
    got=kernel_convolution(torch.tensor(rho,device=G['R2'].device),G).cpu().numpy()
    assert np.max(np.abs(got-ref))/max(1.,np.max(np.abs(ref)))<2e-9

@pytest.mark.parametrize('kernel',['newton','log','q2d'])
def test_padding_2_and_3_agree_for_edge_density(kernel):
    c=cfg(Ngrid=13,kernel=kernel,l_z=.4,G_C=1.)
    G=make_grid(dict(c,pad=2))
    rho=torch.tensor(np.random.default_rng(8).random((13,13)),device=G['R2'].device)
    a=kernel_convolution(rho,G)
    b=kernel_convolution(rho,make_grid(dict(c,pad=3)))
    assert (a-b).abs().max().item()<2e-12

def test_default_config_lists_are_independent():
    a,b=default_cfg(),default_cfg()
    assert a['sweep_values'] is not b['sweep_values'], 'shallow copies share mutable defaults'

def test_python_dispatch_rejects_unknown_keys():
    c=cfg(mode='tf_only',beta2=30.)
    c['omega']=.9
    with pytest.raises(ValueError,match='unknown'):
        modes.run(c)

def test_grid_rejects_fractional_size():
    with pytest.raises(ValueError): make_grid(cfg(Ngrid=32.9))

def test_radial_average_excludes_points_beyond_rmax():
    G=make_grid(cfg(Ngrid=64,L=6.))
    r,p=radial_average(G['R2'],G,bins=10,rmax=2.)
    assert p[-1]<=4., f'last bin {p[-1]} contains samples outside rmax=2'

def test_flattop_tf_satisfies_strong_trap_equation():
    c=cfg(beta2=-10.,beta3=10.,trap_coeff=50.,Ngrid=64,L=3.,tf_cache=False)
    G=make_grid(c); tf=extract_tf(c,G); n=tf['n']; mask=n>.1*n.max()
    mu_field=G['V']+c['beta2']*n+c['beta3']*n*n
    err=(mu_field[mask]-tf['mu']).abs().max().item()
    assert err<.05*max(1.,abs(tf['mu'])), f'claimed converged TF violates Euler equation by {err}'

def test_sweep_resamples_when_box_changes_at_same_N(monkeypatch):
    original=modes.ground_state; calls=[]
    def capture(c,G=None,psi0=None,verbose=False):
        result=original(c,G=G,psi0=psi0,verbose=False)
        calls.append((G,psi0,result[0]))
        return result
    monkeypatch.setattr(modes,'ground_state',capture)
    monkeypatch.setattr(modes,'sweep_figs',lambda *a,**k:None)
    modes.mode_sweep(cfg(mode='sweep',Ngrid=32,maxit=3,sweep_param='L',sweep_values=[6.,12.]))
    oldG,_,oldpsi=calls[0]; newG,passed,_=calls[1]
    expected=resample_state(oldpsi,oldG,newG)
    normalized=passed/torch.sqrt((passed.abs()**2).sum()*dV(newG))
    assert (normalized-expected).abs().max().item()<1e-10

def test_energy_stop_cannot_certify_high_residual():
    c=cfg(beta2=200.,step=1e-8,step_max=1e-8,conv_window=5,energy_tol=1e-9,maxit=30)
    _,_,obs=ground_state(c,verbose=False)
    assert not obs['converged'] or obs['resid_rel']<1e-3, obs

@pytest.mark.parametrize('N',[32,48,64])
def test_nonlinear_schrodinger_newton_reference(N):
    c=cfg(dimension='3D',s=0,beta2=0.,G_C=10.,kernel='newton',L=3.,Ngrid=N,maxit=1500,res_tol=1e-7)
    psi,G,obs=ground_state(c,verbose=False)
    from becgpp.fields import mass_radius
    r99=mass_radius(psi.abs()**2,G,.99)
    print({'SN_N':N,'E_G2':obs['E']/100,'mu_G2':obs['mu']/100,'R99_G':r99*10,'resid':obs['resid_rel'],'virial':obs['virial_rel']})
    assert obs['resid_rel']<1e-6
    assert abs(obs['E']/100/(-.05426)-1)<.03
    assert abs(r99*10/9.946-1)<.03

def test_complex_seed_preserves_double_precision():
    G=make_grid(cfg()); p=torch.exp(-.5*G['R2']).to(torch.complex64)
    result,_,obs=ground_state(cfg(maxit=10),G=G,psi0=p,verbose=False)
    assert result.dtype==torch.complex128
