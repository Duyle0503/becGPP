# Copy this ENTIRE file into ONE Kaggle/Colab Python cell after enabling a GPU.
# No !pip, no second cell, no replacement of the environment's CUDA PyTorch.
PROFILE = "full"            # "quick": reduced grids/budget; not paper evidence
RUN_PAPER = True            # full: original 8 showcase physical grids, all seeds
RUN_METHOD_COMPARISON = True # PCG variants vs ITP at equal energy targets
ONLY_GPU_REGRESSIONS = True # True: rerun only the repaired 97-test CUDA job
CPU_THREADS = 2             # fixed and recorded for the CPU/GPU timing comparison
JOB_TIMEOUT_S = 10800       # per job; a timeout is incomplete, never a pass
SOURCE_DIR = None           # optional offline checkout at the exact pinned commit
ALLOW_CPU_SELFTEST = False  # developer smoke only; NEVER counts as GPU evidence
SOLVER_COMMIT = '75f0a304acf2ff61dceb3c5ac4df10a9ef84b197'

import os,sys,json,time,hashlib,subprocess,tempfile,zipfile,urllib.request
import shutil,threading,queue,platform,statistics,csv
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
if PROFILE not in ('quick','full'):raise ValueError(PROFILE)
if not torch.cuda.is_available() and not ALLOW_CPU_SELFTEST:
    raise RuntimeError('CUDA is unavailable. Enable a GPU in notebook settings, then rerun this cell. CPU fallback is not GPU verification.')
BACKEND='cuda' if torch.cuda.is_available() else 'cpu'
base=Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
ROOT=Path(tempfile.mkdtemp(prefix='becgpp_gpu_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_',dir=base))
print('RESULT DIRECTORY:',ROOT,flush=True)

EXPECTED_MODULE_HASHES = {'becgpp/__init__.py': 'caa194ce3b77a627ff8f698f2087e5c69d39cb8074eb3d3b4fce36114a58c689', 'becgpp/__main__.py': 'fda8a2d68e55cb5447db6e92d4cf53fde893a94d0f5e97d42873c2d35c599d66', 'becgpp/cli.py': '24cccd8fd82ab742dd95991280a46e640b85426c1ddfdcbb1ebec2dff3fa7da0', 'becgpp/config.py': '6c7fa4ce89b86922bd5c89319ef2d848fae16e9ea251044ea7a6d01ca1ffde3f', 'becgpp/constants.py': 'd639c4ae1b9c32ea3e7ba4b69a698fcd4d2ca97dfd5ef563a69f3b16a56c2298', 'becgpp/diagnostics.py': '2e65a9e6350ae5b6bcb11dff94e644f7c1efc72521b6c8be616a7445d3f7caf6', 'becgpp/fields.py': 'dbeae4fb2ec1bd8ca3b901aef0cc8f358d54218ac485b2d7bdc0df8cf8bf9003', 'becgpp/figures.py': '1f91419b33ccbc32ff06cdff769a549b0d97e8dac7dc42fc4a7e87654b2ec333', 'becgpp/grid.py': 'e04aa8e4a16f500b77d77d6128a2de0005259f9472814e7958a313a352aea42f', 'becgpp/interactions.py': '195d1c7bf4e08288c88e485e031c06c2ad87116841ea38c5e83aecb9574565ef', 'becgpp/io.py': 'ce98e62d08962b21ec41d4e6b79d343ea3620a011fd796820491d2326f602d64', 'becgpp/modes.py': '93970f83a19e88d0078c5043731b9680a104c4b352c06fe46bf271f507de4356', 'becgpp/operators.py': 'c38d83bb434d6a5ec43631f94ebfa637fe61a84d65377de42677626fe5bbc794', 'becgpp/paths.py': '44dd2c098986c7da274466124c8cdaed441f3de8f71aff006fdadf0d9ca210c5', 'becgpp/seeds.py': '2a449135c7beb2b643c6c08c4d7c681a244be04c4f92c6094508e1660dd459f5', 'becgpp/solvers.py': 'ff75f523a83b0d56657de3c13c629b05fa15dac2141725526ba35807a50f5f73', 'becgpp/thomasfermi.py': '458f49baa8d810838f67dfcde2a9e0360896c7f76dce9ea31e8b7804732fe8f5', 'becgpp/units.py': 'a910852f2117736e26d7677271d1b99250cc8a1426970d463468ea02b2f19847', 'becgpp/vortex.py': '3e6ed6b6c16f863ddc944ad30415c2c67c310a2c947b66c1cfe63c4493a2dec8', 'tests/test_audit.py': '659fc676cf73cd74c92ffccb21f6070dc54e70de530ea6a0c983bfd5534bff89', 'tests/test_deep_audit.py': 'ea06c1bb8861cbf59d105d3e1cfbe646696adc6001c44ed0d846da876b229682', 'tests/test_kernels.py': '9467ab1e15bc937202022ce8ead7a026ecc9b3611d00d61e2a5498801b3c4b64', 'tests/test_smoke.py': '0ed31ef0b1cb8871d8f1f926428cb54c5c0c07e223068e55a74a6a76876e48e3', 'tests/test_submission.py': '9e26ee10292ad18b127afd541bfda826d6b61b7df104133d3d947aa127d585ff', 'tests/test_validation.py': '6d22d4527f5019d20ef5e69b6006df818ea7634e6a3b4806482b4c1e293f4bfc'}
REFERENCE_CASES = [('harmonic2d', {'beta2': 100.0}), ('quartic2d', {'beta2': 30.0, 's': 4, 'trap_coeff': 0.25}), ('quintic2d', {'beta2': 0.0, 'beta3': 50.0}), ('newton2d', {'beta2': 30.0, 'G_C': 3.0, 'kernel': 'newton'}), ('log2d', {'beta2': 30.0, 'G_C': 3.0, 'kernel': 'log'}), ('q2d', {'dimension': 'quasi2D', 'beta2': 30.0, 'G_C': 3.0, 'kernel': 'q2d', 'l_z': 0.4}), ('cq_free', {'s': 0, 'beta2': -50.0, 'beta3': 50.0, 'L': 5.0}), ('harmonic3d', {'dimension': '3D', 'Ngrid': 24, 'beta2': 50.0}), ('sn3d', {'dimension': '3D', 'Ngrid': 32, 'L': 3.0, 's': 0, 'beta2': 0.0, 'G_C': 10.0, 'kernel': 'newton'}), ('rotating2d', {'beta2': 100.0, 'Omega': 0.8, 'L': 8.0, 'Ngrid': 64, 'seed': 'triangular'})]
PAPER_CASES = [('2D_cubicquintic_flattop', {'dimension': '2D', 's': 2, 'trap_coeff': 0.5, 'Omega': 1.0, 'beta2': -250, 'beta3': 250, 'G_C': 0, 'kernel': 'none', 'L': 6, 'Ngrid': 384, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'tf', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 1, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('2D_gravitylike_droplet', {'dimension': '2D', 's': 2, 'trap_coeff': 0.5, 'Omega': 1.0, 'beta2': 100, 'beta3': 0, 'G_C': 10, 'kernel': 'newton', 'L': 10, 'Ngrid': 384, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'tf', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 1, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('2D_harmonic_repulsive', {'dimension': '2D', 's': 2, 'trap_coeff': 0.5, 'Omega': 0.0, 'beta2': 200, 'beta3': 0, 'G_C': 0, 'kernel': 'none', 'L': 10, 'Ngrid': 256, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'tf', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 1, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('2D_quartic_trap', {'dimension': '2D', 's': 4, 'trap_coeff': 0.25, 'Omega': 0.0, 'beta2': 200, 'beta3': 0, 'G_C': 0, 'kernel': 'none', 'L': 8, 'Ngrid': 256, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'tf', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 1, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('2D_rotating_lattice', {'dimension': '2D', 's': 2, 'trap_coeff': 0.5, 'Omega': 0.9, 'beta2': 200, 'beta3': 0, 'G_C': 0, 'kernel': 'none', 'L': 12, 'Ngrid': 384, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'triangular', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 3, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('3D_newton_bosonstar', {'dimension': '3D', 's': 2, 'trap_coeff': 0.5, 'Omega': 0.0, 'beta2': 100, 'beta3': 0, 'G_C': 20, 'kernel': 'newton', 'L': 8, 'Ngrid': 128, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'tf', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 1, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('3D_rotating_oblate', {'dimension': '3D', 's': 2, 'trap_coeff': 0.5, 'Omega': 0.9, 'beta2': 100, 'beta3': 0, 'G_C': 5, 'kernel': 'newton', 'L': 10, 'Ngrid': 128, 'pad': 2, 'maxit': 60000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'triangular', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 3, 'conv_window': 2000, 'precond_shift_min': 1.0}), ('quasi2D_newton', {'dimension': 'quasi2D', 's': 2, 'trap_coeff': 0.5, 'Omega': 0.0, 'beta2': 100, 'beta3': 0, 'G_C': 5, 'kernel': 'newton', 'L': 10, 'Ngrid': 256, 'pad': 2, 'maxit': 40000, 'res_tol': 0.0001, 'energy_tol': 1e-08, 'step': 0.5, 'step_max': 2.0, 'linesearch_max': 12, 'seed': 'tf', 'seed_winding': 0, 'seed_noise': 0.01, 'seed_phase_noise': 0.05, 'rng': 0, 'nseeds': 1, 'conv_window': 2000, 'precond_shift_min': 1.0})]
WORKER = r'''
"""Worker embedded into the single-cell runner; do not run directly."""
import os,sys,json,time,math,traceback,hashlib,platform,csv
from pathlib import Path
import numpy as np
import scipy
from scipy.special import k0e
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import torch

JOB=json.loads(Path(sys.argv[1]).read_text())
OUT=Path(JOB['out']);OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,JOB['source'])
torch.set_num_threads(JOB.get('cpu_threads',2))
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True,warn_only=True)
from becgpp import default_cfg,make_grid,make_seed,ground_state,DEV,__version__,paths
from becgpp import mode_validate,diagnostics
from becgpp.grid import dV,_memory_estimate_gb
from becgpp.interactions import clear_kernel_cache,kernel_convolution,long_range_phi
from becgpp.operators import energy_components,apply_H,compute_residual
paths.configure(str(OUT/'diagnostics'))

def build_reference(c):
    N=c['Ngrid']; d=3 if c['dimension']=='3D' else 2; dx=2*c['L']/N; dv=dx**d
    xx=(np.arange(N)-N//2)*dx
    coords=np.meshgrid(*([xx]*d),indexing='ij'); R2=sum(x*x for x in coords)
    kk=2*np.pi*np.fft.fftfreq(N,dx); ks=np.meshgrid(*([kk]*d),indexing='ij'); K2=sum(k*k for k in ks)
    V=c['trap_coeff']*R2**(c['s']/2) if c['s']>0 else np.zeros_like(R2)
    ker=None
    if c['G_C']!=0 and c['kernel']!='none':
        jj=np.arange(-(N-1),N)*dx; ds=np.meshgrid(*([jj]*d),indexing='ij')
        ker=np.zeros_like(ds[0]); nd,wt=np.polynomial.legendre.leggauss(4)
        for ids in np.ndindex(*((4,)*d)):
            r2=sum((v+dx/2*nd[j])**2 for v,j in zip(ds,ids))
            value=1/np.sqrt(r2) if c['kernel']=='newton' else (-.5*np.log(r2) if c['kernel']=='log' else k0e(r2/(4*c['l_z']**2))/(np.sqrt(2*np.pi)*c['l_z']))
            ker+=np.prod([wt[j]/2 for j in ids])*value
        if c['kernel']=='newton': val=(4*np.arcsinh(1) if d==2 else 2.3800773640)/dx
        elif c['kernel']=='log': val=-np.log(dx)+1.5+.5*np.log(2)-np.pi/4
        else:
            from scipy.integrate import dblquad
            l=c['l_z']
            val=4*dblquad(lambda y,x:k0e((x*x+y*y)/(4*l*l))/(np.sqrt(2*np.pi)*l),0,dx/2,lambda x:0,lambda x:dx/2,epsabs=1e-10)[0]/dx**2
        ker[(N-1,)*d]=val
    shape=(N,)*d; size=N**d
    def energy_gradient(v):
        raw=(v[:size]+1j*v[size:]).reshape(shape)
        norm=np.sqrt(np.sum(abs(raw)**2)*dv); p=raw/norm; rho=abs(p)**2
        fp=np.fft.fftn(p); kin=np.fft.ifftn(.5*K2*fp)
        lz=-1j*(coords[0]*np.fft.ifftn(1j*ks[1]*fp)-coords[1]*np.fft.ifftn(1j*ks[0]*fp))
        phi=-c['G_C']*fftconvolve(rho,ker,mode='same')*dv if ker is not None else 0.
        E=(np.sum((p.conj()*kin).real)+np.sum(V*rho)+.5*c['beta2']*np.sum(rho**2)+(c['beta3']/3)*np.sum(rho**3)-c['Omega']*np.sum((p.conj()*lz).real)+.5*np.sum(phi*rho))*dv
        Hp=kin+(V+c['beta2']*rho+c['beta3']*rho**2+phi)*p-c['Omega']*lz
        mu=np.sum((p.conj()*Hp).real)*dv; grad=(Hp-mu*p)/norm
        return float(E),2*dv*np.concatenate([grad.real.ravel(),grad.imag.ravel()])
    energy_gradient.reference_kernel=ker
    return energy_gradient,dv

def write(name,value):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    target=OUT/name;tmp=target.with_suffix(target.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False));tmp.replace(target)

def sync():
    if DEV=='cuda':torch.cuda.synchronize()

class ITP:
    """Normalized gradient flow, Strang-split; rotation by x/y alternating directions.
    Written independently of the becGPP solver (only the long-range potential of
    case D reuses becGPP's free-space convolution)."""

    def __init__(self, cfg, G):
        self.cfg, self.G = cfg, G
        self.b2, self.b3 = float(cfg["beta2"]), float(cfg["beta3"])
        self.O, self.gc = float(cfg["Omega"]), float(cfg["G_C"])
        self.ndim = G["ndim"]
        N, dx = G["N"], G["dx"]
        self.k = 2 * math.pi * torch.fft.fftfreq(N, d=dx).to(DEV)
        self.X, self.Y = G["coords"][0], G["coords"][1]
        self.dv = dx ** self.ndim

    def _nonlin(self, psi, phi=None):
        """Local potential; the long-range part is convolved only when phi is None."""
        rho = psi.abs() ** 2
        V = self.G["V"] + self.b2 * rho + self.b3 * rho ** 2
        if self.gc != 0.0 and self.G["kernel"] != "none":
            if phi is None:
                phi = long_range_phi(rho, self.G, self.gc)
            V = V + phi
        return V, phi

    def _kin(self, psi, dt):
        k = self.k
        if self.O == 0.0:
            return torch.fft.ifftn(torch.exp(-0.5 * dt * self.G["K2"]) * torch.fft.fftn(psi))
        sh = [1] * self.ndim
        shx = list(sh); shx[0] = -1
        shy = list(sh); shy[1] = -1
        kx, ky = k.reshape(shx), k.reshape(shy)
        ex = torch.exp(-0.5 * dt * (0.5 * kx ** 2 + self.O * self.Y * kx))   # half step of H_x
        ey = torch.exp(-dt * (0.5 * ky ** 2 - self.O * self.X * ky))        # full step of H_y
        psi = torch.fft.ifft(ex * torch.fft.fft(psi, dim=0), dim=0)
        psi = torch.fft.ifft(ey * torch.fft.fft(psi, dim=1), dim=1)
        psi = torch.fft.ifft(ex * torch.fft.fft(psi, dim=0), dim=0)
        if self.ndim == 3:
            shz = list(sh); shz[2] = -1
            kz = k.reshape(shz)
            psi = torch.fft.ifft(torch.exp(-0.5 * dt * kz ** 2) * torch.fft.fft(psi, dim=2), dim=2)
        return psi

    def _normalize(self, psi):
        return psi / torch.sqrt((psi.abs() ** 2).sum() * self.dv)

    def energy(self, psi):
        return energy_components(psi, self.G, self.b2, self.b3, self.O, self.gc)["E"].item()

    def run(self, psi, dts, E_ref=None, stop_tol=None, budget=float("inf"),
            etol=1e-11, check=25, maxsteps=40000):
        """Returns final psi, list of stage-final energies, and the history
        [(wall time, step, E)] sampled every `check` steps (timer excludes nothing:
        the energy evaluations are part of the method's cost, as for becGPP)."""
        hist = [(0.0, 0, self.energy(psi))]
        sync()
        t0 = time.time()
        steps = 0
        E_stage = []
        stop = "stages_done"
        for dt in dts:
            E_old = hist[-1][2]
            n = 0
            while n < maxsteps:
                for _ in range(check):
                    # symmetric split step with the potential V(psi_n) held over the
                    # step: one long-range convolution per step, and a fixed point
                    # with a smaller dt-bias than re-evaluating V at the half step
                    V, _ = self._nonlin(psi)
                    psi = self._kin(psi * torch.exp(-0.5 * dt * V), dt)
                    psi = self._normalize(psi * torch.exp(-0.5 * dt * V))
                n += check
                steps += check
                E = self.energy(psi)                    # .item() synchronizes
                t = time.time() - t0
                hist.append((t, steps, E))
                if E_ref is not None and stop_tol is not None and abs(E - E_ref) <= stop_tol * abs(E_ref):
                    stop = "reached_stop_tol"
                    break
                if t > budget:
                    stop = "budget"
                    break
                if abs(E - E_old) <= etol * max(1.0, abs(E)):
                    break
                E_old = E
            E_stage.append(hist[-1][2])
            print(f"      ITP dt={dt:g}: E={hist[-1][2]:.12f} steps={steps} t={hist[-1][0]:.0f}s", flush=True)
            if stop != "stages_done":
                break
        return psi, E_stage, hist, stop

def config(**kw):
    base=dict(mode='single',dimension='2D',Ngrid=48,L=6.,beta2=0.,beta3=0.,
              kernel='none',s=2,Omega=0.,seed='gaussian',seed_noise=0.,
              seed_phase_noise=0.,maxit=4000,res_tol=1e-6,energy_tol=0.,
              save_figs=False,save_ckpt=False,show_inline=False,
              want_lll=False,want_vortices=False,record_trace=True,check=1000)
    base.update(kw);return default_cfg(**base)

def grid(c):
    dim=3 if c['dimension'].lower()=='3d' else 2
    if DEV=='cuda':
        free,total=torch.cuda.mem_get_info()
        est=_memory_estimate_gb(dim,c['Ngrid'],c['pad'],c['kernel'])
        if est*1e9>free*.75:
            raise MemoryError(f'Estimated {est:.2f} GB exceeds 75% of free VRAM {free/1e9:.2f} GB; grid was not reduced.')
    return make_grid(c)

def solve(c,initial=None):
    G=grid(c)
    if initial:
        p0=torch.as_tensor(np.load(initial)['psi'],device=DEV,dtype=torch.complex128)
    else:p0=make_seed(G,c)
    p0=p0/torch.sqrt((p0.abs()**2).sum()*dV(G))
    np.savez_compressed(OUT/'initial.npz',psi=p0.detach().cpu().numpy())
    sync();started=time.perf_counter()
    p,G,o=ground_state(c,G=G,psi0=p0,verbose=True)
    sync();elapsed=time.perf_counter()-started
    assert p.device.type==DEV and p.dtype==torch.complex128
    assert G['R2'].dtype==torch.float64 and G['R2'].device.type==DEV
    dv=dV(G);mass=float((p.abs()**2).sum().item()*dv)
    assert torch.isfinite(p).all().item() and abs(mass-1)<1e-10
    trace=o.pop('trace',[])
    with (OUT/'trace.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seconds','iteration','energy','relative_residual']);w.writerows(trace)
    violations=[i for i in range(1,len(trace)) if trace[i][2]>trace[i-1][2]+1e-12*max(1,abs(trace[i-1][2]))]
    assert not violations,f'Accepted energies increased at {violations[:3]}'
    mu_from_H=float((p.conj()*apply_H(p,G,c)).real.sum().item()*dv)
    assert abs(mu_from_H-o['mu'])<1e-9*max(1,abs(o['mu']))
    rho=p.abs()**2
    edge=torch.zeros_like(rho,dtype=torch.bool)
    for axis in range(G['ndim']):
        sl=[slice(None)]*G['ndim'];sl[axis]=slice(0,2);edge[tuple(sl)]=True
        sl[axis]=slice(-2,None);edge[tuple(sl)]=True
    edge_mass=float(rho[edge].sum().item()*dv)
    d=diagnostics(p,G,c) if JOB.get('diagnostics',False) else {}
    if JOB.get('diagnostics',False):
        from becgpp.thomasfermi import extract_tf
        tf=extract_tf(c,G)
        if tf:
            d['tf_metadata']={k:v for k,v in tf.items() if k!='n'}
            np.savez_compressed(OUT/'tf_density.npz',density=tf['n'].detach().cpu().numpy())
    np.savez_compressed(OUT/'state.npz',psi=p.detach().cpu().numpy())
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a=rho.detach().cpu().numpy();sl=a if a.ndim==2 else a[:,:,a.shape[2]//2]
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(sl.T,origin='lower',extent=[-c['L'],c['L']]*2)
    ax.set(xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,label='Density');fig.tight_layout();fig.savefig(OUT/'density.png',dpi=140);plt.close(fig)
    if a.ndim==3:
        m=a.shape[0]//2;fig,axes=plt.subplots(1,3,figsize=(12,3.4))
        for ax,image,label in zip(axes,[a[:,:,m],a[:,m,:],a[m,:,:]],['z = 0','y = 0','x = 0']):
            im=ax.imshow(image.T,origin='lower',extent=[-c['L'],c['L']]*2);ax.set_title(label);fig.colorbar(im,ax=ax)
        fig.tight_layout();fig.savefig(OUT/'density_slices.png',dpi=160);plt.close(fig)
    phase=np.angle(p.detach().cpu().numpy());phase=phase if phase.ndim==2 else phase[:,:,phase.shape[2]//2]
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(phase.T,origin='lower',extent=[-c['L'],c['L']]*2,cmap='twilight',vmin=-np.pi,vmax=np.pi)
    ax.set(xlabel='x',ylabel='y');fig.colorbar(im,ax=ax,label='Phase');fig.tight_layout();fig.savefig(OUT/'phase.png',dpi=140);plt.close(fig)
    checks=dict(mass_error=abs(mass-1),energy_monotone=True,mu_identity_error=abs(mu_from_H-o['mu']),
                strict_residual_pass=o['resid_rel']<c['res_tol'],edge_mass=edge_mass,
                boundary_pass=edge_mass<1e-5,virial_pass=o['virial_rel']<5e-3)
    result=dict(cfg=c,observables=o,diagnostics=d,checks=checks,solve_seconds=elapsed,
                actual_device=str(p.device),dtype=str(p.dtype))
    write('solve.json',result)
    return result

def execute():
    kind=JOB['kind']
    if kind=='pytest':
        import pytest,xml.etree.ElementTree as ET
        args=[str(Path(JOB['source'])/'tests'),'-q','--junitxml='+str(OUT/'pytest.xml')]
        rc=pytest.main(args);root=ET.parse(OUT/'pytest.xml').getroot();s=root.find('testsuite')
        stats=dict(s.attrib)
        expected=97
        passed=rc==0 and int(stats['tests'])==expected and all(int(stats.get(k,0))==0 for k in ['failures','errors','skipped'])
        return dict(status='PASS' if passed else 'FAIL',stats=stats,expected_tests=expected,
                    excluded_gpu_reference_cases=0)
    if kind=='analytic':
        result=mode_validate(default_cfg(validate_N=JOB['N'],show_inline=False))
        write('analytic.json',result)
        return dict(status='PASS' if result['passed'] else 'FAIL',metrics=result)
    if kind=='kernel':
        rows=[]
        for dim,kernel,lz,N in [('2D','newton',0.,9),('2D','log',0.,9),('quasi2D','q2d',.3,9),('3D','newton',0.,5)]:
            c=config(dimension=dim,kernel=kernel,l_z=lz,G_C=1.,Ngrid=N,L=2.)
            # The independent functional's linear convolution builds its own
            # quadrature kernel; compare its gravitational energy to CUDA FFT.
            ef,dv=build_reference(c);G=grid(c)
            raw=np.random.default_rng(7).normal(size=(N,)*G['ndim'])+0j
            p=torch.as_tensor(raw,device=DEV,dtype=torch.complex128);p/=torch.sqrt((p.abs()**2).sum()*dv)
            ref=ef(np.r_[raw.real.ravel(),raw.imag.ravel()])[0]
            got=energy_components(p,G,c['beta2'],c['beta3'],c['Omega'],c['G_C'])['E'].item()
            rho=p.abs()**2
            a=kernel_convolution(rho,G)
            b=kernel_convolution(rho,grid(dict(c,pad=3)))
            # Direct sums of the independently integrated kernel, no FFT in
            # this field reference. Every cell, including edges, contributes.
            ids=np.indices((N,)*G['ndim']).reshape(G['ndim'],-1).T
            offsets=ids[:,None,:]-ids[None,:,:]+N-1
            matrix=ef.reference_kernel[tuple(offsets[:,:,i] for i in range(G['ndim']))]
            direct=(matrix@rho.detach().cpu().numpy().ravel()*dv).reshape(rho.shape)
            field_error=float(np.max(abs(a.detach().cpu().numpy()-direct))/max(1.,np.max(abs(direct))))
            delta=abs(got-ref)/max(1,abs(ref));paderr=(a-b).abs().max().item()
            rows.append(dict(dimension=dim,kernel=kernel,energy_relative_error=delta,field_relative_error=field_error,padding_max_error=paderr,
                             actual_device=str(a.device),passed=delta<2e-9 and field_error<2e-9 and paderr<2e-11))
            clear_kernel_cache()
        from becgpp.interactions import q2d_virial_kernel
        u=torch.tensor([1.,2.],device=DEV,dtype=torch.float64)
        thin=float((q2d_virial_kernel(u,1e-8)+1/u).abs().max().item())
        write('kernels.json',dict(cases=rows,thin_virial_error=thin))
        return dict(status='PASS' if all(x['passed'] for x in rows) and thin<1e-10 else 'FAIL',cases=rows,thin_virial_error=thin)
    if kind=='solve':
        r=solve(config(**JOB['cfg']),JOB.get('initial'))
        checks=r['checks']
        status='PASS' if checks['strict_residual_pass'] and checks['boundary_pass'] and checks['virial_pass'] else 'REVIEW'
        return dict(status=status,**r)
    if kind=='reference':
        c=config(**JOB['cfg']);r=solve(c);a=np.load(OUT/'initial.npz')['psi'];p=np.load(OUT/'state.npz')['psi']
        ef,dv=build_reference(c)
        vec=lambda x:np.r_[x.real.ravel(),x.imag.ravel()]
        initial=vec(a);got=vec(p);e,g=ef(initial)
        direction=np.random.default_rng(23).normal(size=len(initial));direction/=np.linalg.norm(direction)
        eps=1e-5;fd=(ef(initial+eps*direction)[0]-ef(initial-eps*direction)[0])/(2*eps)
        fd_error=abs(fd-np.dot(g,direction))/max(1,abs(fd))
        opt=minimize(ef,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=4000,ftol=2e-15,gtol=1e-10,maxls=40,maxcor=12))
        E=r['observables']['E'];delta=abs(E-opt.fun)/max(1,abs(opt.fun));same=abs(E-ef(got)[0])/max(1,abs(E))
        metrics=dict(energy_difference=delta,functional_difference=same,gradient_fd_error=fd_error,
                     lbfgs_success=bool(opt.success),lbfgs_message=str(opt.message),lbfgs_iterations=int(opt.nit),
                     energy_pass=delta<2e-7 and same<1e-9 and fd_error<1e-7,
                     strict_residual_pass=r['checks']['strict_residual_pass'])
        write('reference.json',metrics)
        # Energy agreement is separate from a tighter requested residual.
        return dict(status=('PASS' if metrics['strict_residual_pass'] else 'REVIEW') if metrics['energy_pass'] else 'FAIL',**r,reference=metrics)
    if kind=='timing':
        c=config(**JOB['cfg']);c.update(res_tol=0.,energy_tol=0.,record_trace=False,maxit=JOB['iterations'])
        G=grid(c);p0=make_seed(G,c);p0/=torch.sqrt((p0.abs()**2).sum()*dV(G))
        # Prime FFT plans, kernels and the actual solver, then restart from the
        # identical seed for every timed repeat. Include energy line searches.
        ground_state(dict(c,maxit=3),G=G,psi0=p0,verbose=False);sync()
        rows=[]
        for repeat in range(JOB['repeats']):
            sync();start=time.perf_counter();p,_,o=ground_state(c,G=G,psi0=p0,verbose=False);sync()
            rows.append(dict(repeat=repeat,seconds=time.perf_counter()-start,iters=o['iters'],E=o['E'],resid=o['resid_rel'],stop=o['stop_reason']))
        full=all(x['iters']==JOB['iterations'] and x['stop']=='maxit' for x in rows)
        return dict(status='PASS' if full else 'REVIEW',cfg=c,records=rows,fixed_iteration_complete=full,
                    median_seconds=float(np.median([x['seconds'] for x in rows])),
                    timing_scope='Warm kernel/FFT/solver; fresh identical seed; synchronized solver call; grid construction excluded.')
    if kind=='methods':
        c=config(**JOB['cfg']);G=grid(c);p0=make_seed(G,c)
        p0=p0/torch.sqrt((p0.abs()**2).sum()*dV(G));np.savez_compressed(OUT/'initial.npz',psi=p0.cpu().numpy())
        ground_state(dict(c,maxit=3),G=G,psi0=p0,verbose=False);sync()
        methods={};states={}
        for label,beta in [('PCG_pp','pr_precond'),('PCG_pr','pr')]:
            sync();t=time.perf_counter();p,_,o=ground_state(dict(c,cg_beta=beta),G=G,psi0=p0,verbose=True);sync()
            assert p.device.type==DEV and torch.isfinite(p).all().item()
            methods[label]=dict(energy=o['E'],residual=o['resid_rel'],stop=o['stop_reason'],seconds=time.perf_counter()-t,history=o.pop('trace'))
            states[label]=p;np.savez_compressed(OUT/(label+'.npz'),psi=p.detach().cpu().numpy())
        reference_before=min(x['energy'] for x in methods.values())
        itp=ITP(c,G);itp.run(p0.clone(),[.02],maxsteps=3,check=1);sync()
        sync();t=time.perf_counter()
        p,stages,hist,stop=itp.run(p0.clone(),[.02,.005,.00125,.0003125],E_ref=reference_before,
            stop_tol=1e-7,budget=JOB['itp_budget_s'],maxsteps=JOB['itp_maxsteps'],check=25)
        sync();resid=compute_residual(p,G,c)[2]
        assert p.device.type==DEV and torch.isfinite(p).all().item()
        assert abs((p.abs()**2).sum().item()*dV(G)-1)<1e-10
        methods['ITP']=dict(energy=itp.energy(p),residual=resid,stop=stop,seconds=time.perf_counter()-t,history=hist,stage_energies=stages)
        np.savez_compressed(OUT/'ITP.npz',psi=p.detach().cpu().numpy())
        chosen=min(methods,key=lambda k:methods[k]['energy']);reference=methods[chosen]['energy'];scale=max(1,abs(reference))
        for label,data in methods.items():
            data['time_to_relative_energy']={str(tol):next((row[0] for row in data['history'] if abs(row[2]-reference)/scale<tol),None) for tol in [1e-4,1e-5,1e-6]}
        with (OUT/'method_traces.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['method','seconds','iteration','energy','relative_residual_if_available'])
            for label,data in methods.items():
                for row in data['history']:w.writerow([label,*row,*([None] if len(row)==3 else [])])
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        for label,data in methods.items():
            ax.semilogy([r[0] for r in data['history']],[max(abs(r[2]-reference)/scale,1e-18) for r in data['history']],label=label)
        ax.set(xlabel='Synchronized elapsed time (s)',ylabel='Relative energy difference');ax.legend();fig.tight_layout();fig.savefig(OUT/'time_to_accuracy.png',dpi=160);plt.close(fig)
        attained=all(data['time_to_relative_energy']['1e-06'] is not None for data in methods.values())
        good_reference=methods[chosen]['residual']<1e-6
        return dict(status='PASS' if attained and good_reference else 'REVIEW',cfg=c,methods=methods,
            reference_energy=reference,reference_source=chosen,reference_residual=methods[chosen]['residual'],
            note='Lowest returned candidate energy, not a proof of global ground state. ITP integration is independent; its convolution and diagnostic energy reuse becGPP operators.')
    raise ValueError(kind)

started=time.perf_counter()
try:
    if JOB['backend']=='cuda':
        assert torch.cuda.is_available() and DEV=='cuda','CUDA required; CPU fallback is forbidden'
        torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    else:assert DEV=='cpu','CPU worker must start with CUDA_VISIBLE_DEVICES empty'
    result=execute()
except KeyboardInterrupt:
    result=dict(status='INTERRUPTED',error='User interrupted this job')
except (MemoryError,torch.cuda.OutOfMemoryError) as exc:
    result=dict(status='BLOCKED_MEMORY',error=str(exc),traceback=traceback.format_exc())
except Exception as exc:
    result=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
result.update(job=JOB,backend=DEV,package_version=__version__,seconds=time.perf_counter()-started,
              environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()))
if DEV=='cuda':
    result['cuda']=dict(name=torch.cuda.get_device_name(0),runtime=torch.version.cuda,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
write('result.json',result)
print('RESULT',JOB['name'],result['status'],flush=True)
sys.exit(0 if result['status'] in ('PASS','REVIEW') else 1)

'''


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def source_hashes(source):
    files=sorted((source/'becgpp').glob('*.py'))+sorted((source/'tests').glob('*.py'))
    return {p.relative_to(source).as_posix():hashlib.sha256(p.read_text(encoding='utf8').replace('\r\n','\n').encode()).hexdigest()
            for p in files}

if SOURCE_DIR is None:
    archive=ROOT/'pinned_source.zip'
    url=f'https://codeload.github.com/Duyle0503/becGPP/zip/{SOLVER_COMMIT}'
    with urllib.request.urlopen(url,timeout=60) as response,archive.open('wb') as f:
        shutil.copyfileobj(response,f)
    extracted=ROOT/'source';extracted.mkdir()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(extracted/item.filename).resolve()
            if not target.is_relative_to(extracted.resolve()):raise ValueError('Unsafe source archive path')
        z.extractall(extracted)
    SOURCE=next(p for p in extracted.iterdir() if p.is_dir() and (p/'becgpp').is_dir())
else:SOURCE=Path(SOURCE_DIR).expanduser().resolve()
actual=source_hashes(SOURCE)
if actual!=EXPECTED_MODULE_HASHES:
    raise RuntimeError('Source module hashes differ from the verified pinned solver. Refusing mixed-version verification.')
dump(ROOT/'source_manifest.json',dict(commit=SOLVER_COMMIT,source=str(SOURCE),normalized_module_sha256=actual))

# Install only missing optional tools; keep the notebook's working CUDA build.
import importlib.util
needed=[name for name in ['numpy','scipy','matplotlib','pytest','yaml'] if importlib.util.find_spec(name) is None]
if needed:
    names=['PyYAML' if n=='yaml' else n for n in needed]
    subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',*names],check=True)
env_info=dict(created_utc=datetime.now(timezone.utc).isoformat(),profile=PROFILE,solver_commit=SOLVER_COMMIT,
              python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda_runtime=torch.version.cuda,
              gpu_evidence=BACKEND=='cuda',cpu_threads=CPU_THREADS,
              gpu_name=torch.cuda.get_device_name(0) if BACKEND=='cuda' else None)
env_info['cpu_model']=platform.processor()
env_info['regressions_only']=ONLY_GPU_REGRESSIONS
if Path('/proc/cpuinfo').is_file():
    env_info['cpu_model']=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),env_info['cpu_model'])
dump(ROOT/'environment.json',env_info)
for name,cmd in [('pip_freeze.txt',[sys.executable,'-m','pip','freeze']),('nvidia_smi.txt',['nvidia-smi'])]:
    try:
        r=subprocess.run(cmd,text=True,capture_output=True,timeout=60);(ROOT/name).write_text(r.stdout+'\n'+r.stderr)
    except Exception as exc:(ROOT/name).write_text(repr(exc))
worker=ROOT/'worker.py';worker.write_text(WORKER,encoding='utf8')
jobs=[];outcomes=[]
def add(name,kind,backend=BACKEND,**kw):
    out=ROOT/'jobs'/name;out.mkdir(parents=True,exist_ok=True)
    job=dict(name=name,kind=kind,backend=backend,source=str(SOURCE),out=str(out),cpu_threads=CPU_THREADS,**kw)
    jobs.append(job);return out

add('01_cpu_regressions','pytest','cpu')
if BACKEND=='cuda':add('02_cuda_regressions','pytest')
add('03_analytic_gate','analytic',N=512 if PROFILE=='full' else 256)
add('04_independent_kernel_fields','kernel')
for name,cfg in REFERENCE_CASES:
    c=dict(cfg,res_tol=1e-8)
    # Keep the verified 4000-iteration budget even in quick mode. The rotating
    # reference needs more than 1200 steps to meet its energy-agreement gate.
    add('ref_'+name,'reference',cfg=c)

parity=[]
for name,cfg in [('harmonic2d',dict(beta2=100.)),('q2d',dict(dimension='quasi2D',kernel='q2d',l_z=.4,G_C=3.,beta2=30.)),
                 ('newton3d',dict(dimension='3D',s=0,Ngrid=32,L=3.,beta2=0.,G_C=10.,kernel='newton')),
                 ('rotating2d',dict(Ngrid=64,L=8.,beta2=100.,Omega=.8,seed='triangular'))]:
    a=add('parity_cpu_'+name,'solve','cpu',cfg=cfg)
    b=add('parity_gpu_'+name,'solve',cfg=cfg,initial=str(a/'initial.npz'))
    parity.append(dict(name=name,cpu=a.name,gpu=b.name))

convergence=[]
for name,cfg in [('newton2d',dict(beta2=30.,G_C=3.,kernel='newton')),
                 ('log2d',dict(beta2=30.,G_C=3.,kernel='log')),
                 ('q2d',dict(dimension='quasi2D',beta2=30.,G_C=3.,kernel='q2d',l_z=.4)),
                 ('sn3d',dict(dimension='3D',s=0,beta2=0.,G_C=10.,kernel='newton',L=3.))]:
    Ns=([32,48,64] if PROFILE=='full' else [24,32]) if cfg.get('dimension')=='3D' else ([64,96,128] if PROFILE=='full' else [32,48])
    group=[]
    for N in Ns:
        tag='conv_'+name+'_N'+str(N);add(tag,'solve',cfg=dict(cfg,Ngrid=N));group.append(tag)
    N=Ns[-1];L=cfg.get('L',6.)
    for suffix,extra in [('pad3',dict(Ngrid=N,pad=3)),('box',dict(Ngrid=int(round(N*1.25)),L=L*1.25))]:
        tag='conv_'+name+'_'+suffix;add(tag,'solve',cfg=dict(cfg,**extra));group.append(tag)
    convergence.append(dict(name=name,jobs=group,grid_jobs=group[:-2],padding_job=group[-2],box_job=group[-1]))

production=[]
if RUN_PAPER:
    for name,original in PAPER_CASES:
        cfg=dict(original);cfg.update(mode='single',energy_tol=0.,res_tol=1e-4,want_vortices=True,want_lll=True)
        seeds=['triangular','tf','gaussian'] if original.get('nseeds',1)>1 else [original['seed']]
        cfg['nseeds']=1
        if PROFILE=='quick':cfg.update(Ngrid=32 if cfg['dimension']=='3D' else 64,maxit=1200)
        tags=[]
        for j,seed in enumerate(seeds):
            tag='paper_'+name+'_seed'+str(j);c=dict(cfg,seed=seed,rng=j)
            add(tag,'solve',cfg=c,diagnostics=True,original_configuration=original,
                paper_resolution=PROFILE=='full');tags.append(tag)
        production.append(dict(name=name,jobs=tags,original_configuration=original))
    # The historical quasi2D showcase was the thin-disc 1/r option. Include a
    # finite-width q2d case as additional coverage, explicitly a new case.
    add('paper_extra_finite_width_q2d','solve',cfg=dict(dimension='quasi2D',Ngrid=256 if PROFILE=='full' else 64,
        L=10.,beta2=100.,G_C=5.,kernel='q2d',l_z=.5,maxit=20000 if PROFILE=='full' else 1200,want_vortices=True,want_lll=True),diagnostics=True)

timings=[]
for dim,Ns in [('2D',[128,256,512] if PROFILE=='full' else [64]),('3D',[32,48,64] if PROFILE=='full' else [24])]:
    for N in Ns:
        cfg=dict(dimension=dim,Ngrid=N,L=8.,beta2=100.,G_C=5.,kernel='newton')
        tags=[]
        for backend in ['cpu',BACKEND]:
            label='cpu' if len(tags)==0 else 'gpu'
            tag=f'timing_{dim}_N{N}_{label}';add(tag,'timing',backend,cfg=cfg,iterations=80 if PROFILE=='full' else 12,repeats=3);tags.append(tag)
        timings.append(dict(name=f'{dim}_N{N}',cpu=tags[0],gpu=tags[1]))
if RUN_METHOD_COMPARISON:
    for name,cfg in [('A_harmonic2d',dict(beta2=200.,L=10.,Ngrid=256,seed='tf')),
                     ('B_lattice2d',dict(beta2=200.,Omega=.9,L=12.,Ngrid=256,seed='triangular')),
                     ('C_harmonic3d',dict(dimension='3D',beta2=200.,L=8.,Ngrid=96,seed='tf')),
                     ('D_newton3d',dict(dimension='3D',beta2=100.,G_C=20.,kernel='newton',L=8.,Ngrid=96,seed='tf'))]:
        c=dict(cfg,res_tol=1e-8,maxit=40000)
        if PROFILE=='quick':c.update(Ngrid=24 if c.get('dimension')=='3D' else 64,maxit=3000)
        add('methods_'+name,'methods',cfg=c,itp_budget_s=900 if PROFILE=='full' else 60,itp_maxsteps=40000 if PROFILE=='full' else 2000)
dump(ROOT/'plan.json',dict(jobs=jobs,parity=parity,convergence=convergence,production=production,timings=timings))
if ONLY_GPU_REGRESSIONS:
    if BACKEND!='cuda':raise RuntimeError('The regression-only rerun requires a real CUDA GPU.')
    jobs=[j for j in jobs if j['name']=='02_cuda_regressions']
    parity=[];convergence=[];production=[];timings=[]
    dump(ROOT/'plan.json',dict(jobs=jobs,scope='97 repaired CUDA regression tests only'))
if ALLOW_CPU_SELFTEST and BACKEND=='cpu':
    # Developer-only path to verify the harness locally without CUDA. The
    # report explicitly stays GPU=false and can never receive overall PASS.
    keep={'03_analytic_gate','04_independent_kernel_fields','ref_harmonic2d','ref_q2d','ref_sn3d','ref_rotating2d','timing_2D_N64_cpu'}
    jobs=[j for j in jobs if j['name'] in keep]
    parity=[];convergence=[];production=[];timings=[]
    dump(ROOT/'plan.json',dict(jobs=jobs,cpu_selftest_only=True))

def run_job(job):
    out=Path(job['out']);payload=out/'job.json';dump(payload,job)
    env=os.environ.copy();env.update(PYTHONPATH=str(SOURCE),OMP_NUM_THREADS=str(CPU_THREADS),MKL_NUM_THREADS=str(CPU_THREADS),PYTHONHASHSEED='0')
    if job['backend']=='cpu':env['CUDA_VISIBLE_DEVICES']=''
    # Fresh process per job prevents device caches/imports from leaking between
    # CPU/GPU runs. Keep GPU jobs on the notebook's first visible device.
    started=time.monotonic();proc=subprocess.Popen([sys.executable,'-u',str(worker),str(payload)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,text=True,bufsize=1)
    messages=queue.Queue()
    def read_output():
        for line in proc.stdout:messages.put(line)
    reader=threading.Thread(target=read_output,daemon=True);reader.start();last=time.monotonic()
    try:
        with (out/'stdout.log').open('w',encoding='utf8') as log:
            while proc.poll() is None or not messages.empty() or reader.is_alive():
                try:
                    line=messages.get(timeout=1);log.write(line);log.flush()
                    if line.startswith(('RESULT','[gs]','  [gs]','FAILED','ERROR')):print(line.rstrip(),flush=True)
                except queue.Empty:pass
                if time.monotonic()-last>30:
                    print('  running',job['name'],round(time.monotonic()-started),'seconds',flush=True);last=time.monotonic()
                if time.monotonic()-started>JOB_TIMEOUT_S:
                    proc.kill();proc.wait();return dict(status='TIMEOUT',job=job,seconds=time.monotonic()-started)
        path=out/'result.json'
        result=json.loads(path.read_text()) if path.exists() else dict(status='FAIL',job=job,error='Worker produced no result; inspect stdout.log')
        if proc.returncode and result.get('status') in ('PASS','REVIEW'):result['status']='FAIL';result['error']='Nonzero worker exit'
        return result
    except BaseException:
        proc.kill();proc.wait();raise

def summaries():
    lookup={r['job']['name']:r for r in outcomes}
    pair_results=[]
    for pair in parity:
        a=lookup.get(pair['cpu'],{});b=lookup.get(pair['gpu'],{})
        if 'observables' not in a or 'observables' not in b:pair_results.append(dict(name=pair['name'],status='INCOMPLETE'));continue
        ae=a['observables']['E'];be=b['observables']['E'];err=abs(ae-be)/max(1,abs(ae))
        import numpy as np
        p=np.load(ROOT/'jobs'/pair['cpu']/'state.npz')['psi'];q=np.load(ROOT/'jobs'/pair['gpu']/'state.npz')['psi']
        dv=(2*a['cfg']['L']/a['cfg']['Ngrid'])**p.ndim
        density=float(np.sqrt(np.sum((abs(p)**2-abs(q)**2)**2)*dv))
        overlap=np.vdot(p,q)*dv;phase=np.exp(-1j*np.angle(overlap))
        wave=float(np.sqrt(np.sum(abs(p-phase*q)**2)*dv))
        strict=a['checks']['strict_residual_pass'] and b['checks']['strict_residual_pass']
        pair_results.append(dict(name=pair['name'],status='PASS' if err<2e-7 and strict else 'REVIEW',energy_relative_difference=err,
            density_L2_difference=density,phase_aligned_wavefunction_L2_difference=wave,strict_residual_both=strict,
            note='Rotating states can occupy different branches; density/wave differences are diagnostics, not a global-minimum test.'))
    convergence_results=[]
    for group in convergence:
        records=[lookup.get(tag,{}) for tag in group['jobs']]
        if any('observables' not in r for r in records):convergence_results.append(dict(name=group['name'],status='INCOMPLETE'));continue
        grid_records=records[:-2];fine=grid_records[-1];scale=max(1,abs(fine['observables']['E']))
        grid_delta=abs(fine['observables']['E']-grid_records[-2]['observables']['E'])/scale
        padding_delta=abs(fine['observables']['E']-records[-2]['observables']['E'])/scale
        box_delta=abs(fine['observables']['E']-records[-1]['observables']['E'])/scale
        ok=all(r['checks']['strict_residual_pass'] and r['checks']['boundary_pass'] for r in records)
        ok=ok and grid_delta<5e-3 and padding_delta<2e-6 and box_delta<5e-3
        convergence_results.append(dict(name=group['name'],status='PASS' if ok else 'REVIEW',last_grid_relative_delta=grid_delta,
            padding_relative_delta=padding_delta,box_relative_delta=box_delta,
            note='Predeclared finite-grid checks; inspect trends and residuals before using paper accuracy claims.'))
    selected=[]
    for group in production:
        available=[lookup[t] for t in group['jobs'] if t in lookup and 'observables' in lookup[t]]
        best=min(available,key=lambda r:r['observables']['E']) if available else None
        selected.append(dict(name=group['name'],completed_seeds=len(available),planned_seeds=len(group['jobs']),
            selected_job=best['job']['name'] if best else None,selected_energy=best['observables']['E'] if best else None,
            selected_residual=best['observables']['resid_rel'] if best else None,
            status='PASS' if best and len(available)==len(group['jobs']) and best['status']=='PASS' else 'REVIEW'))
    speed=[]
    for pair in timings:
        a=lookup.get(pair['cpu'],{});b=lookup.get(pair['gpu'],{})
        complete=a.get('fixed_iteration_complete',False) and b.get('fixed_iteration_complete',False)
        speed.append(dict(name=pair['name'],status='PASS' if complete else 'INCOMPLETE',
            cpu_seconds=a.get('median_seconds'),gpu_seconds=b.get('median_seconds'),
            cpu_over_gpu=a['median_seconds']/b['median_seconds'] if complete and BACKEND=='cuda' else None,
            note='Same fixed iteration budget with warmup and three repeats; this is not time-to-accuracy or an ITP comparison.'))
    assessments=pair_results+convergence_results+selected+speed
    method_results=[dict(name=r['job']['name'],status=r['status'],reference_energy=r.get('reference_energy'),
        reference_residual=r.get('reference_residual'),methods={k:dict(energy=v['energy'],residual=v['residual'],stop=v['stop'],time_to_relative_energy=v['time_to_relative_energy']) for k,v in r.get('methods',{}).items()})
        for r in outcomes if r['job']['kind']=='methods']
    complete=len(outcomes)==len(jobs)
    counts={status:sum(r.get('status')==status for r in outcomes) for status in sorted({r.get('status') for r in outcomes})}
    all_pass=complete and BACKEND=='cuda' and all(r['status']=='PASS' for r in outcomes+assessments)
    hard_failure=any(r.get('status')=='FAIL' for r in outcomes)
    result=dict(environment=env_info,planned_jobs=len(jobs),completed_jobs=len(outcomes),status_counts=counts,
        overall='FAIL' if hard_failure else ('PASS' if all_pass else 'REVIEW_OR_INCOMPLETE'),gpu_executed=BACKEND=='cuda',
        parity=pair_results,convergence=convergence_results,production=selected,timings=speed,method_comparisons=method_results,
        limitations=['Same-grid agreement is not continuum convergence or global minimality.',
                     'Reference energy agreement does not imply the requested 1e-8 residual was attained.',
                     'ITP method comparisons use the lowest returned candidate as reference; shared diagnostic operators are disclosed in each result.',
                     'Quick profile and CPU selftest cannot replace full paper-resolution CUDA runs.'])
    dump(ROOT/'SUMMARY.json',result)
    with (ROOT/'jobs_summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['name','backend','status','energy','relative_residual','seconds'])
        for r in outcomes:
            o=r.get('observables',{});w.writerow([r['job']['name'],r['job']['backend'],r['status'],o.get('E'),o.get('resid_rel'),r.get('seconds')])
    dump(ROOT/'outcomes.json',outcomes)
    return result

try:
    print('Planned jobs:',len(jobs),'profile:',PROFILE,'paper grids:',RUN_PAPER and PROFILE=='full',flush=True)
    for index,job in enumerate(jobs,1):
        print(f'[{index}/{len(jobs)}] {job["name"]} ({job["backend"]})',flush=True)
        result=run_job(job);outcomes.append(result);summary=summaries()
        print(' ->',result['status'],flush=True)
except KeyboardInterrupt:
    print('Interrupted. Completed results are being archived.',flush=True)
finally:
    summary=summaries()
    # Exclude source/dependencies; include every job log, state and trace.
    evidence=[p for p in ROOT.rglob('*') if p.is_file() and 'source' not in p.relative_to(ROOT).parts and p.name!='pinned_source.zip']
    dump(ROOT/'SHA256.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(evidence) if p.name!='SHA256.json'})
    archive=ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if p.is_file() and 'source' not in p.relative_to(ROOT).parts and p.name!='pinned_source.zip':z.write(p,p.relative_to(ROOT))
    print('\nFINAL:',summary['overall'],'GPU executed:',summary['gpu_executed'])
    print('SUMMARY:',ROOT/'SUMMARY.json');print('SEND THIS ZIP BACK:',archive)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(archive)))
    except ImportError:pass
