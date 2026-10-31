# Copy this ENTIRE file into ONE Kaggle/Colab Python cell with CUDA enabled.
# This is a predeclared protocol, not a claim that these jobs have been run.
"""Tagged-release trapped-3D refinement, with separate grid/box/padding checks."""
SOLVER_COMMIT = "1c7eef613d147b6857375f9a61342b38cae1bc86"
EXPECTED_MODULE_HASHES = {'becgpp/__init__.py': 'e15ce998013c4c179da3c3ec111330ee1a59a9c58f01a7957bb8fbf8aa2f32be', 'becgpp/__main__.py': 'fda8a2d68e55cb5447db6e92d4cf53fde893a94d0f5e97d42873c2d35c599d66', 'becgpp/cli.py': '24cccd8fd82ab742dd95991280a46e640b85426c1ddfdcbb1ebec2dff3fa7da0', 'becgpp/config.py': 'bc8bf1ae35bc7cedbed62952c51d2185d58f7c2863998f14a7b68ccf552c8c1c', 'becgpp/constants.py': 'fc7c0b901a4325a3c52cf04086dc3eca08d8ed91bf26884ac9cc62c7ce2a221c', 'becgpp/diagnostics.py': '2e65a9e6350ae5b6bcb11dff94e644f7c1efc72521b6c8be616a7445d3f7caf6', 'becgpp/fields.py': 'dbeae4fb2ec1bd8ca3b901aef0cc8f358d54218ac485b2d7bdc0df8cf8bf9003', 'becgpp/figures.py': '1f91419b33ccbc32ff06cdff769a549b0d97e8dac7dc42fc4a7e87654b2ec333', 'becgpp/grid.py': 'e04aa8e4a16f500b77d77d6128a2de0005259f9472814e7958a313a352aea42f', 'becgpp/interactions.py': '195d1c7bf4e08288c88e485e031c06c2ad87116841ea38c5e83aecb9574565ef', 'becgpp/io.py': 'ce98e62d08962b21ec41d4e6b79d343ea3620a011fd796820491d2326f602d64', 'becgpp/modes.py': '93970f83a19e88d0078c5043731b9680a104c4b352c06fe46bf271f507de4356', 'becgpp/operators.py': 'c38d83bb434d6a5ec43631f94ebfa637fe61a84d65377de42677626fe5bbc794', 'becgpp/paths.py': '44dd2c098986c7da274466124c8cdaed441f3de8f71aff006fdadf0d9ca210c5', 'becgpp/seeds.py': '2a449135c7beb2b643c6c08c4d7c681a244be04c4f92c6094508e1660dd459f5', 'becgpp/solvers.py': 'a2e05b0ecd0da409681e4de65ea0f1ee3c5b9135f301a17bd5eb6026086bcef3', 'becgpp/thomasfermi.py': '458f49baa8d810838f67dfcde2a9e0360896c7f76dce9ea31e8b7804732fe8f5', 'becgpp/units.py': 'a910852f2117736e26d7677271d1b99250cc8a1426970d463468ea02b2f19847', 'becgpp/vortex.py': '3e6ed6b6c16f863ddc944ad30415c2c67c310a2c947b66c1cfe63c4493a2dec8'}
SAVE_STATES = True
JOB_TIMEOUT_SECONDS = 7200

import hashlib, io, json, math, os, platform, subprocess, sys, tempfile, time, urllib.request, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. This cell does not substitute CPU results or reduce the planned grids.")
assert len(EXPECTED_MODULE_HASHES) == 19
ROOT = Path(tempfile.mkdtemp(prefix="becgpp_trapped3d_refinement_"))
archive_url = "https://codeload.github.com/Duyle0503/becGPP/zip/" + SOLVER_COMMIT
with urllib.request.urlopen(archive_url, timeout=120) as response:
    source_archive = zipfile.ZipFile(io.BytesIO(response.read()))
assert source_archive.testzip() is None
for name in source_archive.namelist():
    destination = (ROOT / "source" / name).resolve()
    assert destination.is_relative_to((ROOT / "source").resolve())
    source_archive.extract(name, ROOT / "source")
SOURCE = ROOT / "source" / ("becGPP-" + SOLVER_COMMIT)
for name, expected in EXPECTED_MODULE_HASHES.items():
    actual = hashlib.sha256((SOURCE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert actual == expected, name

def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

WORKER = r'''
import csv, gc, hashlib, json, math, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
job=json.loads(Path(sys.argv[1]).read_text());out=Path(job['out']);out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,job['source'])
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True,warn_only=True)
from becgpp import default_cfg, make_grid, make_seed, ground_state, mass_quantiles, DEV
from becgpp.grid import dV, _memory_estimate_gb
from becgpp.interactions import clear_kernel_cache
assert DEV=='cuda'
def save(data):
    (out/'result.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
try:
    c=default_cfg(dimension='3D',s=2,trap_coeff=.5,Omega=0.,beta2=100.,beta3=0.,G_C=20.,kernel='newton',
                  L=job['L'],Ngrid=job['N'],pad=job['pad'],seed='tf',rng=0,nseeds=1,
                  seed_noise=.01,seed_phase_noise=.05,res_tol=1e-6,energy_tol=0.,maxit=4000,
                  step=1.,step_max=3.,linesearch_max=8,precond_shift_min=.5,cg_beta='pr_precond',
                  save_figs=False,save_ckpt=False,show_inline=False,want_vortices=False,want_lll=False,record_trace=True)
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    free,total=torch.cuda.mem_get_info()
    estimate=_memory_estimate_gb(3,job['N'],job['pad'],'newton')
    if estimate*1e9 > .8*free:
        save(dict(status='BLOCKED_MEMORY',estimated_gb=estimate,free_bytes=free,total_bytes=total,job=job))
        sys.exit(0)
    G=make_grid(c);p0=make_seed(G,c);p0/=torch.sqrt((p0.abs()**2).sum()*dV(G))
    initial_hash=hashlib.sha256(p0.detach().cpu().numpy().tobytes()).hexdigest()
    ground_state(dict(c,maxit=3,res_tol=0.,energy_tol=0.),G=G,psi0=p0.clone(),verbose=False)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    p,G,o=ground_state(c,G=G,psi0=p0.clone(),verbose=False)
    torch.cuda.synchronize();elapsed=time.perf_counter()-started
    assert p.device.type=='cuda' and p.dtype==torch.complex128 and torch.isfinite(p).all().item()
    mass_error=abs((p.abs()**2).sum().item()*dV(G)-1.)
    edge=torch.zeros_like(p.real,dtype=torch.bool)
    for ax in range(3):
        ids=[slice(None)]*3;ids[ax]=slice(0,2);edge[tuple(ids)]=True
        ids[ax]=slice(-2,None);edge[tuple(ids)]=True
    edge_mass=float((p.abs()**2)[edge].sum().item()*dV(G))
    R90=float(mass_quantiles(p.abs()**2,G,(.90,))[0]);dx=2*c['L']/c['Ngrid']
    trace=o.pop('trace')
    with (out/'trace.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seconds','iteration','energy','relative_residual']);w.writerows(trace)
    if job['save_states']:np.savez_compressed(out/'state.npz',psi=p.detach().cpu().numpy())
    # Coarse-grid virial errors are recorded, not rejected as though they had
    # to satisfy the finest-grid accuracy criterion before refinement.
    passed=o['resid_rel']<1e-6 and mass_error<1e-10 and edge_mass<1e-6
    save(dict(status='PASS' if passed else 'REVIEW',job=job,cfg=c,energy=o['E'],mu=o['mu'],residual=o['resid_rel'],
              virial_relative=o['virial_rel'],R90=R90,points_per_R90=R90/dx,dx=dx,mass_error=mass_error,
              edge_mass_two_cells=edge_mass,iterations=o['iters'],stop_reason=o['stop_reason'],
              initial_array_sha256=initial_hash,seconds=elapsed,actual_device=str(p.device),dtype=str(p.dtype),
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              gpu=torch.cuda.get_device_name(0),torch=torch.__version__,python=sys.version,platform=platform.platform()))
except torch.cuda.OutOfMemoryError as error:
    save(dict(status='BLOCKED_MEMORY',job=job,error=str(error)))
except Exception as error:
    traceback.print_exc();save(dict(status='FAIL',job=job,error=str(error)))
    sys.exit(1)
'''
(ROOT / "worker.py").write_text(WORKER)
jobs = [dict(name="grid_N" + str(n), N=n, L=8., pad=2) for n in [96,128,160,192]]
jobs += [dict(name="padding_N128_pad3",N=128,L=8.,pad=3),
         dict(name="box_L10_N160",N=160,L=10.,pad=2)]
for job in jobs:
    job.update(source=str(SOURCE),out=str(ROOT / "jobs" / job["name"]),save_states=SAVE_STATES)
dump(ROOT / "plan.json", dict(jobs=jobs,source_commit=SOLVER_COMMIT,
     physical_case="3D harmonic trap, beta2=100,G_C=20,Omega=0; original trapped showcase physics",
     controls="Grid study L=8,pad=2; separate same-N padding check; separate same-dx=.125 box check.",
     criteria=dict(residual=1e-6,mass_error=1e-10,edge_mass_two_cells=1e-6,finest_virial_relative=5e-4,
                   last_grid_relative_energy_change=1e-3,padding_relative_energy_change=2e-11,box_relative_energy_change=1e-5,
                   observed_order_interval=[1.5,2.5]),
     limitations="A pass is a finite-resolution study, not 1e-8 continuum accuracy or a global minimum proof."))
dump(ROOT / "source_manifest.json",dict(commit=SOLVER_COMMIT,normalized_module_sha256=EXPECTED_MODULE_HASHES))
dump(ROOT / "environment.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
     torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),device=0,threads=2))
results=[]
for job in jobs:
    job_path=ROOT/'jobs'/job['name']/'job.json';dump(job_path,job)
    print('RUN',job['name'],flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MPLBACKEND='Agg',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log=job_path.parent/'stdout.log'
    with log.open('w') as stream:
        try:
            process=subprocess.run([sys.executable,str(ROOT/'worker.py'),str(job_path)],stdout=stream,stderr=subprocess.STDOUT,
                                   env=env,timeout=JOB_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            dump(job_path.parent/'result.json',dict(status='TIMEOUT',job=job))
    result_path=job_path.parent/'result.json'
    result=json.loads(result_path.read_text()) if result_path.exists() else dict(status='INCOMPLETE',job=job)
    results.append(result);print(job['name'],result['status'],result.get('energy'),result.get('residual'),flush=True)
dump(ROOT/'outcomes.json',results)
by={r['job']['name']:r for r in results};comparisons={}
if all(r['status'] in ('PASS','REVIEW') for r in results):
    e=[by['grid_N'+str(n)]['energy'] for n in [96,128,160,192]];h=[16/n for n in [96,128,160,192]]
    def observed_order(energies,widths):
        a,b,c=energies;x,y,z=widths
        if (a-b)*(b-c)<=0:return None
        ratio=(a-b)/(b-c)
        def f(p):return (x**p-y**p)/(y**p-z**p)-ratio
        lo,hi=.25,6.
        if f(lo)*f(hi)>0:return None
        for _ in range(70):
            mid=(lo+hi)/2
            if f(lo)*f(mid)<=0:hi=mid
            else:lo=mid
        return (lo+hi)/2
    p1=observed_order(e[:3],h[:3]);p2=observed_order(e[1:],h[1:])
    rel=lambda a,b:abs(a-b)/max(1.,abs(b))
    fit=np.polyfit(np.array(h[1:])**2,np.array(e[1:]),1)
    comparisons=dict(last_grid_relative_energy_change=rel(e[-2],e[-1]),
       padding_relative_energy_change=rel(by['padding_N128_pad3']['energy'],by['grid_N128']['energy']),
       box_relative_energy_change=rel(by['box_L10_N160']['energy'],by['grid_N128']['energy']),
       observed_orders=[p1,p2],finest_virial_relative=by['grid_N192']['virial_relative'],assumed_h2_extrapolated_energy=float(fit[1]))
    comparisons['criteria_pass']=comparisons['last_grid_relative_energy_change']<1e-3 and comparisons['padding_relative_energy_change']<2e-11 and comparisons['box_relative_energy_change']<1e-5 and comparisons['finest_virial_relative']<5e-4 and all(p is not None and 1.5<p<2.5 for p in [p1,p2])
overall='PASS' if all(r['status']=='PASS' for r in results) and comparisons.get('criteria_pass') else 'REVIEW_OR_INCOMPLETE'
dump(ROOT/'SUMMARY.json',dict(overall=overall,completed_jobs=len(results),planned_jobs=len(jobs),comparisons=comparisons,
     status_counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},
     evidence_scope='Tagged-release GPU trapped3D grid/box/padding study; raw failures retained; no grid reduction.'))
hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and 'source' not in p.relative_to(ROOT).parts}
dump(ROOT/'SHA256.json',hashes)
ZIP=ROOT.with_suffix('.zip')
with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and 'source' not in p.relative_to(ROOT).parts:z.write(p,p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(ZIP) as z:assert z.testzip() is None
print('SUMMARY',overall,comparisons,flush=True)
print('Download',ZIP,'SHA256',hashlib.sha256(ZIP.read_bytes()).hexdigest(),flush=True)
