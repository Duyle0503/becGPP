"""Validate uploaded evidence, recompute saved CUDA states on CPU, and replot."""
import argparse,csv,gc,hashlib,json,math,os,subprocess,sys,zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--software',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--figures',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.raw=a.raw.resolve();a.out.mkdir(parents=True,exist_ok=True);a.figures.mkdir(parents=True,exist_ok=True)
    def save(name,obj):
        (a.out/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
    hashes=json.loads((a.raw/'SHA256.json').read_text());bad=[]
    for name,expected in hashes.items():
        path=(a.raw/name).resolve()
        if not path.is_relative_to(a.raw):raise ValueError('Unsafe evidence path')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:bad.append(name)
    assert not bad,bad
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None
        assert json.loads(z.read('SHA256.json'))==hashes
    manifest=json.loads((a.raw/'source_manifest.json').read_text())
    for name,expected in manifest['normalized_module_sha256'].items():
        raw=subprocess.run(['git','show',manifest['commit']+':'+name],cwd=a.software,check=True,capture_output=True).stdout
        assert hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()==expected,name
    modules={name:expected for name,expected in manifest['normalized_module_sha256'].items() if name.startswith('becgpp/')}
    assert all(hashlib.sha256((a.software/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for n,h in modules.items())
    outcomes=json.loads((a.raw/'outcomes.json').read_text());summary=json.loads((a.raw/'SUMMARY.json').read_text())
    by={x['job']['name']:x for x in outcomes}
    assert len(outcomes)==len(by)==summary['completed_jobs']==summary['planned_jobs']==71
    counts={s:sum(x['status']==s for x in outcomes) for s in ['PASS','REVIEW','FAIL']}
    assert counts==summary['status_counts']
    for name,x in by.items():
        disk=json.loads((a.raw/'jobs'/name/'result.json').read_text())
        assert disk==x,name
    errors=[]
    xml=ET.parse(a.raw/'jobs/02_cuda_regressions/pytest.xml').getroot()
    for test in xml.iter('testcase'):
        f=test.find('failure')
        if f is not None:
            assert 'device' in (f.text or '') and 'cpu' in (f.text or '')
            errors.append(dict(test=test.attrib['name'],type=f.attrib.get('type'),message=f.attrib.get('message')))
    assert len(errors)==16
    refs=[x for x in outcomes if x['job']['kind']=='reference']
    assert len(refs)==10 and all(x['reference']['energy_pass'] for x in refs)
    audit=dict(source_commit=manifest['commit'],raw_files_verified=len(hashes),archive_crc_valid=True,
               archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),numerical_module_count=len(modules),
               source_modules_unchanged=True,status_counts=counts,cuda_regression_fixture_errors=errors,
               independent_energy_checks=10,strict_reference_residual_passes=sum(x['checks']['strict_residual_pass'] for x in refs),
               max_nonrotating_energy_difference=max(x['reference']['energy_difference'] for x in refs if x['job']['name']!='ref_rotating2d'),
               rotating_energy_difference=by['ref_rotating2d']['reference']['energy_difference'],environment=summary['environment'])
    save('evidence_audit.json',audit);print('Integrity/provenance verified:',len(hashes),'files;',counts,flush=True)

    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
    sys.path.insert(0,str(a.software))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from becgpp import make_grid,paths
    from becgpp.operators import observables,compute_residual
    from becgpp.interactions import clear_kernel_cache
    from becgpp.fields import center_of_mass,radial_average,gaussian_smooth
    from becgpp.figures import save_state_figs,_savefig
    import matplotlib.pyplot as plt
    assert not torch.cuda.is_available()
    paths.configure(str(a.out/'plots'))
    figures=[];checks=[]
    if a.resume and (a.out/'saved_state_cpu_checks.json').is_file():
        checks=json.loads((a.out/'saved_state_cpu_checks.json').read_text())
    selected={x['name']:x['selected_job'] for x in summary['production']}
    selected['extra_finite_width_q2d']='paper_extra_finite_width_q2d'
    for label,name in selected.items():
        x=by[name];c=x['cfg'];directory=a.raw/'jobs'/name
        if any(check['case']==label for check in checks):
            for f in sorted(a.figures.glob('gpu_scan_'+label+'*.pdf')):
                figures.append(dict(file=f.name,source_job=name,state_sha256=hashes['jobs/'+name+'/state.npz'],operation='replot saved CUDA state; no minimization',sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
            print('Already verified:',label,flush=True)
            continue
        state=np.load(directory/'state.npz',allow_pickle=False)['psi']
        assert state.dtype==np.complex128 and np.isfinite(state).all()
        psi=torch.from_numpy(state);G=make_grid(c);dv=G['dx']**G['ndim']
        mass=(psi.abs()**2).sum().item()*dv
        o=observables(psi,G,c);_,_,resid=compute_residual(psi,G,c)
        e_error=abs(o['E']-x['observables']['E'])/max(1.,abs(o['E']))
        assert e_error<1e-10 and abs(mass-1)<1e-10
        assert abs(resid-x['observables']['resid_rel'])<1e-9
        check=dict(case=label,job=name,energy_cpu=o['E'],energy_gpu=x['observables']['E'],relative_energy_difference=e_error,
                   mass_error=abs(mass-1),residual_cpu=resid,residual_gpu=x['observables']['resid_rel'],virial_cpu=o['virial_rel'])
        checks.append(check);save('saved_state_cpu_checks.json',checks)
        rid='gpu_scan_'+label;save_state_figs(psi,G,c,rid)
        if label in ['2D_harmonic_repulsive','2D_cubicquintic_flattop','2D_quartic_trap','2D_gravitylike_droplet','3D_newton_bosonstar']:
            rho=psi.abs()**2;d=x['diagnostics'];center=center_of_mass(rho,G)
            R=d.get('R_1pct');rmax=1.3*R if R and math.isfinite(R) else 1.3*d['R90']
            if G['ndim']==2 and d['Nv']>=6:
                av=math.sqrt(2*math.pi/(math.sqrt(3)*max(abs(c['Omega']),1e-12)))
                rho=torch.clamp(gaussian_smooth(rho,G,min(.8*av,.15*d['R90'])),min=0.)
            rr,nn=radial_average(rho,G,rmax=rmax,center=center)
            tf=np.load(directory/'tf_density.npz',allow_pickle=False)['density']
            tf_mass=float(tf.sum()*dv);assert abs(tf_mass-1)<1e-7
            rt,nt=radial_average(tf,G,rmax=rmax)
            fig,ax=plt.subplots(figsize=(3.4,2.7));ax.plot(rr,nn,color='#1f4e79',label='GPE');ax.plot(rt,nt,'--',color='#c1272d',label='TF')
            ax.set(xlabel=r'$r$',ylabel=r'$n(r)$',xlim=(0,rmax),ylim=(0,None));ax.legend();_savefig(fig,str(Path(paths.FIG_DIR)/(rid+'_TF')))
            check['stored_tf_mass_error']=abs(tf_mass-1)
        for f in sorted(Path(paths.FIG_DIR).glob(rid+'*.pdf')):
            dest=a.figures/f.name;dest.write_bytes(f.read_bytes())
            figures.append(dict(file=f.name,source_job=name,state_sha256=hashes['jobs/'+name+'/state.npz'],operation='replot saved CUDA state; no minimization',sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        del psi,G,state;clear_kernel_cache();gc.collect();print('CPU cross-check + figures:',label,format(e_error,'.2e'),flush=True)
    save('saved_state_cpu_checks.json',checks)
    fig,axes=plt.subplots(2,2,figsize=(7.8,5.8),constrained_layout=True)
    for panel,(ax,name) in enumerate(zip(axes.flat,['A_harmonic2d','B_lattice2d','C_harmonic3d','D_newton3d'])):
        x=by['methods_'+name];ref=x['reference_energy'];scale=max(1,abs(ref))
        for label,color,style in [('PCG_pp','#1f4e79','-'),('PCG_pr','#1f4e79','--'),('ITP','#c1272d','-')]:
            h=x['methods'][label]['history'];ax.semilogy([v[0] for v in h],[max(abs(v[2]-ref)/scale,1e-15) for v in h],style,color=color,label=label)
        ax.axhline(1e-6,color='.5',ls=':',lw=.8);ax.set(xlabel='Time (s)',ylabel=r'$|E-E_{ref}|/\max(1,|E_{ref}|)$');ax.legend(fontsize=7)
        ax.text(-.12,1.03,'('+chr(97+panel)+')',transform=ax.transAxes,fontsize=10)
    _savefig(fig,str(a.figures/'gpu_method_comparison'))
    fig,ax=plt.subplots(figsize=(4.6,3.3))
    for dim,marker in [('2D','o'),('3D','s')]:
        rows=[v for v in summary['timings'] if v['name'].startswith(dim)]
        n=[int(v['name'].split('N')[1]) for v in rows];ax.plot([v**(2 if dim=='2D' else 3) for v in n],[v['cpu_over_gpu'] for v in rows],marker+'-',label=dim)
    ax.set(xscale='log',xlabel='Grid points',ylabel='CPU / GPU time');ax.legend();_savefig(fig,str(a.figures/'gpu_cpu_speedup'))
    for name in ['gpu_method_comparison.pdf','gpu_cpu_speedup.pdf']:
        f=a.figures/name;figures.append(dict(file=name,source_job='method/timing records',operation='plot uploaded histories and synchronized median timings',sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    save('figure_provenance.json',figures)
    print('Finished',len(checks),'saved-state checks and',len(figures),'PDFs',flush=True)

if __name__=='__main__':main()
