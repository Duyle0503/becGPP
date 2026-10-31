"""Build portable CPC archives; validate reused data and omit build caches.
Integrity PASS is distinct from physical validation or journal readiness.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess, zipfile
from workspace_paths import find_workspace
ROOT=find_workspace(); OUT=ROOT/'data/submission'
EXCLUDED={'.git','.lake','__pycache__','.pytest_cache','build','dist','becgpp.egg-info','becgpp_out','latex_source_check','pdf_pages','pdf_qa'}
def stream_sha(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def sha(path):
    with path.open('rb') as f:return stream_sha(f)
def tree(folder,prefix,exclusions=EXCLUDED):
    for base,dirs,files in os.walk(folder):
        dirs[:]=sorted(d for d in dirs if d not in exclusions and not d.startswith('discarded_environment_trial'))
        for name in sorted(files):
            if name.startswith('discarded_environment_trial') or name.endswith(('.pyc','.tmp')) or name in {'packaging.log','packaging_final.log'}:continue
            p=Path(base)/name
            yield p,(Path(prefix)/p.relative_to(folder)).as_posix()
def check_zip(path,compare_files=False):
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert z.testzip() is None
        name='CHECKSUMS/'+path.name.removesuffix('.zip')+'.json';records=json.loads(z.read(name))
        assert set(z.namelist())==set(records)|{name}
        for arc,value in records.items():
            assert not Path(arc).is_absolute() and '..' not in Path(arc).parts and ':' not in arc
            with z.open(arc) as f:assert stream_sha(f)==value,arc
            if compare_files and arc!='data/README.md':assert sha(ROOT/arc)==value,arc
    return records
def make_zip(name,members):
    target=OUT/name;temp=target.with_suffix('.partial.zip');members=list(members)
    assert len({a for _,a in members})==len(members);records={}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p,arc in members:
            records[arc]=sha(p)
            compression=zipfile.ZIP_STORED if p.suffix.lower() in {'.zip','.whl','.gz','.png','.jpg','.npz'} else zipfile.ZIP_DEFLATED
            z.write(p,arc,compress_type=compression)
        z.writestr('CHECKSUMS/'+name.removesuffix('.zip')+'.json',json.dumps(records,indent=2)+'\n')
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for arc,value in records.items():
            with z.open(arc) as f:assert stream_sha(f)==value,arc
    if target.exists():
        backup=ROOT/'data/archive/before_final_revision_2026_10_29/submission_components';backup.mkdir(parents=True,exist_ok=True)
        if not (backup/name).exists():shutil.copy2(target,backup/name)
    temp.replace(target)
    print(name,len(members),'files;',target.stat().st_size,'bytes; CRC and hashes verified',flush=True)
    return target
def source_members():
    repo=ROOT/'code/becGPP';manifest=ROOT/'code/SOURCE_MANIFEST.json'
    if (repo/'.git').exists():
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True).strip(),'Commit tracked changes before packaging'
        names=subprocess.check_output(['git','ls-files','-z'],cwd=repo).decode().rstrip('\0').split('\0')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        data=dict(source_commit=commit,numerical_release_commit='1c7eef613d147b6857375f9a61342b38cae1bc86',
                  description='Release 1.2.2 numerical modules/tests plus later documentation and verification tools',files={n:sha(repo/n) for n in names})
        manifest.write_text(json.dumps(data,indent=2)+'\n')
    else:
        data=json.loads(manifest.read_text());names=sorted(data['files'])
        assert all(sha(repo/n)==v for n,v in data['files'].items())
    return [(repo/n,'code/becGPP/'+n) for n in names]+[(manifest,'code/SOURCE_MANIFEST.json')]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rebuild-reproducibility',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((ROOT/'data/review/FINAL_PROJECT_CHECK.json').read_text())['integrity_status']=='PASS'
    source=make_zip('Software_source.zip',source_members());latex=ROOT/'latex'
    texts='\n'.join((latex/n).read_text(encoding='utf-8') for n in ('main.tex','candidate_verification.tex','controlled_study.tex'))
    figures=sorted(set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',texts)))
    lm=[(latex/n,'latex/'+n) for n in ('main.tex','references.bib','candidate_verification.tex','controlled_study.tex')]
    lm += [(latex/'build/main.bbl','latex/main.bbl')]+[(latex/'figs'/n,'latex/figs/'+n) for n in figures]
    latex_source=make_zip('LaTeX_source.zip',lm)
    em=list(tree(ROOT/'data/review','data/review'))+list(tree(ROOT/'data/verification','data/verification'))
    em+=list(tree(ROOT/'data/history/pinned_sources','data/history/pinned_sources'))+[(ROOT/'data/README.md','data/README.md')]
    evidence=make_zip('Verification.zip',em);raw=OUT/'Reproducibility_data.zip'
    if raw.exists() and not args.rebuild_reproducibility:
        old=json.loads((OUT/'SHA256.json').read_text())['Reproducibility_data.zip']
        assert sha(raw)==old['sha256'] and raw.stat().st_size==old['bytes'];records=check_zip(raw,True)
        if (ROOT/'data/raw/release_1_2_2_cuda').exists():
            assert any(n.startswith('data/raw/release_1_2_2_cuda/') for n in records),'New CUDA evidence requires --rebuild-reproducibility'
        print('Reproducibility_data.zip reused; all archived bytes verified. Verification.zip overlays the current data README.',flush=True)
    else:
        rm=[]
        for name in ('gpu_2026_10_17','gpu_retest_2026_10_17','release_1_2_2_ci','release_1_2_2_cuda'):rm+=list(tree(ROOT/'data/raw'/name,'data/raw/'+name))
        rm+=list(tree(ROOT/'data/history/legacy_v1_0','data/history/legacy_v1_0'))
        rm+=[(ROOT/'data/README.md','data/README.md'),(ROOT/'data/raw/archives/becgpp_gpu_20261017T141120Z_cwk2teki.zip','data/raw/archives/becgpp_gpu_20261017T141120Z_cwk2teki.zip')]
        raw=make_zip('Reproducibility_data.zip',rm)
    payload=[(ROOT/'README.md','README.md'),(source,'code/Software_source.zip'),(latex_source,'latex/LaTeX_source.zip'),
             (latex/'main.pdf','latex/main.pdf'),(evidence,'data/Verification.zip'),(raw,'data/Reproducibility_data.zip')]
    payload+=list(tree(ROOT/'code/gpu','code/gpu'))+list(tree(latex/'submission','latex/submission'))+list(tree(ROOT/'code/tools','code/tools'))
    payload += [(f,a) for f,a in tree(ROOT/'code/releases','code/releases') if re.search(r'becgpp-1\.2\.2(?![0-9a-z])',f.name)]
    final=make_zip('CPC_submission_package.zip',payload)
    (OUT/'SHA256.json').write_text(json.dumps({f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in [source,latex_source,evidence,raw,final]},indent=2)+'\n')
if __name__=='__main__':main()
