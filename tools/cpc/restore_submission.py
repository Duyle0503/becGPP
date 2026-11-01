"""Restore nested CPC archives with path, CRC and per-file SHA256 checks.

Run RESTORE.py after extracting the master archive, or use --bundle PATH.
The large release dataset is stored once, as the original GPU ZIP.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile


def stream_sha(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def restore(archive,dest,manifest_name,overlay_readme=False):
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        records=json.loads(z.read(manifest_name))
        assert set(z.namelist())==set(records)|{manifest_name}
        for info in z.infolist():
            name=info.filename
            assert not Path(name).is_absolute() and '..' not in Path(name).parts and ':' not in name and '\\' not in name
            assert not stat.S_ISLNK(info.external_attr>>16)
            target=(dest/name).resolve()
            assert target.is_relative_to(dest.resolve())
            if name in records:
                with z.open(name) as f:assert stream_sha(f)==records[name],name
            if target.exists():
                with target.open('rb') as f:current=stream_sha(f)
                expected=records[name] if name in records else hashlib.sha256(z.read(name)).hexdigest()
                if current!=expected and not (overlay_readme and name=='data/README.md'):
                    raise RuntimeError('Refusing to overwrite a different existing file: '+str(target))
            z.extract(info,dest)
        for name,value in records.items():
            with (dest/name).open('rb') as f:assert stream_sha(f)==value,name
    print('Restored',archive.name,len(records),'verified files',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();root=args.bundle.resolve()
    components=[('code/Software_source.zip',root,'CHECKSUMS/Software_source.json',False),
                ('latex/LaTeX_source.zip',root,'CHECKSUMS/LaTeX_source.json',False),
                ('data/Reproducibility_data.zip',root,'CHECKSUMS/Reproducibility_data.json',False),
                ('data/raw/archives/becgpp_trapped3d_v2_5d8306c9b9eb.zip',root/'data/raw/trapped3d_release_2026_10_31','SHA256.json',False),
                ('data/Verification.zip',root,'CHECKSUMS/Verification.json',True)]
    estimated=0
    for name,_,_,_ in components:
        with zipfile.ZipFile(root/name) as z:estimated+=sum(i.file_size for i in z.infolist())
    if shutil.disk_usage(root).free < estimated+128*1024**2:
        raise RuntimeError('Not enough disk space to restore all components; use a drive with at least '+str(estimated+128*1024**2)+' free bytes')
    for name,dest,manifest,overlay in components:
        dest.mkdir(parents=True,exist_ok=True);restore(root/name,dest,manifest,overlay)
    print('Build latex/main.tex, then run python code/tools/verify_final_project.py. No Git history is required.')


if __name__=='__main__':main()
