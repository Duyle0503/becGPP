"""Capture the actual historical Git blobs needed for an offline evidence audit."""
import ast
import hashlib
import json
import subprocess
from workspace_paths import find_workspace

ROOT = find_workspace()
repo = ROOT / 'code/becGPP'
for stem, commit in [('GPU_CHECK_ONE_CELL','5f66bfc8635a1791078fdd503160e341fb12e82c'),
                     ('GPU_RETEST_ONLY','75f0a304acf2ff61dceb3c5ac4df10a9ef84b197')]:
    # GPU_CHECK_ONE_CELL was updated to the repaired regression commit. Use
    # the batch's immutable raw source manifest for its historical hashes.
    raw = ROOT / 'data/raw' / ('gpu_2026_10_17' if stem == 'GPU_CHECK_ONE_CELL' else 'gpu_retest_2026_10_17')
    source = json.loads((raw / 'source_manifest.json').read_text())
    expected = source['normalized_module_sha256']
    out = ROOT / 'data/history/pinned_sources' / commit
    entries = {}
    for name, value in expected.items():
        blob = subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=repo)
        norm = hashlib.sha256(blob.replace(b'\r\n',b'\n')).hexdigest()
        assert norm == value, name
        blob_id = subprocess.check_output(['git','rev-parse',f'{commit}:{name}'],cwd=repo,text=True).strip()
        assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest() == blob_id
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
        entries[name]=dict(normalized_sha256=norm,git_blob_sha1=blob_id)
    (out/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
    (out/'SNAPSHOT.json').write_text(json.dumps(dict(commit=commit,files=entries,
        provenance='Captured with git show from the local repository; normalized hashes match the original run manifest. Not a signed commit attestation.'),indent=2)+'\n')
    print(commit,len(entries),'historical files captured and checked')
