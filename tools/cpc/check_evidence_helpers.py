"""Small structural checks for evidence readers. These do not run CUDA tests."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
from workspace_paths import find_workspace
from pinned_source import verify_pinned
from review_cuda_retest import test_ids

ROOT=find_workspace()
allowed=ROOT/'data/review/final_revision_2026_10_29'
allowed.mkdir(parents=True,exist_ok=True)
with tempfile.TemporaryDirectory(prefix='helper_checks_',dir=allowed) as name:
    tmp=Path(name).resolve()
    assert tmp.is_relative_to(allowed.resolve())
    source=ROOT/'data/raw/release_1_2_2_ci/pytest_cpu.xml'
    assert len(test_ids(source,109))==109
    rejected=[]
    for mutation in ('duplicate_id','failure_element','skip_count'):
        tree=ET.parse(source);root=tree.getroot();cases=root.findall('.//testcase')
        if mutation=='duplicate_id':cases[1].attrib.update(cases[0].attrib)
        elif mutation=='failure_element':ET.SubElement(cases[0],'failure')
        else:root.find('.//testsuite').set('skipped','1')
        bad=tmp/(mutation+'.xml');tree.write(bad)
        try:test_ids(bad,109)
        except AssertionError:rejected.append(mutation)
        else:raise AssertionError('Accepted corrupt XML: '+mutation)
    commit='5f66bfc8635a1791078fdd503160e341fb12e82c'
    expected=json.loads((ROOT/'data/raw/gpu_2026_10_17/source_manifest.json').read_text())['normalized_module_sha256']
    repo=tmp/'code/becGPP'
    for path in expected:
        dest=repo/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/'code/becGPP'/path).read_bytes())
    snapshot=tmp/'data/history/pinned_sources'/commit
    shutil.copytree(ROOT/'data/history/pinned_sources'/commit,snapshot)
    good=verify_pinned(repo,commit,expected)
    assert good['historical_snapshot']>0 and good['git_object']==0
    target=snapshot/'becgpp/solvers.py';target.write_bytes(target.read_bytes()+b'\n# altered\n')
    try:verify_pinned(repo,commit,expected)
    except AssertionError:rejected.append('altered_historical_source')
    else:raise AssertionError('Accepted altered historical bytes')
report=dict(scope='Structural validation only; not CUDA execution',valid_release_cpu_ids=109,
            corrupt_inputs_rejected=rejected,offline_source_confirmation=good)
(allowed/'HELPER_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
