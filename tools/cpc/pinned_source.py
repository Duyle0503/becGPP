"""Check source hashes recorded for an earlier commit against a later checkout.

Evidence runs pin the exact source they executed. After a release, files may
have changed, so a hash is accepted if it matches either the working tree or
the git object of the pinned commit (``git show <commit>:<path>``). Without a
git checkout, captured historical files in data/history/pinned_sources are
checked against the original run's expected hashes. Line endings are
normalized to LF. A snapshot is evidence of captured bytes, not a Git signature.
"""
import hashlib
import subprocess
import json
import re
from pathlib import Path


def _norm_sha(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def verify_pinned(repo, commit, expected):
    """Return counts of hashes confirmed in the working tree and via git."""
    repo = Path(repo)
    has_git = (repo / ".git").exists()
    assert re.fullmatch(r'[0-9a-f]{40}', commit), 'Use the full historical commit'
    counts = dict(working_tree=0, git_object=0, historical_snapshot=0)
    for name, value in expected.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and ':' not in name
        path = repo / name
        if path.is_file() and _norm_sha(path.read_bytes()) == value:
            counts["working_tree"] += 1
            continue
        if not has_git:
            snapshot = repo.parents[1] / 'data/history/pinned_sources' / commit
            record = json.loads((snapshot / 'SNAPSHOT.json').read_text())
            assert record['commit'] == commit
            entry = record['files'][name]
            blob = (snapshot / name).read_bytes()
            assert _norm_sha(blob) == value == entry['normalized_sha256'], name
            blob_id = hashlib.sha1(b'blob ' + str(len(blob)).encode() + b'\0' + blob).hexdigest()
            assert blob_id == entry['git_blob_sha1'], name
            counts['historical_snapshot'] += 1
            continue
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        assert _norm_sha(blob) == value, f"{name}: pinned hash does not match commit {commit[:12]}"
        counts["git_object"] += 1
    return counts
