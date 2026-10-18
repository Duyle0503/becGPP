"""Check source hashes recorded for an earlier commit against a later checkout.

Evidence runs pin the exact source they executed. After a release, files may
have changed, so a hash is accepted if it matches either the working tree or
the git object of the pinned commit (``git show <commit>:<path>``). Without a
git checkout, only working-tree matches can be confirmed. Line endings are
normalized to LF in both cases.
"""
import hashlib
import subprocess
from pathlib import Path


def _norm_sha(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def verify_pinned(repo, commit, expected):
    """Return counts of hashes confirmed in the working tree and via git."""
    repo = Path(repo)
    has_git = (repo / ".git").exists()
    counts = dict(working_tree=0, git_object=0)
    for name, value in expected.items():
        path = repo / name
        if path.is_file() and _norm_sha(path.read_bytes()) == value:
            counts["working_tree"] += 1
            continue
        if not has_git:
            raise AssertionError(f"{name}: differs from pinned commit {commit[:12]} and no git "
                                 "history is available to confirm the pinned content")
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
        assert _norm_sha(blob) == value, f"{name}: pinned hash does not match commit {commit[:12]}"
        counts["git_object"] += 1
    return counts
