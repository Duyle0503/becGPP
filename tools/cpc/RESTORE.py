"""Run this file in an extracted CPC master package to restore its components."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
# This same file is also kept under code/tools in the working bundle.
if not (root/'code/tools/restore_submission.py').is_file():root=root.parents[1]
subprocess.run([sys.executable,str(root/'code/tools/restore_submission.py'),'--bundle',str(root)],check=True)
