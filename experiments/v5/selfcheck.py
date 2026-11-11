"""Run the official kit's SELF_CHECK on an existing zip.  usage: python selfcheck.py <kit dir> <zip> <data dir>"""
import sys, os, json, subprocess, tempfile
kit, zp, data = sys.argv[1:4]
sys.path.insert(0, kit)
src = open(os.path.join(kit, "make_submission.py")).read()
SELF_CHECK = src.split('SELF_CHECK = r"""')[1].split('"""')[0]
from problems import PROBLEMS, metadata_for
probs = [{**p, "metadata": metadata_for(p["objective"])} for p in PROBLEMS]
r = subprocess.run([sys.executable, "-c", SELF_CHECK, os.path.abspath(zp), os.path.abspath(data), json.dumps(probs)],
                   capture_output=True, text=True, cwd=tempfile.gettempdir())
print(r.stdout.strip() or r.stderr[-1500:])
