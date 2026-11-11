"""Train your framework on the three development datasets and build the leaderboard zip.

    python make_submission.py path/to/folder_with_development_csvs [--out submission.zip]

Steps:
  1. warns if your Python or library versions differ from the leaderboard's (requirements.txt);
  2. for each problem, fits StudentSolution from solution.py on the full development data and saves the
     fitted object as pN_model.joblib (it must contain every preprocessing step);
  3. saves the P3 threshold from get_threshold() in p3_threshold.json and the versions in manifest.json;
  4. zips solution.py, your other .py helper files and the files above;
  5. checks the zip the way the leaderboard does: loads each model in a fresh Python process and predicts on
     raw development features.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import import_module

import joblib
import numpy as np
import pandas as pd

from problems import PROBLEMS, metadata_for
from solution import StudentSolution

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_FILES = {"make_submission.py", "local_test.py", "problems.py"}
# Distribution name -> import name, for the libraries whose versions must match the leaderboard.
CHECKED = {"numpy": "numpy", "pandas": "pandas", "scikit-learn": "sklearn", "lightgbm": "lightgbm",
           "xgboost": "xgboost", "catboost": "catboost", "imbalanced-learn": "imblearn"}


def local_versions():
    versions = {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    for dist, module in CHECKED.items():
        try:
            versions[dist] = import_module(module).__version__
        except ImportError:
            pass
    return versions


def required_versions():
    req = {}
    with open(os.path.join(HERE, "requirements.txt")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "==" in line:
                name, ver = line.split("==")
                req[name.replace("xgboost-cpu", "xgboost")] = ver
    return req


def check_versions(versions):
    req = required_versions()
    wrong = [f"{lib} {versions[lib]} (need {req[lib]})" for lib in CHECKED if lib in versions and lib in req
             and versions[lib] != req[lib]]
    if not versions["python"].startswith("3.11."):
        wrong.append(f"python {versions['python']} (need 3.11)")
    if wrong:
        print("WARNING: version mismatch: " + "; ".join(wrong) + ".\nThe leaderboard still scores your "
              "submission, but saved models can fail to load or predict differently there. Recommended: "
              "pip install -r requirements.txt  (in a Python 3.11 environment)\n")


SELF_CHECK = r"""
import json, os, sys, zipfile, tempfile
import joblib, numpy as np, pandas as pd
zip_path, data_dir, problems = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
d = tempfile.mkdtemp()
zipfile.ZipFile(zip_path).extractall(d)
sys.path.insert(0, d)
import solution
sys.modules["__main__"] = solution
for prob in problems:
    df = pd.read_csv(os.path.join(data_dir, prob["file"])).sample(500, random_state=0)
    X = df.drop(columns=[prob["target"], "record_id"])
    model = joblib.load(os.path.join(d, prob["key"] + "_model.joblib"))
    p = np.asarray(model.predict_proba(X, prob["metadata"]), dtype=float)
    p = p[:, 1] if p.ndim == 2 else p
    assert p.shape == (len(X),), f"{prob['key']}: predict_proba returned shape {p.shape}"
    assert np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1, f"{prob['key']}: probabilities not in [0, 1]"
t = json.load(open(os.path.join(d, "p3_threshold.json")))["threshold"]
assert 0 <= t <= 1
print("self-check passed")
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="folder containing the three development CSV files")
    ap.add_argument("--out", default="submission.zip")
    a = ap.parse_args()

    versions = local_versions()
    check_versions(versions)
    files = tempfile.mkdtemp(prefix="da5401_submission_")
    threshold = None
    for prob in PROBLEMS:
        df = pd.read_csv(os.path.join(a.data_dir, prob["file"]))
        X = df.drop(columns=[prob["target"], "record_id"])
        y = df[prob["target"]].astype(int)
        metadata = metadata_for(prob["objective"])
        t0 = time.time()
        sol = StudentSolution().fit(X, y, dict(metadata))
        joblib.dump(sol, os.path.join(files, f"{prob['key']}_model.joblib"), compress=3)
        print(f"{prob['key']}: trained and saved in {time.time() - t0:.0f} s")
        if metadata["requires_threshold"]:
            threshold = float(sol.get_threshold())
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                sys.exit(f"get_threshold() returned {threshold}; it must be between 0 and 1")
    with open(os.path.join(files, "p3_threshold.json"), "w") as f:
        json.dump({"threshold": threshold}, f)
    with open(os.path.join(files, "manifest.json"), "w") as f:
        json.dump({"versions": versions, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)

    code = sorted(n for n in os.listdir(HERE) if n.endswith(".py") and n not in KIT_FILES)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in code:
            z.write(os.path.join(HERE, n), n)
        for n in sorted(os.listdir(files)):
            z.write(os.path.join(files, n), n)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out} ({size_mb:.1f} MB): " + ", ".join(code + sorted(os.listdir(files))))

    probs = [{**p, "metadata": metadata_for(p["objective"])} for p in PROBLEMS]
    check = subprocess.run([sys.executable, "-c", SELF_CHECK, os.path.abspath(a.out), os.path.abspath(a.data_dir),
                            json.dumps(probs)], capture_output=True, text=True, cwd=tempfile.gettempdir())
    if check.returncode != 0:
        sys.exit("Self-check FAILED (the leaderboard would reject this zip):\n" + check.stderr[-2000:])
    print(check.stdout.strip() + f": upload {a.out} on the My Submissions tab")


if __name__ == "__main__":
    main()
