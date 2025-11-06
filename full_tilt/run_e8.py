from pathlib import Path
import subprocess
import sys
import os

data_dir = sys.argv[1]

lat_ds = data_dir + "/lat_ds.npy"
rad_ds = data_dir + "/radial_ds.npy"

results_folder = data_dir + "/results"

try: os.mkdir(results_folder)
except FileExistsError: pass

venv_python = Path("~/Desktop/.venv/bin/python3").expanduser()

# gauss 1 stack classifier
script_path = "optimizers/run_e8_classifier.py"
args = [lat_ds, results_folder + "/e8_classifier_70.csv"]

cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 2 stack classifier
script_path = "optimizers/run_e8_classifier_90.py"
args = [lat_ds, results_folder + "/e8_classifier_90.csv"]

cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
