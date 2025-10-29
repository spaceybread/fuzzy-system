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

# radial classifier
script_path = "optimizers/run_radial_classifier.py"
args = [rad_ds, results_folder + "/radial_classifier.csv"]

cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
    
# gauss no stack classifier
script_path = "optimizers/run_gauss_no_stack.py"
args = [lat_ds, results_folder + "/gauss_no_stack_classifier.csv"]

cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 1 stack classifier
script_path = "optimizers/run_gauss_1_stack.py"
args = [lat_ds, results_folder + "/gauss_1_stack_classifier.csv"]

cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 2 stack classifier
script_path = "optimizers/run_gauss_2_stack.py"
args = [lat_ds, results_folder + "/gauss_2_stack_classifier.csv"]

cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
