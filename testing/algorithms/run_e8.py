from pathlib import Path
import subprocess
import sys
import os

data_dir = sys.argv[1]

lat_ds = data_dir + "tests/lat_ds.npy"
rad_ds = data_dir + "tests/radial_ds.npy"

results_folder = data_dir + "tests/results"

try: os.mkdir(results_folder)
except FileExistsError: pass

COEFFS = [
0.70477294921875, 0.86492919921875
]


venv_python = Path("~/Desktop/.venv/bin/python3").expanduser()

# e8 classifier 70
script_path = "run_e8_classifier.py"
args = [rad_ds, results_folder + "/e8_70.csv", str(COEFFS[0])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# e8 classifier 90
script_path = "run_e8_classifier.py"
args = [rad_ds, results_folder + "/e8_90.csv", str(COEFFS[1])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
    
