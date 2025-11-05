from pathlib import Path
import subprocess
import sys
import os

data_dir = sys.argv[1]

lat_ds = data_dir + "tests/radial_ds.npy"
rad_ds = data_dir + "tests/lat_ds.npy"

results_folder = data_dir + "tests/results"

try: os.mkdir(results_folder)
except FileExistsError: pass

COEFFS = [
0.9514329731464386, 1.1060214638710022,
0.70013427734375, 0.85980224609375,
0.6204080581665039, 0.7499685287475586,
0.5334930419921875, 0.6405792236328125
]


venv_python = Path("~/Desktop/.venv/bin/python3").expanduser()

# radial classifier 70
script_path = "run_radial_classifier.py"
args = [rad_ds, results_folder + "/radial_70.csv", str(COEFFS[0])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# radial classifier 90
script_path = "run_radial_classifier.py"
args = [rad_ds, results_folder + "/radial_90.csv", str(COEFFS[1])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
    
# gauss no stack classifier 70
script_path = "run_gauss_no_stack.py"
args = [lat_ds, results_folder + "/gauss_0_70.csv", str(COEFFS[2])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss no stack classifier 90
script_path = "run_gauss_no_stack.py"
args = [lat_ds, results_folder + "/gauss_0_90.csv", str(COEFFS[3])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 1 stack classifier 70
script_path = "run_gauss_1_stack.py"
args = [lat_ds, results_folder + "/gauss_1_70.csv", str(COEFFS[4])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 1 stack classifier 90
script_path = "run_gauss_1_stack.py"
args = [lat_ds, results_folder + "/gauss_1_90.csv", str(COEFFS[5])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 2 stack classifier 70
script_path = "run_gauss_2_stack.py"
args = [lat_ds, results_folder + "/gauss_2_70.csv", str(COEFFS[6])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)

# gauss 2 stack classifier 90
script_path = "run_gauss_2_stack.py"
args = [lat_ds, results_folder + "/gauss_2_90.csv", str(COEFFS[7])]
cmd = [str(venv_python), str(script_path)] + args
result = subprocess.run(cmd)
