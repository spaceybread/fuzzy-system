import numpy as np
import sys 

def repair_test_splits(full_rad_file, full_lat_file, train_rad_file, train_lat_file, save_test_rad, save_test_lat):

    full_rad = np.load(full_rad_file, allow_pickle=True).item()
    full_lat = np.load(full_lat_file, allow_pickle=True).item()
    train_rad = np.load(train_rad_file, allow_pickle=True).item()
    train_lat = np.load(train_lat_file, allow_pickle=True).item()

    test_keys_rad = list(set(full_rad.keys()) - set(train_rad.keys()))
    test_keys_lat = list(set(full_lat.keys()) - set(train_lat.keys()))

    if len(test_keys_rad) == 0 or len(test_keys_lat) == 0:
        print("⚠️ No test keys found — training set may cover all data.")
        return

    test_rad = {k: full_rad[k] for k in test_keys_rad if k in full_rad}
    test_lat = {k: full_lat[k] for k in test_keys_lat if k in full_lat}

    np.save(save_test_rad, test_rad)
    np.save(save_test_lat, test_lat)

    print(f"✅ Repaired test splits created:")
    print(f"  Test radial: {len(test_rad)} entries saved → {save_test_rad}")
    print(f"  Test latent: {len(test_lat)} entries saved → {save_test_lat}")


i_rad_file = sys.argv[1]
i_lat_file = sys.argv[2]
train_rad = sys.argv[3]
train_lat = sys.argv[4]
test_rad = sys.argv[5]
test_lat = sys.argv[6]

repair_test_splits(i_rad_file, i_lat_file, train_rad, train_lat, test_rad, test_lat)
