import numpy as np

data = np.load("lat_ds.npy", allow_pickle=True).item()

keys = list(data.keys())
new_map = {}

for k in keys: 
    new_map[k] = (
            data[k][0][0], 
            data[k][1], 
            [a[0] for a in data[k][2]], 
            [a[0] for a in data[k][3]]
            )

np.save("lat_ds_2.npy", new_map)
