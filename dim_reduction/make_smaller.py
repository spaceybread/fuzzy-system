from sklearn.decomposition import PCA
from collections import defaultdict
import numpy as np
import random

def load_data(npz_file, text_file, save_path, idx, T, R, new_dim): 
    npz = np.load(npz_file)
    ids = [x.strip().split('/')[-idx] for x in open(text_file)]

    if (len(ids) != npz.shape[0]): print("Size Mismatch")

    grp = defaultdict(list)
    for s, v in zip(ids, npz): grp[s].append(v)
    
    rem = []

    for k in grp.keys(): 
        if len(grp[k]) < T + R: rem.append(k)

    for r in rem: grp.pop(r)
    
    # for each key in grp, grp[k] contains
    # about 18 or so vectors

    all_vecs = np.vstack([np.array(vs) for vs in grp.values()])

    pca = PCA(n_components=new_dim)
    pca.fit(all_vecs)

    reduced_grp = {}
    for k, vecs in grp.items():
        vecs = np.array(vecs)
        reduced_grp[k] = pca.transform(vecs)

    np.save(save_path, reduced_grp)


def make_datasets(npz_file, save_path_rad, save_path_lat, T, R):
    data = np.load(npz_file, allow_pickle=True).item()
    all_keys = list(data.keys())
    
    ma, mb = {}, {}

    for x in data.keys():
        try: booga_boo = random.sample(data[x], k = T + R)
        except:
            print(len(data[x]), R, T)
            continue
        
        registration_vecs = booga_boo[:R]
        intra_vecs = booga_boo[R:]

        inter_vecs = []
        inter_key = x

        for i in range(T):
            while inter_key == x: inter_key = random.choices(all_keys, k = 1)[0]
            inter_vecs.append(random.choices(data[inter_key], k = 1)[0])

        center = np.mean(registration_vecs, axis = 0)
        init_rad = np.max([np.linalg.norm(ve - center) for ve in registration_vecs])
        
        in_dist = [np.linalg.norm(ve - center) for ve in intra_vecs]
        out_dist = [np.linalg.norm(ve - center) for ve in inter_vecs]

        ma[x] = (init_rad, in_dist, out_dist)
        mb[x] = (center, init_rad, intra_vecs, inter_vecs)

    np.save(save_path_rad, ma)
    np.save(save_path_lat, mb)
    
