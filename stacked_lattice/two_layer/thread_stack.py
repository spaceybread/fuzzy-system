import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def get_data(npz_file): return np.load(npz_file, allow_pickle=True).item()

def round_scaled(vec, scale): return np.round(vec / scale) * scale

def random_vector(n): return np.random.uniform(0, 100, size=n)

def gen(vec, scale):
    rdm = round_scaled(random_vector(len(vec)), scale)
    helper = rdm - vec
    return helper, rdm

def recov(helper, vec, scale):
    return round_scaled(helper + vec, scale)

from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing as mp

# Assume gen(), recov(), round_scaled(), etc. are defined as before.

def _check_pairs_chunk(chunk, c_vec, q_vec, n1_scale, OFFSET):
    """Worker: checks a batch of (i, j) pairs. Returns True if any match."""
    for (i, j) in chunk:
        # Prepare 4 variants for this (i, j)
        base = np.copy(c_vec)
        variants = np.stack([
            base + np.eye(len(c_vec))[i] * OFFSET + np.eye(len(c_vec))[j] * OFFSET,  # aa
            base + np.eye(len(c_vec))[i] * OFFSET - np.eye(len(c_vec))[j] * OFFSET,  # ab
            base - np.eye(len(c_vec))[i] * OFFSET + np.eye(len(c_vec))[j] * OFFSET,  # ba
            base - np.eye(len(c_vec))[i] * OFFSET - np.eye(len(c_vec))[j] * OFFSET   # bb
        ])

        # Vectorized eval for these 4 variants
        for x in variants:
            helper, a = gen(x, n1_scale)
            b = recov(helper, q_vec, n1_scale)
            if np.array_equal(a, b):
                return True
    return False


import numpy as np

def match(c_vec, q_vec, scale):
    OFFSET = scale / 2

    # --- Base check ---
    helper, a = gen(c_vec, scale)
    b = recov(helper, q_vec, scale)
    if np.array_equal(a, b):
        return True

    # --- First layer ---
    n0_scale = scale * 0.95
    n1_scale = scale * 0.80

    # Try single-offsets vectorized
    offsets = np.eye(len(c_vec)) * OFFSET
    variants = np.concatenate([c_vec + offsets, c_vec - offsets])
    helpers, A = zip(*(gen(v, n0_scale) for v in variants))
    B = [recov(h, q_vec, n0_scale) for h in helpers]
    if any(np.array_equal(a, b) for a, b in zip(A, B)):
        return True

    # --- Danger zone (pairwise) ---
    n = len(c_vec)
    i_idx, j_idx = np.triu_indices(n, k=1)

    # Build all 4 offset combinations for every pair
    signs = np.array([[+1, +1], [+1, -1], [-1, +1], [-1, -1]])
    total = len(i_idx) * 4

    # Base copy
    batch = np.repeat(c_vec[None, :], total, axis=0)

    # Fill indices
    pair_indices = np.repeat(np.arange(len(i_idx)), 4)
    i_all, j_all = i_idx[pair_indices], j_idx[pair_indices]
    s_all = np.tile(signs, (len(i_idx), 1))

    batch[np.arange(total), i_all] += s_all[:, 0] * OFFSET
    batch[np.arange(total), j_all] += s_all[:, 1] * OFFSET

    # Vectorized generation and recovery
    # (if gen/recov accept batches, otherwise loop once here)
    results = []
    for x in batch:
        helper, a = gen(x, n1_scale)
        b = recov(helper, q_vec, n1_scale)
        results.append(np.array_equal(a, b))

    return any(results)


def run_bin_search(data, alpha): 
    hi, lo = 1, 0

    keys = list(data.keys())
    res = {}

    for _ in tqdm(range(16)): 
        tchk, fchk = 0, 0
        tks, fks = 0, 0
        
        coeff = (hi + lo) / 2

        for key in tqdm(keys):
            rad = data[key][1] * coeff
            cen = data[key][0]

            tchk += sum([1 if match(cen, val, rad) else 0 for val in data[key][2]])
            tks += len(data[key][2])
            fchk += sum([1 if match(cen, val, rad) else 0 for val in data[key][3]])
            fks += len(data[key][3])

        tmr, fmr = tchk / tks, fchk / fks
        res[coeff] = [tmr, fmr]

        if tmr > alpha: hi = coeff
        else: lo = coeff

    return res, coeff

def run_sweep(data, save_path): 
    res_ma = {"coeff": [], "TMR": [], "FMR": []}

    for i in range(70, 101, 40):
        resdb, idx = run_bin_search(data, i / 100)
        
        res_ma["coeff"].append(idx)
        res_ma["TMR"].append(resdb[idx][0])
        res_ma["FMR"].append(resdb[idx][1])
    
    print("Done! Check:", save_path)
    pd.DataFrame.from_dict(res_ma, orient='columns').to_csv(save_path, index=False)

def main(): 
    src_file = sys.argv[1]
    dst_file = sys.argv[2]

    data = get_data(src_file)
    run_sweep(data, dst_file)

if __name__ == "__main__": main()
