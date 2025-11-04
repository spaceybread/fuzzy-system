import numpy as np
from itertools import combinations

def generate_vectors_np(n, k):
    if k > n or k < 0: return np.empty((0, n), dtype=int)
    if k == 0: return np.zeros((1, n), dtype=int)
    
    combs = np.array(list(combinations(range(n), k)))
    
    signs = np.array(np.meshgrid(*([[-1, 1]] * k), indexing='ij')).reshape(k, -1).T

    result = np.zeros((len(combs) * len(signs), n), dtype=int)
    for i, comb in enumerate(combs):
        block = np.zeros((len(signs), n), dtype=int)
        block[:, comb] = signs
        result[i * len(signs):(i + 1) * len(signs)] = block
    return result

print(generate_vectors_np(5, 0))
print(generate_vectors_np(5, 1))
print(generate_vectors_np(5, 2))
