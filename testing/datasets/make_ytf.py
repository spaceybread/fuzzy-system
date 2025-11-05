import make_smaller

data_dir = "/Users/home/Documents/fuzzy-system/testing/datasets/ytf_64/"
src_dir = "/Users/home/Documents/curly-waffle/face-data/ytf/" 

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "files.txt"

Tl, Tu, R = 5, 100, 8

mapped_data = data_dir + "intermediate.npy"
radial_data = data_dir + "radial_ds.npy"
lat_data = data_dir + "lat_ds.npy"

make_smaller.load_data(npz_file, text_file, mapped_data, 3, Tl, Tu, R, 64)

make_smaller.make_datasets(mapped_data, radial_data, lat_data, Tl, R)
