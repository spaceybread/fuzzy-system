import make_test_data

data_dir = "/Users/home/Documents/fuzzy-system/testing/datasets/voice_data/"
src_dir = "/Users/home/Documents/curly-waffle/voice_data/"

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "filenames.txt"

Tl, Tu, R = 5, 10, 8

mapped_data = data_dir + "intermediate.npy"
radial_data = data_dir + "radial_ds.npy"
lat_data = data_dir + "lat_ds.npy"

make_test_data.load_data(npz_file, text_file, mapped_data, 3, R, Tl, Tu)
make_test_data.make_datasets(mapped_data, radial_data, lat_data, Tl, R)
