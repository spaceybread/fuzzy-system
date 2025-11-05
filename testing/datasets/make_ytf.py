import make_smaller

data_dir = "/Users/home/Documents/fuzzy-system/testing/datasets/ytf_128/"
src_dir = "/Users/home/Documents/curly-waffle/face-data/ytf/" 

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "files.txt"

T, R = 10, 8

mapped_data = data_dir + "/inter/" + "intermediate.npy"
radial_data = data_dir + "/inter/" + "radial_ds.npy"
lat_data = data_dir + "/inter/" + "lat_ds.npy"

train_rad = data_dir + "/train/" + "radial_ds.npy"
train_lat = data_dir + "/train/" + "lat_ds.npy"
tests_rad = data_dir + "/tests/" + "radial_ds.npy"
tests_lat = data_dir + "/tests/" + "lat_ds.npy"

make_smaller.load_data(npz_file, text_file, mapped_data, 3, T, R, 128)

make_smaller.make_datasets(mapped_data, radial_data, lat_data, T, R)
make_smaller.split_datasets(radial_data, lat_data, train_rad, tests_lat, train_lat, tests_lat)
