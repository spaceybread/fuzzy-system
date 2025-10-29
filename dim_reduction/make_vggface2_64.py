import make_smaller

data_dir = "/Users/home/Documents/fuzzy-system/full_tilt/datasets/vggface2-resnet_64/"
src_dir = "/Users/home/Documents/curly-waffle/face-data/vggface2-resnet/" 

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "files.txt"

T, R = 10, 8

mapped_data = data_dir + "intermediate.npy"
radial_data = data_dir + "radial_ds.npy"
lat_data = data_dir + "lat_ds.npy"

make_smaller.load_data(npz_file, text_file, mapped_data, 2, T, R, 64)

make_smaller.make_datasets(mapped_data, radial_data, lat_data, T, R)
