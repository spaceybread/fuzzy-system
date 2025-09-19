import make_intermediate_data

data_dir = "/Users/home/Documents/fuzzy-system/data_processing/vggface2-resnet/"
src_dir = "/Users/home/Documents/curly-waffle/face-data/vggface2-resnet/" 

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "files.txt"

T, R = 10, 8

mapped_data = data_dir + "intermediate.npy"
radial_data = data_dir + "radial_ds.npy"
lat_data = data_dir + "lat_ds.npy"

make_intermediate_data.load_data(npz_file, text_file, mapped_data, 2, T, R)
make_intermediate_data.make_datasets(mapped_data, radial_data, lat_data, T, R)
