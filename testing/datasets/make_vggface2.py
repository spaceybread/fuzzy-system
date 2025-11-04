import make_smaller

data_dir = "/Users/home/Documents/fuzzy-system/testing/datasets/vggface2_128/"
src_dir = "/Users/home/Documents/curly-waffle/face-data/vggface2-resnet/" 

npz_file = src_dir + "embeddings.npy"
text_file = src_dir + "files.txt"

Tl, Tu, R = 5, 10, 8

mapped_data = data_dir + "intermediate.npy"
radial_data = data_dir + "radial_ds.npy"
lat_data = data_dir + "lat_ds.npy"

make_smaller.load_data(npz_file, text_file, mapped_data, 2, Tl, Tu, R, 128)

make_smaller.make_datasets(mapped_data, radial_data, lat_data, Tl, R)
