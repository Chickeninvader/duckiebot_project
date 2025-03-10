import h5py
with h5py.File("/home/duckie/Desktop/simulation/robomimic/datasets/lift/ph/low_dim_v141.hdf5", "r") as f:
    print("Keys (Top Level Groups/Datasets):", list(f.keys()))
    for key in f.keys():
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            print(f"Dataset: {key}, Shape: {obj.shape}, Type: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"Group: {key}, Contains: {list(obj.keys())}")
