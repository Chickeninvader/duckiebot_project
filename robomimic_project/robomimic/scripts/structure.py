import h5py
import os


def save_hdf5_structure(file_path):
    """
    Saves the structure of an HDF5 file to a text file in the same directory.

    Args:
        file_path (str): Path to the HDF5 file.
    """
    output_txt_path = os.path.splitext(file_path)[0] + "_structure.txt"

    def recursively_print(name, obj, file):
        file.write(name + "\n")
        if isinstance(obj, h5py.Group):
            file.write("  (Group)\n")
        elif isinstance(obj, h5py.Dataset):
            file.write(f"  (Dataset) Shape: {obj.shape}, Type: {obj.dtype}\n")
        else:
            file.write(f" Other stuff: {obj} \n")

    with h5py.File(file_path, "r") as f, open(output_txt_path, "w") as file:
        f.visititems(lambda name, obj: recursively_print(name, obj, file))

    print(f"Structure saved to {output_txt_path}")


# Example usage
save_hdf5_structure('/duckiebot_project/preprocess_data/record/converted_standard/all_demos_old.hdf5')
save_hdf5_structure('/duckiebot_project/preprocess_data/record/converted_standard/all_demos_new.hdf5')

# Example usage
# print_hdf5_demo_structure('/home/duckie/Desktop/simulation/robomimic/datasets/lift/ph/low_dim_v141.hdf5', demo_name="demo_1")

# save_hdf5_structure("/home/duckie/Desktop/simulation/robomimic/datasets/lift/ph/low_dim_v141.hdf5")

# import h5py

# def print_hdf5_structure_with_attrs(file_path):
#     def recursively_print(name, obj):
#         print(name)  # Prints dataset or group name
#         if isinstance(obj, h5py.Group):
#             print("  (Group)")
#         elif isinstance(obj, h5py.Dataset):
#             print(f"  (Dataset) Shape: {obj.shape}, Type: {obj.dtype}")

#         # Print attributes if available
#         for attr_name, attr_value in obj.attrs.items():
#             print(f"    [Attribute] {attr_name}: {attr_value}")

#     with h5py.File(file_path, "r") as f:
#         f.visititems(recursively_print)

# # Example usage
# print_hdf5_structure_with_attrs('/home/duckie/Desktop/simulation/robomimic/datasets/lift/ph/low_dim_v141.hdf5')
