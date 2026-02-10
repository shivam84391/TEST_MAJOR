import h5py

file_path = "data/ACDC_training_slices/patient001_frame01_slice_0.h5"

with h5py.File(file_path, "r") as f:
    print("Available keys:", list(f.keys()))
