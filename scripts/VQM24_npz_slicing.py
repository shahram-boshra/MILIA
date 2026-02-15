import os

import numpy as np

# Define the input and output file names
original_file = "/app/08/DFT_saddles.npz"  # <--- REMEMBER TO CHANGE THIS
output_file = "/app/08/DFT_saddles_sliced.npz"

# Define the number of values to keep per key
num_values_to_keep = 1000

# --- 1. Load the original large .npz file ---
try:
    print(f"Attempting to load the original file: {original_file}")
    # Using np.load with allow_pickle=True if your data contains Python objects
    # but generally for numerical data, it's not needed and can be a security risk
    data_original = np.load(original_file, allow_pickle=True)
    print("Original file loaded successfully.")
    print(f"Available keys in the original file: {list(data_original.keys())}")
except FileNotFoundError:
    print(
        f"Error: '{original_file}' not found. Please ensure the file is in the correct directory."
    )
    exit()
except Exception as e:
    print(f"An error occurred while loading the original file: {e}")
    exit()

# --- 2. Create a dictionary to hold the subsetted data ---
subset_data = {}
print(f"\nProcessing keys to keep {num_values_to_keep} values each:")

for key in data_original.keys():
    original_array = data_original[key]

    # Check if the array is large enough and has a dimension to slice
    # This assumes the values you want to reduce are along the first axis.
    # If your data is structured differently (e.g., you need to slice a 2D array
    # along its second dimension, or only certain elements from a 1D array),
    # you'll need to adjust the slicing logic.
    if original_array.ndim >= 1 and original_array.shape[0] >= num_values_to_keep:
        # Slice the array to keep only the first 'num_values_to_keep' values
        subset_array = original_array[:num_values_to_keep]
        subset_data[key] = subset_array
        print(
            f"  - Key '{key}': Original shape {original_array.shape}, New shape {subset_array.shape}"
        )
    else:
        # If the array is smaller than desired, or not suitable for this kind of slicing,
        # you might choose to include it fully or skip it, depending on your needs.
        # For this scenario, let's include it fully if it's smaller,
        # assuming you want all keys present.
        subset_data[key] = original_array
        print(
            f"  - Key '{key}': Array too small ({original_array.shape[0]} values) or not suitable for slicing to {num_values_to_keep}. Including full array."
        )

# --- 3. Save the subsetted data to a new .npz file ---
try:
    print(f"\nSaving the subsetted data to '{output_file}'...")
    np.savez(output_file, **subset_data)
    print(f"Smaller .npz file '{output_file}' created successfully.")
except Exception as e:
    print(f"An error occurred while saving the new file: {e}")

# --- 4. Verify the size (optional) ---
if os.path.exists(output_file):
    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Size of the new file: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
    # You can add a check here if the size is still too large
    # For example, if you aim for exactly 100 MB, this will vary based on data types
    # and compression.
else:
    print("New file not found for size verification.")

# --- 5. Clean up (optional, important for large files) ---
data_original.close()  # Close the loaded npz file to free resources
print("Original .npz file closed.")
