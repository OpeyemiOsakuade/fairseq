import numpy as np
import pandas as pd

# Step 1: Load the CSV file
csv_file_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/spiral_mandarin/vowels.csv'  # Replace with your file path
df = pd.read_csv(csv_file_path)

# Step 2: Convert the DataFrame to a Numpy array
data_array = df.to_numpy()

# Step 3: Save the Numpy array as a .npy file
npy_file_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/spiral_mandarin/vowels.npy'  # Replace with your desired output path
np.save(npy_file_path, data_array)

print(f"Data saved as {npy_file_path}")
