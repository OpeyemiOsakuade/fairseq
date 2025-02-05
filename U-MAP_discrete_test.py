import json
import numpy as np
import umap
import matplotlib.pyplot as plt

# Function to load data from a JSON file
def load_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

# Load data from JSON file (replace 'data.json' with your actual file path)
json_file_path = '/disk/ostrom/homedirs/s2324992/tonal_data/data/mandarin/models/hubert_9/for_phones/0.1_hu_manl9c200_phoneme_reps.json'
data = load_data(json_file_path)

# Prepare data for UMAP
vowels = []
reps = []

for vowel, content in data.items():
    for rep in content["speech_reps"]:
        vowels.append(vowel)
        reps.append(rep)

# Normalize the lengths of speech representations by padding with zeros to the max length
max_length = max(len(rep) for rep in reps)
padded_reps = [rep + [0] * (max_length - len(rep)) for rep in reps]

# Convert to numpy array for UMAP
reps_array = np.array(padded_reps)

# Perform UMAP dimensionality reduction
umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2, random_state=42)
reps_umap = umap_model.fit_transform(reps_array)

# Plot the UMAP results
plt.figure(figsize=(8, 6))
for vowel in set(vowels):
    indices = [i for i, v in enumerate(vowels) if v == vowel]
    plt.scatter(reps_umap[indices, 0], reps_umap[indices, 1], label=vowel, alpha=0.7)

plt.title("UMAP of Speech Representations by Vowel")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Vowel")
plt.grid(True)

# Save the plot
output_path = "umap_speech_representations.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved as {output_path}")
