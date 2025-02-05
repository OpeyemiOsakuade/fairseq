import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D
import umap

# Load embeddings and labels from the .npy files
embeddings = np.load('/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/w2v2_all_vectors/syllables_train_dev_yor_features_last_layer.npy', allow_pickle=True)
labels = np.load('/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/yor_frames/HF_optimized/w2v2_all_vectors/syllables_train_dev_yor_plain_vowels.npy', allow_pickle=True)

# Ensure labels are a 1D NumPy array
labels = np.array(labels)
labels = np.squeeze(labels)
print("Labels shape after squeeze:", labels.shape)

# Process each embedding to get a fixed-length vector (mean pooling)
processed_embeddings = []
for idx, emb in enumerate(embeddings):
    # Mean pooling over time steps to get a fixed-size vector
    emb_mean = emb.mean(axis=0)  # Shape: (feature_dim,)
    processed_embeddings.append(emb_mean)

# Convert the list of processed embeddings to a NumPy array
embeddings_array = np.vstack(processed_embeddings)  # Shape: (num_samples, feature_dim)
print("Embeddings_array shape:", embeddings_array.shape)

# Check if the number of embeddings matches the number of labels
if embeddings_array.shape[0] != labels.shape[0]:
    raise ValueError("Number of embeddings and labels must match.")
# =============================================================================
# Specify the labels you want to plot
# Set 'desired_labels' to None to plot all labels, or provide a list of labels to plot only those
# Example: desired_labels = ['a', 'e', 'i', 'o', 'u'] to plot only those vowels
# =============================================================================
# desired_labels = None  # Replace with a list of labels or keep as None to plot all labels
# desired_labels = ['a', 'e', 'i', 'o', 'u']5X5
desired_labels = ['a','e','eR','i','o','oR','u'] 
# desired_labels = ['aM','aH','aL','eM','eH','eL','eRM','eRH','eRL','iM','iH','iL','oM','oH','oL','oRM','oRH','oRL','u','uH','uL'] 
if desired_labels is not None:
    mask = np.isin(labels, desired_labels)
    embeddings_filtered = embeddings_array[mask]
    filtered_labels = labels[mask]
else:
    embeddings_filtered = embeddings_array
    filtered_labels = labels

# Optional: Normalize the embeddings for better performance of UMAP
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings_filtered)

# Encode labels to integers using LabelEncoder
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(filtered_labels)

# Number of unique labels
num_labels = len(label_encoder.classes_)


# # Define a list of colors for plotting
# # colors = ['red', 'green', 'blue']
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'brown']#, 'gray', 'cyan', 'magenta', 'black']
# # colors = ['red','blue','green','yellow','orange','purple','pink',
# #           'brown','black','crimson','gray','cyan','magenta','teal',
# #           'violet','indigo','maroon','olive','turquoise','gold','emerald']


# Generate 21 unique colors using 'gist_ncar' colormap
# cmap = plt.cm.get_cmap('gist_ncar', num_labels)
# colors = [cmap(i) for i in range(num_labels)]

# Define UMAP parameters
n_neighbors = 15
min_dist = 0.1
random_state = 42

# Apply UMAP for dimensionality reduction
umap_reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    n_components=2,  # Reduce to 2 dimensions
    random_state=random_state
)
embeddings_2d = umap_reducer.fit_transform(embeddings_normalized)

# Ensure there are enough colors for the unique labels
if len(colors) < len(label_encoder.classes_):
    raise ValueError("Not enough colors specified for the number of labels.")

# Map label IDs to colors
color_map = [colors[label_id] for label_id in label_ids]

# Plot the embeddings
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embeddings_2d[:, 0],
    embeddings_2d[:, 1],
    color=color_map,  # Assign colors to each point
    alpha=0.8,
    edgecolors='k',
    linewidths=0.5
)

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=label,
           markerfacecolor=color, markersize=10, markeredgecolor='k')
    for label, color in zip(label_encoder.classes_, colors)
]

# Set the legend title based on whether specific labels are plotted
legend_title = 'Labels' if desired_labels is None else 'Specified Labels'

plt.legend(
    handles=legend_elements,
    title=legend_title,
    loc='best'
)

# Set the plot title
plot_title = 'UMAP w2v2 Embeddings of Yoruba vowels without tone'
if desired_labels is not None:
    plot_title += ' (Specified Labels)'

plt.title(plot_title)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.tight_layout()

# Function to generate the filename based on UMAP parameters
def generate_filename(base_name, params, extension='png'):
    params_str = '_'.join([f'{key}{value}' for key, value in params.items()])
    filename = f'{base_name}_{params_str}.{extension}'
    return filename

# Construct the filename
params = {
    'n_neighbors': n_neighbors,
    'min_dist': min_dist,
    'rs': random_state
}
filename = generate_filename('UMAP_w2v2_Yoruba_vowel_without_tone_embeddings', params)

# Save the plot to a file
plt.savefig(filename, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
