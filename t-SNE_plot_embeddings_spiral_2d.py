import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
import pickle

# Define the paths to your data files
embeddings_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/spiral_mandarin/features.pkl'
labels_vowels_with_tone_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/spiral_mandarin/vowels.pkl'
labels_vowels_without_tone_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/spiral_mandarin/plain_vowels.pkl'
labels_tones_only_path = '/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/spiral_mandarin/tones.pkl'

# Function to load data from pickle files
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load embeddings
embeddings = load_pickle(embeddings_path)

# Load labels
labels_vowels_with_tone = load_pickle(labels_vowels_with_tone_path)
labels_vowels_without_tone = load_pickle(labels_vowels_without_tone_path)
labels_tones_only = load_pickle(labels_tones_only_path)

# Ensure labels are numpy arrays and squeezed
labels_vowels_with_tone = np.squeeze(np.array(labels_vowels_with_tone))
labels_vowels_without_tone = np.squeeze(np.array(labels_vowels_without_tone))
labels_tones_only = np.squeeze(np.array(labels_tones_only))

# Check that the number of labels matches the number of embeddings
if not (len(embeddings) == len(labels_vowels_with_tone) == len(labels_vowels_without_tone) == len(labels_tones_only)):
    raise ValueError("Number of embeddings and labels must match for all label sets.")

# Process embeddings (mean pooling if necessary)
processed_embeddings = []
for emb in embeddings:
    # Check if emb is a 2D array (e.g., (time_steps, feature_dim))
    if isinstance(emb, np.ndarray) and emb.ndim == 2:
        emb_mean = emb.mean(axis=0)
        processed_embeddings.append(emb_mean)
    else:
        # If emb is already a 1D array, use it directly
        processed_embeddings.append(emb)

embeddings_array = np.vstack(processed_embeddings)

# Optional: Shuffle the data
# Combine embeddings and labels into a single list of tuples
combined = list(zip(embeddings_array, labels_vowels_with_tone, labels_vowels_without_tone, labels_tones_only))
np.random.shuffle(combined)
# Unzip the combined list back into separate arrays
embeddings_array, labels_vowels_with_tone, labels_vowels_without_tone, labels_tones_only = zip(*combined)
# Convert embeddings_array back to numpy array
embeddings_array = np.array(embeddings_array)

# Optional: Normalize the embeddings
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings_array)

# Apply t-SNE
perplexity = 30
learning_rate = 200
random_state = 42

tsne = TSNE(
    n_components=2,  # Set to 2 for 2D plotting
    perplexity=perplexity,
    learning_rate=learning_rate,
    random_state=random_state,
    init='pca',
    n_iter=1000
)
embeddings_2d = tsne.fit_transform(embeddings_normalized)

# Prepare to plot
fig, axs = plt.subplots(1, 3, figsize=(20, 6))
label_sets = [
    ('Vowels with Tone', labels_vowels_with_tone),
    ('Vowels without Tone', labels_vowels_without_tone),
    ('Tones Only', labels_tones_only)
]

for ax, (title, labels) in zip(axs, label_sets):
    # Encode labels
    label_encoder = LabelEncoder()
    labels = np.array(labels)
    label_ids = label_encoder.fit_transform(labels)

    # Number of unique labels
    num_labels = len(label_encoder.classes_)

    # Generate colors
    cmap = plt.cm.get_cmap('gist_ncar', num_labels)
    colors = [cmap(i) for i in range(num_labels)]

    # Map label IDs to colors
    color_map = [colors[label_id] for label_id in label_ids]

    # Plot
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        color=color_map,
        alpha=0.8,
        edgecolors='k',
        linewidths=0.5,
        s=30
    )

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=label,
               markerfacecolor=color, markersize=10, markeredgecolor='k')
        for label, color in zip(label_encoder.classes_, colors)
    ]

    ax.legend(
        handles=legend_elements,
        title=title,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        borderaxespad=0.
    )

    ax.set_title(title)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True)

plt.tight_layout()

# Save the plot
filename = 'SPIRAL_English_embeddings_of_Mandarin_vowel_and_tones_2d.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()
