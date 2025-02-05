import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load embeddings and labels from the .npy files
embeddings = np.load('/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/train_man_features_last_layer.npy', allow_pickle=True)
labels = np.load('/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/train_man_plain_vowels.npy', allow_pickle=True)

# Ensure labels are a 1D NumPy array
labels = np.array(labels)
labels = np.squeeze(labels)

# Process each embedding to get a fixed-length vector (mean pooling)
processed_embeddings = [emb.mean(axis=0) for emb in embeddings]
embeddings_array = np.vstack(processed_embeddings)

# Filter the data based on desired labels if specified
desired_labels = None  # Replace with a list of labels or keep as None to plot all labels
if desired_labels is not None:
    mask = np.isin(labels, desired_labels)
    embeddings_filtered = embeddings_array[mask]
    filtered_labels = labels[mask]
else:
    embeddings_filtered = embeddings_array
    filtered_labels = labels

# Normalize the embeddings for better performance
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings_filtered)

# Encode labels to integers
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(filtered_labels)

# Apply 3D t-SNE for dimensionality reduction
tsne = TSNE(
    n_components=3,  # Reduce to 3 dimensions
    perplexity=30,
    learning_rate=200,
    random_state=42
)
embeddings_3d = tsne.fit_transform(embeddings_normalized)

# Create a 3D scatter plot using Plotly
fig = px.scatter_3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    color=filtered_labels,
    labels={'color': 'Label'},
    title='3D t-SNE HuBERT_baseEmbeddings of Mandarin vowels without tone'
)
fig.update_traces(marker=dict(size=3, opacity=0.8))

# Save the plot as an HTML file
fig.write_html('3D_tSNE_HuBERT_baseMandarin_vowels_without_tone.html')

# Display the plot
fig.show()
