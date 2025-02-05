import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D

# Load embeddings and labels from the .npy files
embeddings = np.load('/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/mandarinhu_train_man_features_last_layer.npy', allow_pickle=True)
labels = np.load('/disk/ostrom/homedirs/s2324992/tonal_data/tone_quantization/mandarin_frames/HF_no_syllables/all_vectors/train_man_vowels.npy', allow_pickle=True)


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
# Check the shapes
print("Embeddings_array shape:", embeddings_array.shape)
# print("Labels shape:", labels.shape)

# Check if the number of embeddings matches the number of labels
if embeddings_array.shape[0] != labels.shape[0]:
    raise ValueError("Number of embeddings and labels must match.")
# =============================================================================
# Specify the labels you want to plot
# Set 'desired_labels' to None to plot all labels, or provide a list of labels to plot only those
# Example: desired_labels = ['a', 'e', 'i', 'o', 'u'] to plot only those vowels
# =============================================================================
desired_labels = None  # Replace with a list of labels or keep as None to plot all labels
# desired_labels = ['a', 'e', 'i', 'o', 'u']5X5
# desired_labels = ['a','e','eR','i','o','oR','u'] 
# desired_labels = ['aM','aH','aL','eM','eH','eL','eRM','eRH','eRL','iM','iH','iL','oM','oH','oL','oRM','oRH','oRL','u','uH','uL'] 

# Filter the data based on desired labels if specified
if desired_labels is not None:
    # Create a boolean mask where True indicates the label is in desired_labels
    mask = np.isin(labels, desired_labels)
    print("Mask shape:", mask.shape)
    embeddings_filtered = embeddings_array[mask]
    filtered_labels = labels[mask]
else:
    # Use all data if no specific labels are provided
    embeddings_filtered = embeddings_array
    filtered_labels = labels

# Optional: Normalize the embeddings for better performance of t-SNE
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings_filtered)

# Encode labels to integers using LabelEncoder
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(filtered_labels)

# Number of unique labels
num_labels = len(label_encoder.classes_)

# Generate 21 unique colors using 'gist_ncar' colormap
cmap = plt.cm.get_cmap('gist_ncar', num_labels)
colors = [cmap(i) for i in range(num_labels)]

# Define t-SNE parameters
perplexity = 30
learning_rate = 200
random_state = 42

# Apply t-SNE for dimensionality reduction
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=learning_rate,
    random_state=random_state
)
embeddings_2d = tsne.fit_transform(embeddings_normalized)

# Define a list of colors for plotting
# colors = ['red', 'green', 'blue']
# colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'brown']#, 'gray', 'cyan', 'magenta', 'black']
# colors = ['red','blue','green','yellow','orange','purple','pink',
#           'brown','black','crimson','gray','cyan','magenta','teal',
#           'violet','indigo','maroon','olive','turquoise','gold','emerald']

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
plot_title = 'ManHuBERT Embeddings of Mandarin vowels with tone'
if desired_labels is not None:
    plot_title += ' (Specified Labels)'

plt.title(plot_title)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.tight_layout()

# Function to generate the filename based on t-SNE parameters
def generate_filename(base_name, params, extension='png'):
    params_str = '_'.join([f'{key}{value}' for key, value in params.items()])
    filename = f'{base_name}_{params_str}.{extension}'
    return filename

# Construct the filename
params = {
    'perp': perplexity,
    'lr': learning_rate,
    'rs': random_state
}
filename = generate_filename('ManHuBERT_Mandarin_vowel_with_tone_embeddings', params)

# Save the plot to a file
plt.savefig(filename, dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# # Ensure labels are a NumPy array
# labels = np.array(labels)

# # Specify the labels you want to plot
# desired_labels = ['a', 'e', 'i', 'o', 'u']  # Replace with your desired labels
# # Filter the data to include only the desired labels
# mask = np.isin(labels, desired_labels)
# filtered_labels = labels[mask]
# filtered_embeddings = embeddings[mask]
# # print("Embeddings type:", type(embeddings))
# # print("Embeddings shape:", embeddings.shape)
# # print("Embeddings dtype:", embeddings.dtype)

# # # print("First embedding:", embeddings[0])
# # print("Type of first embedding:", type(embeddings[0]))
# # print("Shape of first embedding:", np.shape(embeddings[0]))
# # print("Shape of 48th embedding:", np.shape(embeddings[47]))
# # print("Shape of 166th embedding:", np.shape(embeddings[165]))


# # Check that the number of embeddings matches the number of labels
# if embeddings.shape[0] != labels.shape[0]:
#     raise ValueError("Number of embeddings and labels must match.")

# # Process each embedding to get a fixed-length vector
# processed_embeddings = []
# for idx, emb in enumerate(embeddings):
#     # Mean pooling over time steps
#     emb_mean = emb.mean(axis=0)  # Shape: (feature_dim,)
#     processed_embeddings.append(emb_mean)

# # Convert the list to a NumPy array
# embeddings = np.vstack(processed_embeddings)  # Shape: (num_samples, feature_dim)

# # Optional: Normalize the embeddings
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# embeddings = scaler.fit_transform(embeddings)

# # Encode labels to integers if they are strings
# label_encoder = LabelEncoder()
# label_ids = label_encoder.fit_transform(labels)

# # Define t-SNE parameters
# perplexity = 30
# learning_rate = 200
# random_state = 42

# # Dimensionality reduction using t-SNE
# tsne = TSNE(
#     n_components=2,
#     perplexity=perplexity,
#     learning_rate=learning_rate,
#     random_state=random_state
# )
# embeddings_2d = tsne.fit_transform(embeddings)

# colors = ['red', 'green', 'blue', 'orange', 'purple','yellow','brown', 'gray','cyan', 'magenta','black']  # Adjust as needed
# color_map = [colors[label_id % len(colors)] for label_id in label_ids]

# # # Plot the embeddings
# # plt.figure(figsize=(10, 8))
# # scatter = plt.scatter(
# #     embeddings_2d[:, 0],
# #     embeddings_2d[:, 1],
# #     c=label_ids,
# #     color=color_map,
# #     alpha=0.8,
# #     edgecolors='k',
# #     linewidths=0.5
# # )
# # Plot the embeddings
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     embeddings_2d[:, 0],
#     embeddings_2d[:, 1],
#     color=color_map,  # Use custom colors
#     alpha=0.8,
#     edgecolors='k',
#     linewidths=0.5
# )

# # # Create a legend with vowel labels
# # handles, _ = scatter.legend_elements(num=len(label_encoder.classes_))
# # legend_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
# # plt.legend(
# #     handles,
# #     legend_labels,
# #     title='Vowels without tone',
# #     loc='best'
# # )

# # Create custom legend handles
# # from matplotlib.lines import Line2D

# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label=label,
#            markerfacecolor=color, markersize=10, markeredgecolor='k')
#     for label, color in zip(label_encoder.classes_, colors)
# ]

# plt.legend(
#     handles=legend_elements,
#     title='Vowels without Tones',
#     loc='best'
# )

# plt.title('HuBERT Embeddings of Yoruba Vowels without tones')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.grid(True)
# plt.tight_layout()

# # Function to generate filename
# def generate_filename(base_name, params, extension='png'):
#     params_str = '_'.join([f'{key}{value}' for key, value in params.items()])
#     filename = f'{base_name}_{params_str}.{extension}'
#     return filename

# # Construct the filename
# params = {
#     'perp': perplexity,
#     'lr': learning_rate,
#     'rs': random_state
# }
# filename = generate_filename('Yoruba_Vowels_without_tone_embeddings', params)

# # Save the plot to a file
# plt.savefig(filename, dpi=300, bbox_inches='tight')

# Display the plot
# plt.show()


# Check that the number of embeddings matches the number of labels
# if embeddings.shape[0] != labels.shape[0]:
#     raise ValueError("Number of embeddings and labels must match.")

# # Process each embedding to get a fixed-length vector
# processed_embeddings = []
# for idx, emb in enumerate(embeddings):
#     # emb is of shape (time_steps, feature_dim)
#     if not isinstance(emb, np.ndarray):
#         raise ValueError(f"Embedding at index {idx} is not a NumPy array.")
#     if emb.ndim != 2:
#         raise ValueError(f"Embedding at index {idx} is not 2D.")
#     # Mean pooling over time steps
#     emb_mean = emb.mean(axis=0)  # Shape: (feature_dim,)
#     processed_embeddings.append(emb_mean)

# # Convert the list to a NumPy array
# embeddings = np.vstack(processed_embeddings)  # Shape: (num_samples, feature_dim)

# # Verify the shape
# print("Processed embeddings shape:", embeddings.shape)

# # Encode labels to integers if they are strings
# label_encoder = LabelEncoder()
# label_ids = label_encoder.fit_transform(labels)
# # Define t-SNE parameters
# perplexity = 30
# # learning_rate = 200
# # random_state = 42

# # Dimensionality reduction using t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)

# # Plot the embeddings
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     embeddings_2d[:, 0],
#     embeddings_2d[:, 1],
#     c=label_ids,
#     cmap='viridis',
#     alpha=0.8,
#     edgecolors='k',
#     linewidths=0.5
# )

# # Create a legend with vowel labels
# handles, _ = scatter.legend_elements(num=len(label_encoder.classes_))
# legend_labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
# plt.legend(
#     handles,
#     legend_labels,
#     title='Tones',
#     loc='best'
# )

# plt.title('HuBERT Embeddings of Tones in Mandarin')
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.grid(True)
# plt.tight_layout()

# # Save the plot to a file
# plt.savefig('Tone_embeddings.png', dpi=300, bbox_inches='tight', format='png')
# _perp{perplexity}
# # Display the plot
# # plt.show()