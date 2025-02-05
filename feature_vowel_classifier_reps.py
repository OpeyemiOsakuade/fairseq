# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import json
# import joblib
# import argparse
# import logging
# from skfda.exploratory.stats import geometric_median
# from skfda.representation.grid import FDataGrid
# def get_parser():
#     # Create a parser for command-line arguments
#     parser = argparse.ArgumentParser(description="Classify extracted features.")
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         required=True,
#         help="Path to the JSON file containing the data with speech representations and labels.",
#     )
#     parser.add_argument(
#         "--model_output_path",
#         type=str,
#         default="classifier_model.pkl",
#         help="Path to save the trained classifier model.",
#     )
#     return parser

# def get_logger():
#     log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
#     logging.basicConfig(format=log_format, level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     return logger

# def load_data(data_path):
#     # Load data from the JSON file
#     logger.info(f"Loading data from {data_path}...")
#     with open(data_path, 'r') as f:
#         data = json.load(f)
#     features = []
#     labels = []
    
#     # Extract features and labels
#     for label, content in data.items():
#         logger.info(f"Processing data for label: {label}")
#         for speech_rep in content['speech_reps']:
#             features.append(np.array(speech_rep))
#             labels.append(label)
    
#     logger.info("Data loaded successfully.")
#     return np.array(features, dtype=object), np.array(labels)

# # def pad_and_flatten_features(features):
# #     # Debug: Inspect the structure of features
# #     print("Inspecting features structure:")
# #     for i, f in enumerate(features):
# #         print(f"Feature {i}: Type={type(f)}, Shape={f.shape if isinstance(f, np.ndarray) else 'Not an array'}")
    
# #     max_length = max(len(f) for f in features)
# #     num_features = features[0].shape[1] if len(features[0].shape) > 1 else 1
    
# #     padded_features = np.zeros((len(features), max_length, num_features))
    
# #     for i, f in enumerate(features):
# #         if len(f.shape) == 1:  # Handle 1D arrays by adding an extra dimension
# #             f = f[:, np.newaxis]
# #         padded_features[i, :len(f), :] = f
    
# #     flattened_features = padded_features.reshape(len(features), -1)
# #     return flattened_features

# # def pad_and_flatten_features(features):
# #     # logger.info("Padding and flattening features...")
# #     # Debug: Inspect the structure of features
# #     logger.info("Inspecting features structure...")
# #     for i, f in enumerate(features):
# #         logger.info(f"Feature {i}: Type={type(f)}, Length={len(f)}")
    
# #     max_length = max(len(f) for f in features)
    
# #     # Pad sequences with zeros to the maximum length and flatten them
# #     padded_features = np.zeros((len(features), max_length))
    
# #     for i, f in enumerate(features):
# #         padded_features[i, :len(f)] = f
# #         logger.info(f"Feature_padded {i}: Type={type(f)}, Length={len(f)}")
# #         logger.info(f"Showing padded features: {padded_features}")
    
# #     return padded_features

# # def compute_geometric_median(features):
# #     logger.info("Computing geometric median of features...")
    
# #     # Convert features to FDataGrid format
# #     fd_features = FDataGrid(features)
    
# #     # Compute the geometric median
# #     median = geometric_median(fd_features)
    
# #     # Extract the data matrix from the FDataGrid object
# #     median_features = median.data_matrix[0, ..., 0]
    
# #     logger.info("Geometric median computed successfully.")

# #     logger.info(f"Showing Geometric median computed features: { median_features}")
# #     return median_features

# # def compute_geometric_median(features):
# #     logger.info("Converting features to float and ensuring consistent shape...")
    
# #     # Convert each feature to a numpy array with float type and consistent length
# #     max_length = max(len(f) for f in features)
    
# #     # Initialize a list to store the padded features
# #     padded_features = []
    
# #     for f in features:
# #         # Convert each feature to a numpy array and pad with zeros if necessary
# #         padded_feature = np.array(f, dtype=np.float64)
# #         if len(padded_feature) < max_length:
# #             padded_feature = np.pad(padded_feature, (0, max_length - len(padded_feature)), mode='constant')
# #         padded_features.append(padded_feature)
    
# #     # Convert the list of padded features to a numpy array
# #     padded_features = np.array(padded_features)
    
# #     # Now convert features to FDataGrid format
# #     fd_features = FDataGrid(padded_features)
    
# #     # Compute the geometric median
# #     median = geometric_median(fd_features)
    
# #     # Extract the data matrix from the FDataGrid object
# #     median_features = median.data_matrix[0, ..., 0]
# #     # Ensure the result is a 2D array (as required by sklearn's StandardScaler)
# #     if median_features.ndim == 1:
# #         median_features = median_features.reshape(1, -1)
    
# #     logger.info("Geometric median computed successfully.")
# #     logger.info(f"Showing Geometric median computed features: { median_features}")
# #     return median_features

# def compute_geometric_median(features, labels):
#     logger.info("Computing geometric median for each class...")
    
#     unique_labels = np.unique(labels)
#     median_features = []
#     median_labels = []

#     for label in unique_labels:
#         # Collect features corresponding to the current label
#         class_features = [np.array(features[i], dtype=np.float64) for i in range(len(features)) if labels[i] == label]
        
#         # Find the maximum length among all features for this class
#         max_length = max(f.shape[0] for f in class_features)
        
#         # Pad all features in this class to have the same length
#         padded_features = []
#         for f in class_features:
#             if f.shape[0] < max_length:
#                 f = np.pad(f, (0, max_length - f.shape[0]), mode='constant')
#             padded_features.append(f)
        
#         # Convert the list of padded features to a 2D array
#         padded_features = np.array(padded_features)
        
#         # Compute geometric median
#         fd_class_features = FDataGrid(padded_features)
#         median = geometric_median(fd_class_features)
#         median_feature = median.data_matrix[0, ..., 0]
        
#         if median_feature.ndim == 1:
#             median_feature = median_feature.reshape(1, -1)
        
#         # Store the median feature and corresponding label
#         median_features.append(median_feature)
#         median_labels.append(label)
    
#     # Stack the median features into a 2D array and convert labels to an array
#     median_features = np.vstack(median_features)
#     median_labels = np.array(median_labels)
    
#     logger.info("Geometric medians computed successfully.")
#     return median_features, median_labels




# def preprocess_data(features):
#     logger.info("Scaling features...")
#     scaler = StandardScaler()
    
#     # Ensure features are in 2D shape (if not already)
#     if features.ndim == 1:
#         features = features.reshape(1, -1)
    
#     scaled_features = scaler.fit_transform(features)
#     return scaled_features, scaler

# def train_classifier(X_train, y_train):
#     logger.info("Initializing Logistic Regression classifier...")
#     classifier = LogisticRegression(max_iter=100, verbose=1)
#     logger.info("Starting training...")
#     classifier.fit(X_train, y_train)
#     logger.info("Training completed.")
#     return classifier

# def evaluate_classifier(classifier, X_test, y_test):
#     logger.info("Evaluating classifier...")
#     y_pred = classifier.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     return accuracy, report


# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     logger = get_logger()
#     logger.info(args)

#     logger.info("Loading data...")
#     # features, labels = load_data(args.features_path, args.labels_path)
#     features, labels = load_data(args.data_path)
#     logger.info(f"Showing features: {features}")
#     logger.info(f"Showing labels: {labels}")
#     logger.info(f"features shape: {features.shape}")


#     logger.info("Computing geometric median of features...")
#     # features = compute_geometric_median(features)
#     features, labels = compute_geometric_median(features, labels)
#     logger.info(f"Features shape after computing geometric median: {features.shape}")

    
#     logger.info("Preprocessing data...")
#     scaled_features, scaler = preprocess_data(features)

#     logger.info("Splitting data into training and testing sets...")
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

#     logger.info("Training classifier...")
#     classifier = train_classifier(X_train, y_train)

#     logger.info("Evaluating classifier...")
#     accuracy, report = evaluate_classifier(classifier, X_test, y_test)
#     logger.info(f"Accuracy: {accuracy}")
#     logger.info(f"Classification Report:\n{report}")

#     logger.info(f"Saving model to {args.model_output_path}...")
#     joblib.dump((classifier, scaler), args.model_output_path)
#     logger.info("Model saved successfully.")


# import json
# import argparse
# import logging
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# from sklearn.linear_model import LogisticRegression
# from torch.utils.data import Dataset, DataLoader
# import torch

# def get_parser():
#     parser = argparse.ArgumentParser(description="Classify extracted features.")
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         required=True,
#         help="Path to the JSON file containing the data with speech representations and labels.",
#     )
#     parser.add_argument(
#         "--model_output_path",
#         type=str,
#         default="classifier_model.npy",
#         help="Path to save the trained classifier model parameters.",
#     )
#     return parser

# def get_logger():
#     log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
#     logging.basicConfig(format=log_format, level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     return logger

# def load_data_from_json(json_file):
#     logger.info(f"Loading data from {json_file}...")
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     return data

# def flatten_data(data):
#     X = []
#     y = []
#     for vowel, content in data.items():
#         if len(content['speech_reps']) > 0:  # Only include non-empty sequences
#             logger.info(f"Processing data for vowel: {vowel}")
#             sequences = content['speech_reps']
#             labels = [vowel] * len(sequences)
#             X.extend(sequences)
#             y.extend(labels)
#     return X, y

# class SpeechDataset(Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.sequences[idx], self.labels[idx]

# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     return torch.stack(sequences), torch.tensor(labels)

# def geometric_median(X):
#     """
#     Compute the geometric median for a set of points.
#     """
#     median = np.median(X, axis=0)
#     for _ in range(10):  # Perform a fixed number of iterations
#         distances = np.linalg.norm(X - median, axis=1)
#         nonzero_distances = distances != 0
#         inv_distances = 1.0 / distances[nonzero_distances]
#         weighted_sum = np.sum(X[nonzero_distances] * inv_distances[:, None], axis=0)
#         median = weighted_sum / np.sum(inv_distances)
#     return median

# def train(model, train_loader, num_epochs=10):
#     model.train()
#     X_train = []
#     y_train = []

#     for epoch in range(num_epochs):
#         for sequences, labels in train_loader:
#             sequences = sequences.numpy()
#             for label in np.unique(labels):
#                 X_class = sequences[labels.numpy() == label]
#                 median_vector = geometric_median(X_class)
#                 X_train.append(median_vector)
#                 y_train.append(label)
        
#         model.fit(X_train, y_train)
#         logger.info(f'Epoch {epoch+1}/{num_epochs}, Training completed.')

#     logger.info(f"Saving model to {args.model_output_path}...")
#     np.save(args.model_output_path, model.coef_)
#     logger.info("Model saved successfully.")

# def evaluate(model, test_loader):
#     model.eval()
#     y_true = []
#     y_pred = []

#     for sequences, labels in test_loader:
#         sequences = sequences.numpy()
#         predictions = model.predict(sequences)
#         y_true.extend(labels.numpy())
#         y_pred.extend(predictions)
    
#     return y_true, y_pred

# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     logger = get_logger()
#     logger.info(args)

#     logger.info("Loading data...")
#     data = load_data_from_json(args.data_path)

#     X, y = flatten_data(data)
#     logger.info(f"Flattened the features: {len(X)} sequences")
#     logger.info(f"Flattened the labels: {len(y)} labels")

#     X, y = zip(*[(seq, label) for seq, label in zip(X, y) if len(seq) > 0])
#     X = list(X)
#     y = list(y)

#     if len(X) == 0:
#         raise ValueError("No valid sequences found. Please check your data.")

#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
#     logger.info(f"Encoded the labels: {len(set(y_encoded))} unique classes")

#     logger.info("Splitting data into training and testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     logger.info("Creating datasets...")
#     train_dataset = SpeechDataset(X_train, y_train)
#     test_dataset = SpeechDataset(X_test, y_test)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

#     model = LogisticRegression(max_iter=1000)

#     logger.info("Training classifier...")
#     train(model, train_loader, num_epochs=20)

#     logger.info("Evaluating classifier...")
#     y_true, y_pred = evaluate(model, test_loader)

#     target_names = label_encoder.classes_
#     logger.info(f"Classes: {target_names}")

#     print(classification_report(y_true, y_pred, target_names=target_names, labels=np.unique(y_true)))
#     logger.info("Classification report generated.")



import json
import argparse
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="Classify extracted features.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the data with speech representations and labels.",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default="classifier_model.npy",
        help="Path to save the trained classifier model parameters.",
    )
    return parser

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def load_data_from_json(json_file):
    logger.info(f"Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def flatten_data(data):
    X = []
    y = []
    for vowel, content in data.items():
        if len(content['speech_reps']) > 0:  # Only include non-empty sequences
            logger.info(f"Processing data for vowel: {vowel}")
            sequences = content['speech_reps']
            labels = [vowel] * len(sequences)
            X.extend(sequences)
            y.extend(labels)
    return X, y

class SpeechDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.tensor(labels)

def geometric_median(X):
    """
    Compute the geometric median for a set of points.
    """
    median = np.median(X, axis=0)
    for _ in range(10):  # Perform a fixed number of iterations
        distances = np.linalg.norm(X - median, axis=1)
        nonzero_distances = distances != 0
        inv_distances = 1.0 / distances[nonzero_distances]
        weighted_sum = np.sum(X[nonzero_distances] * inv_distances[:, None], axis=0)
        median = weighted_sum / np.sum(inv_distances)
    return median

def train(model, train_loader, num_epochs=10):
    X_train = []
    y_train = []

    for epoch in range(num_epochs):
        for sequences, labels in train_loader:
            sequences = sequences.numpy()
            for label in np.unique(labels):
                X_class = sequences[labels.numpy() == label]
                median_vector = geometric_median(X_class)
                X_train.append(median_vector)
                y_train.append(label)

        # Fit the logistic regression model at the end of each epoch
        model.fit(X_train, y_train)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Training completed.')

    logger.info(f"Saving model to {args.model_output_path}...")
    np.save(args.model_output_path, model.coef_)
    logger.info("Model saved successfully.")

def evaluate(model, test_loader):
    y_true = []
    y_pred = []

    for sequences, labels in test_loader:
        sequences = sequences.numpy()
        predictions = model.predict(sequences)
        y_true.extend(labels.numpy())
        y_pred.extend(predictions)
    
    return y_true, y_pred

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info("Loading data...")
    data = load_data_from_json(args.data_path)

    X, y = flatten_data(data)
    logger.info(f"Flattened the features: {len(X)} sequences")
    logger.info(f"Flattened the labels: {len(y)} labels")

    X, y = zip(*[(seq, label) for seq, label in zip(X, y) if len(seq) > 0])
    X = list(X)
    y = list(y)

    if len(X) == 0:
        raise ValueError("No valid sequences found. Please check your data.")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Encoded the labels: {len(set(y_encoded))} unique classes")

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    logger.info("Creating datasets...")
    train_dataset = SpeechDataset(X_train, y_train)
    test_dataset = SpeechDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = LogisticRegression(max_iter=1000)

    logger.info("Training classifier...")
    train(model, train_loader, num_epochs=20)

    logger.info("Evaluating classifier...")
    y_true, y_pred = evaluate(model, test_loader)

    target_names = label_encoder.classes_
    logger.info(f"Classes: {target_names}")

    print(classification_report(y_true, y_pred, target_names=target_names, labels=np.unique(y_true)))
    logger.info("Classification report generated.")
