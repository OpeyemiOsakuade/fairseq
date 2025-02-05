import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import logging
import json
from skfda.exploratory.stats import geometric_median
from skfda.misc.metrics import LpDistance
from skfda import FDataGrid

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
#     # Set up logging configuration
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

# def pad_and_flatten_features(features):
#     # Pad and flatten the features to make them uniform
#     logger.info("Padding and flattening features...")
#     max_length = max(len(f) for f in features)
#     num_features = 1  # Since speech_reps are 1D arrays
    
#     padded_features = np.zeros((len(features), max_length, num_features))
    
#     for i, f in enumerate(features):
#         f = np.array(f).reshape(-1, 1)  # Ensure 2D shape (n_samples, 1)
#         padded_features[i, :len(f), :] = f
    
#     flattened_features = padded_features.reshape(len(features), -1)
#     logger.info("Features padded and flattened.")
#     return flattened_features

# def preprocess_data(features):
#     # Scale the features using StandardScaler
#     logger.info("Scaling features using StandardScaler...")
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(features)
#     logger.info("Features scaled successfully.")
#     return scaled_features, scaler

# def compute_geometric_median(X_train):
#     # Compute the geometric median of the training data
#     logger.info("Converting features to floating point...")
#     X_train = X_train.astype(np.float64)  # Ensure features are in floating point format
    
#     logger.info("Computing geometric median...")
#     X_fdatagrid = FDataGrid(X_train)
#     median = geometric_median(X_fdatagrid, tol=1e-08, metric=LpDistance(p=2))
#     logger.info("Geometric median computed successfully.")
#     return median

# def evaluate_classifier(median, X_test, y_test):
#     # Evaluate the classifier by comparing the test data to the geometric median
#     logger.info("Evaluating classifier...")
#     median_vector = median.data_matrix[0].reshape(-1)  # Flatten median to match the shape of samples in X_test
#     distances = np.linalg.norm(X_test - median_vector, axis=1)
#     y_pred = np.where(distances < np.median(distances), 'class_1', 'class_2')  # Simplified binary classification example
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     logger.info(f"Accuracy: {accuracy}")
#     logger.info(f"Classification Report:\n{report}")
#     return accuracy, report

# if __name__ == "__main__":
#     # Main script execution
#     parser = get_parser()
#     args = parser.parse_args()
#     logger = get_logger()
#     logger.info("Starting the classification process...")
#     logger.info(f"Arguments received: {args}")

#     # Load and process data
#     logger.info("Loading data...")
#     features, labels = load_data(args.data_path)
    
#     logger.info("Padding and flattening features...")
#     features = pad_and_flatten_features(features)
#     logger.info(f"Padded and flattened features shape: {features.shape}")

#     # Preprocess data (scaling)
#     logger.info("Preprocessing data...")
#     scaled_features, scaler = preprocess_data(features)

#     # Split data into training and testing sets
#     logger.info("Splitting data into training and testing sets...")
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

#     # Compute geometric median
#     logger.info("Computing geometric median for the training data...")
#     median = compute_geometric_median(X_train)

#     # Evaluate the classifier
#     logger.info("Evaluating classifier based on geometric median...")
#     accuracy, report = evaluate_classifier(median, X_test, y_test)

#     # Save the model
#     logger.info(f"Saving model to {args.model_output_path}...")
#     joblib.dump((median, scaler), args.model_output_path)
#     logger.info("Model saved successfully.")
#     logger.info("Classification process completed.")


# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# import argparse
# import logging

# def get_parser():
#     parser = argparse.ArgumentParser(description="Classify extracted features.")
#     parser.add_argument(
#         "--features_path",
#         type=str,
#         required=True,
#         help="Path to the .npy file containing extracted features.",
#     )
#     parser.add_argument(
#         "--labels_path",
#         type=str,
#         required=True,
#         help="Path to the .npy file containing labels.",
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

# def load_data(features_path, labels_path):
#     features = np.load(features_path, allow_pickle=True)
#     labels = np.load(labels_path, allow_pickle=True)
#     return features, labels

# def pad_and_flatten_features(features):
#     max_length = max(len(f) for f in features)
#     num_features = features[0].shape[1]
#     padded_features = np.zeros((len(features), max_length, num_features))
    
#     for i, f in enumerate(features):
#         padded_features[i, :len(f), :] = f
    
#     flattened_features = padded_features.reshape(len(features), -1)
#     return flattened_features

# def preprocess_data(features):
#     scaler = StandardScaler()
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
#     features, labels = load_data(args.features_path, args.labels_path)
    
#     logger.info("Padding and flattening features...")
#     features = pad_and_flatten_features(features)
#     logger.info(f"Padded and flattened features shape: {features.shape}")

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

# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import logging
import json
from skfda.exploratory.stats import geometric_median
from skfda.misc.metrics import LpDistance
from skfda import FDataGrid

def get_parser():
    # Create a parser for command-line arguments
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
        default="classifier_model.pkl",
        help="Path to save the trained classifier model.",
    )
    return parser

def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def load_data(data_path):
    # Load data from the JSON file
    logger.info(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    features = []
    labels = []
    
    # Extract features and labels
    for label, content in data.items():
        logger.info(f"Processing data for label: {label}")
        for speech_rep in content['speech_reps']:
            features.append(np.array(speech_rep))
            labels.append(label)
    
    logger.info("Data loaded successfully.")
    return np.array(features, dtype=object), np.array(labels)

def pad_and_flatten_features(features):
    # Debug: Inspect the structure of features
    print("Inspecting features structure:")
    for i, f in enumerate(features):
        print(f"Feature {i}: Type={type(f)}, Shape={f.shape if isinstance(f, np.ndarray) else 'Not an array'}")
    
    max_length = max(len(f) for f in features)
    num_features = features[0].shape[1] if len(features[0].shape) > 1 else 1
    
    padded_features = np.zeros((len(features), max_length, num_features))
    
    for i, f in enumerate(features):
        if len(f.shape) == 1:  # Handle 1D arrays by adding an extra dimension
            f = f[:, np.newaxis]
        padded_features[i, :len(f), :] = f
    
    flattened_features = padded_features.reshape(len(features), -1)
    return flattened_features

def preprocess_data(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

def train_classifier(X_train, y_train):
    logger.info("Initializing Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=100, verbose=1)
    logger.info("Starting training...")
    classifier.fit(X_train, y_train)
    logger.info("Training completed.")
    return classifier

def evaluate_classifier(classifier, X_test, y_test):
    logger.info("Evaluating classifier...")
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info("Loading data...")
    features, labels = load_data(args.features_path, args.labels_path)
    
    logger.info("Padding and flattening features...")
    features = pad_and_flatten_features(features)
    logger.info(f"Padded and flattened features shape: {features.shape}")

    logger.info("Preprocessing data...")
    scaled_features, scaler = preprocess_data(features)

    logger.info("Splitting data into training and testing sets...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)

    logger.info("Training classifier...")
    classifier = train_classifier(X_train, y_train)

    logger.info("Evaluating classifier...")
    accuracy, report = evaluate_classifier(classifier, X_test, y_test)
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"Classification Report:\n{report}")

    logger.info(f"Saving model to {args.model_output_path}...")
    joblib.dump((classifier, scaler), args.model_output_path)
    logger.info("Model saved successfully.")
