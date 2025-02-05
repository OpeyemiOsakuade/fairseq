import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse
import logging

def get_parser():
    parser = argparse.ArgumentParser(description="Classify extracted features.")
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the .npy file containing extracted features.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to the .npy file containing labels.",
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

def load_data(features_path, labels_path):
    features = np.load(features_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    return features, labels

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
