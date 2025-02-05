import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

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

# def load_data(features_path, labels_path):
#     X = np.load(features_path, allow_pickle=True)
#     y = np.load(labels_path, allow_pickle=True)

#     # Convert X to a consistent format (e.g., list of numpy arrays with float32 dtype)
#     X = [np.array(seq, dtype=np.float32) for seq in X]

#     return X, y

def load_data(features_path, labels_path):
    logger.info(f"Loading features from {features_path}...")
    features = np.load(features_path, allow_pickle=True)

    # Ensure the features are converted to a NumPy array of type float32
    if isinstance(features, np.ndarray) and features.dtype == np.object_:
        features = np.array([np.array(f, dtype=np.float32) for f in features])
    else:
        features = features.astype(np.float32)

    logger.info(f"Loading labels from {labels_path}...")
    labels = np.load(labels_path, allow_pickle=True)

    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Features or labels are empty. Please check the data files.")
    
    logger.info(f"Features loaded with shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    return features, labels

# def load_data(features_path, labels_path):
#     logger.info(f"Loading features from {features_path}...")
#     features = np.load(features_path, allow_pickle=True)
#     logger.info(f"Loading labels from {labels_path}...")
#     labels = np.load(labels_path, allow_pickle=True)
#     if len(features) == 0 or len(labels) == 0:
#         raise ValueError("Features or labels are empty. Please check the data files.")
    
#     logger.info(f"Features loaded with shape: {features.shape}")
#     logger.info(f"Labels shape: {labels.shape}")
#     return features, labels
    
class SpeechDataset(Dataset):
    def __init__(self, sequences, labels):
        # Ensure sequences are converted to PyTorch tensors of type float32
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     lengths = torch.tensor([len(seq) for seq in sequences])
    
#     # Convert the sequences to the required shape: (batch size, sequence length, input size)
#     sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
#     return sequences_padded, torch.tensor(labels), lengths

class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            output = model(sequences)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    logger.info(f"Saving model to {args.model_output_path}...")
    torch.save(model.state_dict(), args.model_output_path)
    logger.info("Model saved successfully.")

def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            output = model(sequences)
            _, predicted = torch.max(output, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    return y_true, y_pred

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()
    logger.info(args)

    logger.info("Loading data...")
    X, y = load_data(args.features_path, args.labels_path)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Loaded data is empty. Please check your data.")
    
    logger.info(f"Loaded {len(X)} sequences and {len(y)} labels")
   
    
    input_dim = X.shape[1]  # Directly use the length of the feature vectors as input_dim
    logger.info(f"Input dimension of the features: {input_dim}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Encoded the labels: {len(set(y_encoded))} unique classes")

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    logger.info("Creating datasets...")
    train_dataset = SpeechDataset(X_train, y_train)
    test_dataset = SpeechDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    model = FullyConnectedClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("Training classifier...")
    train(model, train_loader, criterion, optimizer, num_epochs=20)

    logger.info("Evaluating classifier...")
    y_true, y_pred = evaluate(model, test_loader)

    target_names = label_encoder.classes_
    logger.info(f"Classes: {target_names}")

    print(classification_report(y_true, y_pred, target_names=target_names, labels=np.unique(y_true)))
    logger.info("Classification report generated.")
