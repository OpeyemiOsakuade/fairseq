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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

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
    # Reads features and labels stored as NumPy arrays.
    # Validates that neither is empty.
    # Logs the shapes of the loaded data.
    logger.info(f"Loading features from {features_path}...")
    features = np.load(features_path, allow_pickle=True)
    logger.info(f"Loading labels from {labels_path}...")
    labels = np.load(labels_path, allow_pickle=True)
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Features or labels are empty. Please check the data files.")
    
    logger.info(f"Features loaded with shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    return features, labels

class SpeechDataset(Dataset):
    """define custom Pytorch dataset for loading sequences and labels
        - convert sequences and labels to pytorch and integer tensors respectively
        - Implements __len__ and __getitem__ for compatibility with PyTorch's DataLoader.
        connection: Supplies data in batches for model training and evaluation.
        """
    def __init__(self, sequences, labels):
        # Ensure sequences are converted to PyTorch tensors of type float32
        self.sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """Prepare a batch of variable-length sequences for training.
    - Pads sequences to the same length within a batch.
    - Computes original sequence lengths for packing operations in LSTM.
    Why: LSTMs need padded sequences for batch processing, but also need original lengths to ignore padding.
    Connection: Processes each batch before feeding it into the model.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Convert the sequences to the required shape: (batch size, sequence length, input size)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    
    return sequences_padded, torch.tensor(labels), lengths

# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     lengths = torch.tensor([len(seq) for seq in sequences])
#     sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
#     return sequences_padded, torch.tensor(labels), lengths

class LSTMClassifier(nn.Module):
    """Define the LSTM-based neural network model.
        LSTM layer processes input sequences, capturing temporal dependencies.
        Fully connected layer maps LSTM outputs to class predictions.
        Uses packing/unpacking for efficiency with variable-length sequences.
        Why: LSTMs are well-suited for sequential data like speech.
        Connection: Implements the model architecture used for classification."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        # Debugging print to check the shape before packing
        print(f"Input shape before packing: {x.shape}")

        # Pack the sequences to handle varying lengths
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Debugging print to check the packed shape
        print(f"Packed input shape: {packed.data.shape}")

        packed_output, (hidden, cell) = self.lstm(packed)

        # Debugging print to check the shape after LSTM
        print(f"Hidden state shape: {hidden.shape}")

        output = self.fc(hidden[-1])
        return output

# def train(model, train_loader, criterion, optimizer, num_epochs=10, logger=None, model_output_path=None):
#     model.train()
#     for epoch in range(num_epochs):
#         for sequences_padded, labels, lengths in train_loader:
#             optimizer.zero_grad()
#             output = model(sequences_padded, lengths)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()
#         logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

#     if model_output_path:
#         logger.info(f"Saving model to {model_output_path}...")
#         torch.save(model.state_dict(), model_output_path)
#         logger.info("Model saved successfully.")

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    """Purpose: Train the LSTM model.
        - Iterates through batches of training data.
        - Computes the loss using cross-entropy (for multi-class classification).
        - Optimizes model weights using backpropagation.
        - Logs training loss after each epoch.
        Why: Adjusts the model to minimize prediction error.
        Connection: Produces the trained model for evaluation."""
    model.train()
    for epoch in range(num_epochs):
        for sequences_padded, labels, lengths in train_loader:
            optimizer.zero_grad()
            output = model(sequences_padded, lengths)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    logger.info(f"Saving model to {args.model_output_path}...")
    torch.save(model.state_dict(), args.model_output_path)
    logger.info("Model saved successfully.")

def evaluate(model, test_loader):
    """ Evaluate the trained model on the test dataset.
        Disables gradient computation for efficiency.
        Makes predictions on test data.
        Stores true and predicted labels for evaluation.
        Why: Measures model performance on unseen data.
        Connection: Validates the model's generalization capability."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for sequences_padded, labels, lengths in test_loader:
            output = model(sequences_padded, lengths)
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
    # # Ensure that sequences are not empty and determine input_dim
    # X, y = zip(*[(seq, label) for seq, label in zip(X, y) if len(seq) > 0])
    # X = list(X)
    # y = list(y)

    # if len(X) == 0:
    #     raise ValueError("No valid sequences found. Please check your data.")

    # # Check if the sequences are 1D or 2D
    # if len(X[0].shape) == 1:
    #     input_dim = len(X[0])
    # else:
    #     input_dim = X[0].shape[1]  # Assuming each sequence is a list of vectors of fixed size

    #Validate the shape of the data
    # if len(X[0].shape) == 1:
    #     raise ValueError("Each sequence in the feature data should be 2D (sequence_length, feature_dim).")
    
    input_dim = X[0].shape[1] # Dimension of the feature vectors
    logger.info(f"Input dimension of the features: {input_dim}")

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

    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    model = LSTMClassifier(input_dim, hidden_dim, output_dim)
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