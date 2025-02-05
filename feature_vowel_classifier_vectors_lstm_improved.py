import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(description="Classify extracted features.")
    parser.add_argument("--features_path", type=str, required=True, help="Path to the .npy file containing extracted features.")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to the .npy file containing labels.")
    parser.add_argument("--model_output_path", type=str, default="classifier_model.pt", help="Path to save the trained classifier model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Number of hidden units in the LSTM.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
    return parser


def get_logger():
    log_format = "[%(asctime)s] [%(levelname)s]: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def load_data(features_path, labels_path):
    logger.info(f"Loading features from {features_path}...")
    features = np.load(features_path, allow_pickle=True)
    logger.info(f"Loading labels from {labels_path}...")
    labels = np.load(labels_path, allow_pickle=True)

    if features.shape[0] != labels.shape[0]:
        raise ValueError("Mismatch between number of features and labels.")
    if len(features) == 0 or len(labels) == 0:
        raise ValueError("Features or labels are empty. Please check the data files.")

    logger.info(f"Features loaded with shape: {features.shape}")
    logger.info(f"Labels loaded with shape: {labels.shape}")
    return features, labels


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
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, torch.tensor(labels), lengths


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed)
        output = self.fc(hidden[-1])  # Use the last hidden state
        return output


def train(model, train_loader, criterion, optimizer, num_epochs, logger, val_loader=None):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for sequences_padded, labels, lengths in train_loader:
            optimizer.zero_grad()
            output = model(sequences_padded, lengths)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

        if val_loader:
            val_loss, val_accuracy = validate(model, val_loader, criterion)
            logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}")


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences_padded, labels, lengths in val_loader:
            output = model(sequences_padded, lengths)
            loss = criterion(output, labels)
            val_loss += loss.item()
            predictions = output.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    accuracy = correct / total
    return val_loss, accuracy


def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for sequences_padded, labels, lengths in test_loader:
            output = model(sequences_padded, lengths)
            predictions = output.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    return y_true, y_pred


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger = get_logger()

    logger.info("Loading data...")
    X, y = load_data(args.features_path, args.labels_path)

    input_dim = X[0].shape[1]
    logger.info(f"Input dimension: {input_dim}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info(f"Encoded {len(set(y_encoded))} unique classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    train_dataset = SpeechDataset(X_train, y_train)
    test_dataset = SpeechDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    hidden_dim = args.hidden_dim
    output_dim = len(label_encoder.classes_)

    model = LSTMClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("Training the model...")
    train(model, train_loader, criterion, optimizer, args.num_epochs, logger)

    logger.info("Evaluating the model...")
    y_true, y_pred = evaluate(model, test_loader)

    logger.info("Generating classification report...")
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(report)

    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="viridis")
    plt.show()

    logger.info(f"Saving model to {args.model_output_path}...")
    torch.save(model.state_dict(), args.model_output_path)
    logger.info("Model saved successfully.")
