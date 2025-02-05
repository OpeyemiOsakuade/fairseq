# import json
# import argparse
# import logging
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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

# # Step 1: Load Data from JSON File
# def load_data_from_json(json_file):
#     logger.info(f"Loading data from {json_file}...")
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     return data

# # Step 2: Flatten the Data Structure
# # def flatten_data(data):
# #     X = []
# #     y = []
# #     for vowel, content in data.items():
# #         logger.info(f"Processing data for vowel: {vowel}")
# #         sequences = content['speech_reps']
# #         labels = [vowel] * len(sequences)
# #         X.extend(sequences)
# #         y.extend(labels)
# #     return X, y

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

# # Step 3: Prepare Data for PyTorch
# class SpeechDataset(Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.sequences[idx], self.labels[idx]


# # Create data loaders
# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     lengths = torch.tensor([len(seq) for seq in sequences])
#     sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
#     return sequences_padded, torch.tensor(labels), lengths


# # Step 4: Define the LSTM Model
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(LSTMClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, lengths):
#         embedded = self.embedding(x)
#         packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
#         packed_output, (hidden, cell) = self.lstm(packed)
#         output = self.fc(hidden[-1])
#         return output

# # Step 5: Train the Model
# def train(model, train_loader, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         for sequences_padded, labels, lengths in train_loader:
#             optimizer.zero_grad()
#             output = model(sequences_padded, lengths)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

#     # # Save the trained model
#     # logger.info(f"Saving model to {args.model_output_path}...")
#     # torch.save(model.state_dict(), "args.model_output_path")
#     # print("Model saved to lstm_model.pth")
#     # logger.info("Model saved successfully.")

#     logger.info(f"Saving model to {args.model_output_path}...")
#     torch.save(model.state_dict(), args.model_output_path)
#     logger.info("Model saved successfully.")

# # To load the model later
# # model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# # model.load_state_dict(torch.load("lstm_model.pth"))
# # model.eval()  # Set the model to evaluation mode




# # Step 6: Evaluate the Model
# def evaluate(model, test_loader):
#     model.eval()
#     y_true = []
#     y_pred = []
#     with torch.no_grad():
#         for sequences_padded, labels, lengths in test_loader:
#             output = model(sequences_padded, lengths)
#             _, predicted = torch.max(output, 1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(predicted.cpu().numpy())
#     return y_true, y_pred


# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     logger = get_logger()
#     logger.info(args)

#     logger.info("Loading data...")
#     # Load data from your JSON file
#     # json_file_path = "your_data_file.json"  # Replace with your actual file path
#     data = load_data_from_json(args.data_path)

#     # Flatten the data
#     X, y = flatten_data(data)
#     logger.info(f"Flattened the features: {X}")
#     logger.info(f"Flattened the labels: {y}")
#     # Ensure there are no empty sequences in X before calculating vocab_size
#     # X = [seq for seq in X if len(seq) > 0]
#     # Filter out empty sequences and corresponding labels
#     X, y = zip(*[(seq, label) for seq, label in zip(X, y) if len(seq) > 0])
#     # Convert back to lists
#     X = list(X)
#     y = list(y)
#     # Check again if X is not empty after filtering
#     if len(X) == 0:
#         raise ValueError("No valid sequences found. Please check your data.")
#     # Now calculate the vocab_size
#     vocab_size = max(max(seq) for seq in X) + 1  # Assume the sequences are integers starting from 0


#     # Encode the labels
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
#     logger.info(f"Encoded the labels: {y_encoded}")

#     # Split data into training and test sets
#     logger.info("Splitting data into training and testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     # Create datasets
#     logger.info("Creating datasets..")
#     train_dataset = SpeechDataset(X_train, y_train)
#     test_dataset = SpeechDataset(X_test, y_test)

#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

#     # Parameters
#     # vocab_size = max(max(seq) for seq in X) + 1  # Assume the sequences are integers starting from 0
#     embedding_dim = 50
#     hidden_dim = 128
#     output_dim = len(label_encoder.classes_)

#     # Instantiate the model, loss function, and optimizer
#     model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     logger.info("Training classifier...")
#     train(model, train_loader, criterion, optimizer, num_epochs=20)

#     logger.info("Evaluating classifier...")
#     y_true, y_pred = evaluate(model, test_loader)

#     # # Print classification report
#     # print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
#     # logger.info(f"Classification Report:\n{classification_report}")

#     # Print classification report
#     target_names = [label for label in label_encoder.classes_ if label in y_true or label in y_pred]
#     print(classification_report(y_true, y_pred, target_names=target_names))
#     logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, target_names=target_names)}")


import json
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
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the data with speech representations and labels.",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default="classifier_model.pth",
        help="Path to save the trained classifier model.",
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
        self.sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences_padded, torch.tensor(labels), lengths

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        output = self.fc(hidden[-1])
        return output

def train(model, train_loader, criterion, optimizer, num_epochs=10):
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
    data = load_data_from_json(args.data_path)

    X, y = flatten_data(data)
    logger.info(f"Flattened the features: {len(X)} sequences")
    logger.info(f"Flattened the labels: {len(y)} labels")

    X, y = zip(*[(seq, label) for seq, label in zip(X, y) if len(seq) > 0])
    X = list(X)
    y = list(y)

    if len(X) == 0:
        raise ValueError("No valid sequences found. Please check your data.")

    vocab_size = max(max(seq) for seq in X) + 1  # Assume the sequences are integers starting from 0

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

    embedding_dim = 50
    hidden_dim = 128
    output_dim = len(label_encoder.classes_)

    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
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

