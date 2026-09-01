"""
Graph-Based Rumor Detection
BERT + Graph Attention Network (GAT) on the PHEME dataset.

This script is a cleaned, GitHub-ready version of the Colab implementation
used for the Graph-Based Rumor Detection project.

Pipeline:
1. Download the PHEME dataset using KaggleHub.
2. Extract BERT text embeddings and user metadata.
3. Build conversation/reaction graphs.
4. Normalize node features.
5. Balance the training data.
6. Split into train/validation/test sets.
7. Train a GAT-based graph classifier.
8. Report accuracy and F1 score.

Reference implementation details:
- BERT: bert-base-cased
- BERT output: 768 dimensions
- Combined BERT + user features: 773 dimensions
- Projection: 773 -> 712
- GAT: 712 -> 1024, one attention head
- FC layers: 1024 -> 987 -> 463 -> 654 -> 1
- Dropout: 0.121, 0.024, 0.342
- Optimizer: Adam
- Learning rate: 0.0000313
- Weight decay: 0.00221
- Epochs: 30
- Batch size: 32
- Random seed: 42
"""

import os
import json
import random
from datetime import datetime

import kagglehub
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from transformers import BertModel, BertTokenizer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
DATASET_NAME = "manuelcecerepalazzo/pheme-dataset"
GRAPH_FILE = "pheme_graphs.pt"

BERT_MODEL_NAME = "bert-base-cased"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0000313
WEIGHT_DECAY = 0.00221

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=SEED):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ---------------------------------------------------------------------------
# BERT feature extraction
# ---------------------------------------------------------------------------

print(f"Using device: {DEVICE}")
print(f"Loading BERT model: {BERT_MODEL_NAME}")

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
bert_model.eval()


def get_bert_embedding(text):
    """Extract the first-token ([CLS]) BERT embedding for a tweet."""
    text = text or ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    # First token corresponds to [CLS].
    return outputs.last_hidden_state[0][0].cpu().numpy()


# ---------------------------------------------------------------------------
# User feature extraction
# ---------------------------------------------------------------------------

def get_user_features(user_dict, tweet_created_at):
    """
    Extract user/account features.

    Features:
    - verified
    - followers_count
    - friends_count
    - statuses_count
    - account age in days
    """
    user_dict = user_dict or {}

    verified = 1.0 if user_dict.get("verified", False) else 0.0
    followers = float(user_dict.get("followers_count", 0))
    friends = float(user_dict.get("friends_count", 0))
    tweet_count = float(user_dict.get("statuses_count", 0))

    try:
        created_at = datetime.strptime(
            user_dict.get("created_at"),
            "%a %b %d %H:%M:%S +0000 %Y",
        )
        tweet_time = datetime.strptime(
            tweet_created_at,
            "%a %b %d %H:%M:%S +0000 %Y",
        )
        account_age = float((tweet_time - created_at).days)
    except (TypeError, ValueError):
        account_age = 0.0

    return [
        verified,
        followers,
        friends,
        tweet_count,
        account_age,
    ]


# ---------------------------------------------------------------------------
# Dataset parsing and graph construction
# ---------------------------------------------------------------------------

def parse_dataset(base_path):
    """
    Parse the PHEME JSON files and construct one graph per conversation.

    Each tweet/reaction is represented as a node containing:
        BERT text embedding + user/account features.

    Reply relationships are represented as bidirectional graph edges.
    """
    dataset = []
    event_folders = []

    # Locate event directories recursively.
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if not d.startswith("._")]

        if "rumours" in dirs or "non-rumours" in dirs:
            event_folders.append(root)

    event_folders = sorted(event_folders)

    print(f"Found {len(event_folders)} valid event folders:")
    for event_folder in event_folders:
        print(f"  -> {event_folder}")

    if not event_folders:
        return dataset

    # Process every event and both classes.
    for event_path in event_folders:
        for label_str, label_val in [
            ("rumours", 1),
            ("non-rumours", 0),
        ]:
            class_path = os.path.join(event_path, label_str)

            if not os.path.exists(class_path):
                continue

            threads = sorted(
                [
                    directory
                    for directory in os.listdir(class_path)
                    if not directory.startswith("._")
                    and os.path.isdir(os.path.join(class_path, directory))
                ]
            )

            for thread_id in threads:
                thread_path = os.path.join(class_path, thread_id)

                # Locate source tweet directory.
                source_dir = next(
                    (
                        os.path.join(thread_path, directory)
                        for directory in os.listdir(thread_path)
                        if "source" in directory.lower()
                        and os.path.isdir(os.path.join(thread_path, directory))
                    ),
                    None,
                )

                if source_dir is None:
                    continue

                source_files = sorted(
                    [
                        file
                        for file in os.listdir(source_dir)
                        if file.endswith(".json")
                        and not file.startswith("._")
                    ]
                )

                if not source_files:
                    continue

                with open(
                    os.path.join(source_dir, source_files[0]),
                    "r",
                    encoding="utf-8",
                ) as file:
                    source_data = json.load(file)

                nodes = []
                edges = []

                # Source tweet becomes node 0.
                text_emb = get_bert_embedding(source_data.get("text", ""))
                user_feat = get_user_features(
                    source_data.get("user", {}),
                    source_data.get("created_at"),
                )

                nodes.append(
                    np.concatenate([text_emb, user_feat])
                )

                tweet_id_to_idx = {
                    str(source_data["id"]): 0
                }

                # Locate reactions/replies.
                reactions_dir = next(
                    (
                        os.path.join(thread_path, directory)
                        for directory in os.listdir(thread_path)
                        if "reaction" in directory.lower()
                        and os.path.isdir(os.path.join(thread_path, directory))
                    ),
                    None,
                )

                if reactions_dir:
                    reaction_files = sorted(
                        [
                            file
                            for file in os.listdir(reactions_dir)
                            if file.endswith(".json")
                            and not file.startswith("._")
                        ]
                    )

                    for reaction_file in reaction_files:
                        reaction_path = os.path.join(
                            reactions_dir,
                            reaction_file,
                        )

                        with open(
                            reaction_path,
                            "r",
                            encoding="utf-8",
                        ) as file:
                            reaction_data = json.load(file)

                        text_emb = get_bert_embedding(
                            reaction_data.get("text", "")
                        )
                        user_feat = get_user_features(
                            reaction_data.get("user", {}),
                            reaction_data.get("created_at"),
                        )

                        nodes.append(
                            np.concatenate([text_emb, user_feat])
                        )

                        current_idx = len(nodes) - 1

                        tweet_id_to_idx[
                            str(reaction_data["id"])
                        ] = current_idx

                        replied_to = str(
                            reaction_data.get(
                                "in_reply_to_status_id"
                            )
                        )

                        # Fall back to source node if the parent is not
                        # already present in the mapping.
                        target_idx = tweet_id_to_idx.get(
                            replied_to,
                            0,
                        )

                        # Bidirectional edge.
                        edges.append([current_idx, target_idx])
                        edges.append([target_idx, current_idx])

                # Convert node features to tensors and normalize them.
                x = torch.tensor(
                    np.array(nodes),
                    dtype=torch.float,
                )

                scaler = MinMaxScaler()
                x = torch.tensor(
                    scaler.fit_transform(x.numpy()),
                    dtype=torch.float,
                )

                if edges:
                    edge_index = torch.tensor(
                        edges,
                        dtype=torch.long,
                    ).t().contiguous()
                else:
                    edge_index = torch.empty(
                        (2, 0),
                        dtype=torch.long,
                    )

                y = torch.tensor(
                    [label_val],
                    dtype=torch.float,
                )

                dataset.append(
                    Data(
                        x=x,
                        edge_index=edge_index,
                        y=y,
                    )
                )

    return dataset


# ---------------------------------------------------------------------------
# Dataset download and processing
# ---------------------------------------------------------------------------

def load_or_process_dataset():
    """Download/process PHEME and cache the resulting graph dataset."""
    if os.path.exists(GRAPH_FILE):
        print(f"Found saved graphs: {GRAPH_FILE}")
        print("Loading graphs from disk...")
        return torch.load(
            GRAPH_FILE,
            weights_only=False,
        )

    print("Downloading PHEME dataset via KaggleHub...")
    extract_path = kagglehub.dataset_download(DATASET_NAME)

    print(f"Path to dataset files: {extract_path}")
    print("Processing entire dataset and building graphs...")
    print("This step may take a while because BERT embeddings are generated.")

    graphs = parse_dataset(extract_path)

    print("Saving processed graphs to disk...")
    torch.save(graphs, GRAPH_FILE)

    print(f"Total graphs created: {len(graphs)}")

    return graphs


# ---------------------------------------------------------------------------
# Dataset balancing
# ---------------------------------------------------------------------------

def balance_dataset(data_list):
    """
    Balance the dataset by creating masked copies of minority rumor graphs.

    For each synthetic copy, 50% of the node-feature dimensions are masked.
    """
    labels = [int(data.y.item()) for data in data_list]

    rumor_idx = [
        i for i, label in enumerate(labels)
        if label == 1
    ]

    non_rumor_idx = [
        i for i, label in enumerate(labels)
        if label == 0
    ]

    diff = len(non_rumor_idx) - len(rumor_idx)

    if diff > 0 and rumor_idx:
        # Prefer rumor graphs with more than 10 nodes.
        large_rumors = [
            data_list[i]
            for i in rumor_idx
            if data_list[i].x.shape[0] > 10
        ]

        # Fallback if none are large enough.
        if not large_rumors:
            large_rumors = [
                data_list[i]
                for i in rumor_idx
            ]

        duplicates = []

        for _ in range(diff):
            sample = random.choice(large_rumors)
            new_x = sample.x.clone()

            # Mask 50% of feature dimensions.
            num_features_to_mask = int(new_x.shape[1] * 0.5)

            mask_indices = torch.randperm(
                new_x.shape[1]
            )[:num_features_to_mask]

            new_x[:, mask_indices] = 0.0

            duplicates.append(
                Data(
                    x=new_x,
                    edge_index=sample.edge_index,
                    y=sample.y,
                )
            )

        data_list.extend(duplicates)

    return data_list


# ---------------------------------------------------------------------------
# Train/validation/test split
# ---------------------------------------------------------------------------

def create_dataloaders(graphs):
    """Create stratified train, validation and test dataloaders."""
    if not graphs:
        raise ValueError("No graphs were loaded.")

    labels = [
        int(data.y.item())
        for data in graphs
    ]

    # 90% train+validation, 10% test.
    test_size = 0.1 if len(graphs) >= 10 else 1

    sss_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=SEED,
    )

    train_val_idx, test_idx = next(
        sss_test.split(graphs, labels)
    )

    train_val_graphs = [
        graphs[i]
        for i in train_val_idx
    ]

    test_graphs = [
        graphs[i]
        for i in test_idx
    ]

    # Balance only the training/validation pool.
    train_val_graphs = balance_dataset(
        train_val_graphs
    )

    # 80% train, 20% validation.
    labels_tv = [
        int(data.y.item())
        for data in train_val_graphs
    ]

    val_size = (
        0.2
        if len(train_val_graphs) >= 5
        else 1
    )

    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        sss_val.split(
            train_val_graphs,
            labels_tv,
        )
    )

    train_graphs = [
        train_val_graphs[i]
        for i in train_idx
    ]

    val_graphs = [
        train_val_graphs[i]
        for i in val_idx
    ]

    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(
        f"Train size: {len(train_graphs)}, "
        f"Val size: {len(val_graphs)}, "
        f"Test size: {len(test_graphs)}"
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# GAT model
# ---------------------------------------------------------------------------

class RumorGNN(nn.Module):
    """BERT-feature + user-feature GAT classifier."""

    def __init__(self, num_node_features):
        super().__init__()

        # Project combined BERT + user features to 712 dimensions.
        self.projection = nn.Linear(
            num_node_features,
            712,
        )

        # One GAT layer with 1024 output features.
        self.gat1 = GATConv(
            712,
            1024,
            heads=1,
        )

        # Fully connected classifier.
        self.fc1 = nn.Linear(1024, 987)
        self.fc2 = nn.Linear(987, 463)
        self.fc3 = nn.Linear(463, 654)
        self.fc4 = nn.Linear(654, 1)

        # Tuned dropout rates.
        self.drop1 = nn.Dropout(p=0.121)
        self.drop2 = nn.Dropout(p=0.024)
        self.drop3 = nn.Dropout(p=0.342)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Project node features.
        x = self.projection(x)

        # Graph Attention Network.
        x = self.gat1(x, edge_index)
        x = F.relu(x)

        # Mean graph-level readout.
        x = global_mean_pool(x, batch)

        # Fully connected classifier.
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = F.relu(self.fc3(x))
        x = self.drop3(x)

        # Binary classification probability.
        out = torch.sigmoid(self.fc4(x))

        return out


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train(model, loader, optimizer, criterion):
    """Run one training epoch."""
    model.train()

    total_loss = 0.0

    for data in loader:
        data = data.to(DEVICE)

        optimizer.zero_grad()

        out = model(data).squeeze()

        # Handle a single-graph batch safely.
        if out.dim() == 0:
            out = out.unsqueeze(0)

        loss = criterion(
            out,
            data.y,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item()
            * data.num_graphs
        )

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    """Evaluate loss, accuracy and F1 score."""
    model.eval()

    correct = 0
    total_loss = 0.0

    true_preds = []
    true_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)

            out = model(data).squeeze()

            if out.dim() == 0:
                out = out.unsqueeze(0)

            loss = criterion(
                out,
                data.y,
            )

            total_loss += (
                loss.item()
                * data.num_graphs
            )

            pred = (out > 0.5).float()

            correct += int(
                (pred == data.y).sum()
            )

            true_preds.extend(
                pred.cpu().numpy()
            )

            true_labels.extend(
                data.y.cpu().numpy()
            )

    accuracy = correct / len(loader.dataset)

    f1 = f1_score(
        true_labels,
        true_preds,
        zero_division=0,
    )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    return average_loss, accuracy, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the complete rumor-detection pipeline."""
    graphs = load_or_process_dataset()

    if not graphs:
        print("Cannot train model. No data loaded.")
        return

    train_loader, val_loader, test_loader = (
        create_dataloaders(graphs)
    )

    num_features = graphs[0].x.shape[1]

    model = RumorGNN(
        num_node_features=num_features
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCELoss()

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
        )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f}"
            )

    # Final test-set evaluation.
    if len(test_loader.dataset) > 0:
        test_loss, test_acc, test_f1 = evaluate(
            model,
            test_loader,
            criterion,
        )

        print("\n--- FINAL TEST SET RESULTS ---")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")

    # Save trained model.
    torch.save(
        model.state_dict(),
        "rumor_gnn_model.pt",
    )

    print("\nModel saved as: rumor_gnn_model.pt")


if __name__ == "__main__":
    main()
