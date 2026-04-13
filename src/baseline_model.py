import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def build_logistic_regression(random_state=42):
    """Create a Logistic Regression baseline model."""
    return LogisticRegression(max_iter=1000, random_state=random_state)


def train_logistic_regression(X_train, y_train, random_state=42):
    """Train Logistic Regression on the training data."""
    model = build_logistic_regression(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data and return accuracy and AUC."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    return {"accuracy": accuracy, "auc": auc}


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for binary classification."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def build_mlp(input_dim, hidden_dim=64):
    return MLP(input_dim, hidden_dim)


def train_mlp(X_train, y_train, input_dim, epochs=50, batch_size=32, lr=0.001, device="cpu"):
    model = build_mlp(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.from_numpy(X_train.to_numpy(dtype="float32")).to(device)
    y_tensor = torch.from_numpy(y_train.to_numpy(dtype="float32")).unsqueeze(1).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0

        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))

    return model, losses


def evaluate_mlp(model, X_test, y_test, device="cpu"):
    model.eval()
    model.to(device)

    X_tensor = torch.from_numpy(X_test.to_numpy(dtype="float32")).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    preds = (probs > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
    }