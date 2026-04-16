import numpy as np

def split_into_clients(X, y, num_clients=5):
    """Split dataset into equal IID chunks."""
    data_size = len(X)
    shard_size = data_size // num_clients

    clients = []

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size

        X_client = X.iloc[start:end]
        y_client = y.iloc[start:end]

        clients.append((X_client, y_client))

    return clients