import flwr as fl
import torch

from src.baseline_model import build_mlp, train_mlp, evaluate_mlp


class FLClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test, input_dim, device="cpu"):
        self.model, _ = train_mlp(X_train, y_train, input_dim, epochs=1, device=device)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.input_dim = input_dim
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model, _ = train_mlp(
            self.X_train,
            self.y_train,
            self.input_dim,
            epochs=1,
            device=self.device
        )

        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = evaluate_mlp(self.model, self.X_test, self.y_test, device=self.device)

        return float(metrics["auc"]), len(self.X_test), metrics