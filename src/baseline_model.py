from sklearn.linear_model import LogisticRegression


def build_logistic_regression():
    """Create a simple Logistic Regression baseline model."""
    return LogisticRegression(max_iter=1000)
