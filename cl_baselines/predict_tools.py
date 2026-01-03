import numpy as np

from sklearn.linear_model import LinearRegression


def weighted_moving_average_predict(x: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Predict the next time step using weighted moving average (NumPy version).
    
    Args:
        x: ndarray of shape [num_samples, history_len, feature_size]
        weights: optional ndarray of shape [history_len]. If None, use linear weights.
        
    Returns:
        ndarray of shape [num_samples, feature_size]
    """
    num_samples, history_len, feature_size = x.shape

    if weights is None:
        # Default: linear weights [1, 2, ..., history_len]
        weights = np.arange(1, history_len + 1, dtype=np.float32)
    
    assert weights.shape == (history_len,), "weights must have shape [history_len]"

    # Normalize weights
    weights = weights / np.sum(weights)  # shape: [history_len]

    # Reshape weights to broadcast over feature dimension
    weights = weights.reshape(1, history_len, 1)

    # Apply weights and sum over time axis
    weighted_sum = np.sum(x * weights, axis=1)  # shape: [num_samples, feature_size]
    
    return weighted_sum

def linear_regression_predict(
    x: np.ndarray, 
    y: np.ndarray, 
    x_t: np.ndarray
):
    """
    Train a linear regression model directly on flattened sequence input.

    Args:
        x: ndarray of shape [num_samples, history_len, feature_size]
        y: ndarray of shape [num_samples, target_size]
        x_t: ndarray of shape [num_test_samples, history_len, feature_size]

    Returns:
        y_pred: predictions on testing set
    """
    num_samples, history_len, feature_size = x.shape

    # Flatten time and feature dimensions: [num_samples, history_len * feature_size]
    x_flat = x.reshape(num_samples, history_len * feature_size)

    # Train linear regression model
    model = LinearRegression()
    model.fit(x_flat, y)

    # Predict
    y_pred = model.predict(x_t.reshape(x_t.shape[0], -1))

    return model, y_pred
