import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.asarray(x, dtype=float)

    value = np.where(x > 0, x, 0) # Check whether x is larger than 0 or not

    return value