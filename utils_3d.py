import numpy as np
def toHomog(x: np.ndarray) -> np.ndarray:
    """
    x is a dxN array
    output is (d+1)xN array
    """

    d, N = x.shape
    return np.vstack([x, np.ones((1, N))])    

def toInHomog(x: np.ndarray) -> np.ndarray:
    """
        Converts to inhomogeneous coordinates
        Args:
        x_bar : (d+1, n) data points
        Returns:
        x : (d, n)
    """
    x_bar = x[:-1]
    x_bar /= x[-1]
    return x_bar
