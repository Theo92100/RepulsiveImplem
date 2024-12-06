# repulsor/tensors.py

import numpy as np

class Tensors:
    """Class for tensor operations using NumPy."""
    @staticmethod
    def create_tensor(shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)
    
    @staticmethod
    def tensor_dot(a, b):
        return np.dot(a, b)
    
    # Add more tensor operations as needed
