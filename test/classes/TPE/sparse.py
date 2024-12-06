# repulsor/sparse.py

import scipy.sparse as sp

class SparseMatrices:
    """Class for sparse matrix operations using SciPy."""
    @staticmethod
    def create_sparse_matrix(data, indices, indptr, shape):
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    
    @staticmethod
    def sparse_dot(a, b):
        return a.dot(b)
    
    # Add more sparse operations as needed
