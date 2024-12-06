# repulsor/solvers.py

import scipy.sparse.linalg as spla

def conjugate_gradient(A, b, x0=None, tol=1e-5, maxiter=None):
    """Conjugate Gradient solver using SciPy."""
    x, info = spla.cg(A, b, x0=x0, tol=tol, maxiter=maxiter)
    return x, info

def gmres(A, b, x0=None, tol=1e-5, maxiter=None):
    """GMRES solver using SciPy."""
    x, info = spla.gmres(A, b, x0=x0, tol=tol, maxiter=maxiter)
    return x, info
