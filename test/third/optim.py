from scipy.optimize import minimize
import numpy as np
from . import energies
from scipy.sparse.linalg import LinearOperator, cg
def flatten_configurations(configurations):
    # Exclude initial and final configurations (fixed)
    return np.concatenate([mesh.vertices.flatten() for mesh in configurations[1:-1]])

def reshape_configurations(flat_vertices, configurations):
    n_vertices = configurations[0].vertices.shape[0]
    reshaped_configs = [configurations[0]]  # Start with initial mesh

    idx = 0
    for _ in range(1, len(configurations) - 1):
        vertices = flat_vertices[idx:idx + n_vertices * 3].reshape((n_vertices, 3))
        mesh = configurations[0].copy()
        mesh.vertices = vertices
        reshaped_configs.append(mesh)
        idx += n_vertices * 3

    reshaped_configs.append(configurations[-1])  # End with final mesh
    return reshaped_configs

def trust_region_minimization(configurations):
    # Flatten configurations for optimization
    x0 = flatten_configurations(configurations)

    # Define the objective function
    def objective(x_flat):
        configs = reshape_configurations(x_flat, configurations)
        return energies.total_energy(configs)

    # Define the gradient
    def jacobian(x_flat):
        configs = reshape_configurations(x_flat, configurations)
        grads = energies.compute_gradient(configs)
        return np.concatenate([grad.flatten() for grad in grads])

    # Trust-region optimization
    result = minimize(
        objective,
        x0,
        method='trust-ncg',
        jac=jacobian,
        # hess=compute_hessian,  # If available
        options={'xtol': 1e-8, 'gtol': 1e-8, 'maxiter': 100}
    )

    # Update configurations with optimized values
    optimized_configs = reshape_configurations(result.x, configurations)
    return optimized_configs



def steihaug_cg(H, c, delta):
    # H: Hessian matrix (as a LinearOperator)
    # c: Gradient vector
    # delta: Trust-region radius

    def matvec(x):
        return H @ x

    linop = LinearOperator(H.shape, matvec=matvec)

    # Implement the CG method with trust-region constraint
    x, info = cg(linop, -c, maxiter=1000)
    if np.linalg.norm(x) <= delta:
        return x
    else:
        # Implement logic for stepping to the boundary of the trust region
        return x * (delta / np.linalg.norm(x))
