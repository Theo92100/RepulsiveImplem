

# Ajoutez le chemin absolu du répertoire contenant 'energies.py'
#sys.path.append('/Users/theoniemann/Desktop/MVA/GDA/Thèse/third')

import energies as e
from scipy.optimize import minimize
import numpy as np
import trimesh

mesh_initial = trimesh.load('../Meshes/hand/0_simplified.off')
gradientofphi = e.gradient_phi(mesh_initial)
print(gradientofphi)